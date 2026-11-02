"""Opt-in write-once journal wrapper for shared-portfolio Gym episodes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes, to_json_value
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv

PORTFOLIO_EPISODE_JOURNAL_SCHEMA = "shared_portfolio_episode_journal_v2"
_DECISION_TRACE_SCHEMA = "shared_portfolio_decision_trace_v1"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_ACTION_VALUES = 256


@dataclass(frozen=True, slots=True)
class PortfolioEpisodeEventReference:
    """Immutable pointer to one event written by the episode journal."""

    sequence: int
    relative_path: str
    sha256: str


def _action_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"type": "non_finite_float", "value": value.hex()}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _action_descriptor(value: object) -> dict[str, object]:
    if isinstance(value, np.ndarray):
        descriptor: dict[str, object] = {
            "container": "numpy.ndarray",
            "dtype": value.dtype.name,
            "shape": [int(size) for size in value.shape],
        }
        if value.size <= _MAX_ACTION_VALUES:
            descriptor["values"] = _action_values(value.tolist())
        else:
            descriptor["values_omitted"] = True
        return descriptor
    if isinstance(value, (tuple, list)):
        return {
            "container": type(value).__name__,
            "values": _action_values(value),
        }
    return {"container": "scalar", "value": _action_scalar(value)}


def _action_values(value: object) -> object:
    if isinstance(value, np.generic):
        return _action_scalar(value)
    if isinstance(value, np.ndarray):
        if value.size > _MAX_ACTION_VALUES:
            return {"type": "array", "shape": [int(size) for size in value.shape]}
        return _action_values(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_action_values(item) for item in value]
    if isinstance(value, Mapping):
        return {"type": "mapping", "size": len(value)}
    return _action_scalar(value)


def _strict_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a built-in integer")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a built-in boolean")
    return value


def _strict_finite(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real non-boolean number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _environment_context(env: SharedPortfolioGymEnv) -> dict[str, object]:
    profile = env.market_order_profile
    return {
        "observation_schema": env.observation_schema,
        "dataset_id": env.dataset.dataset_id,
        "symbols": list(env.dataset.symbols),
        "start_index": env.start_index,
        "stop_index": env.stop_index,
        "initial_capital": env.initial_capital,
        "target_abs_weights": list(env.target_abs_weights),
        "bound_digest": env.bound.digest,
        "execution_cost": to_json_value(env.execution_cost),
        "risk_config": to_json_value(env.risk_config),
        "market_order_profile_digest": None if profile is None else profile.digest,
    }


class PortfolioEpisodeJournalWrapper(
    gym.Wrapper[
        dict[str, np.ndarray],
        np.ndarray,
        dict[str, np.ndarray],
        np.ndarray,
    ]
):
    """Persist source action/reward/endpoint facts before returning transitions.

    The wrapper is a single-writer software journal. It does not save policy
    logits, model state, or observations and does not claim rollback or
    power-loss-wide transaction guarantees.
    """

    def __init__(
        self,
        env: gym.Env[dict[str, np.ndarray], np.ndarray],
        *,
        root: str | Path,
        declared_context_digest: str,
    ) -> None:
        super().__init__(env)
        if not isinstance(self.unwrapped, SharedPortfolioGymEnv):
            raise TypeError("env must unwrap to SharedPortfolioGymEnv")
        if type(declared_context_digest) is not str or not _DIGEST_RE.fullmatch(
            declared_context_digest
        ):
            raise ValueError("declared_context_digest must be a lowercase SHA-256")

        self.root = Path(root).expanduser().absolute()
        if self.root.exists() or self.root.is_symlink():
            raise FileExistsError(f"episode journal root already exists: {self.root}")
        self.root.mkdir(parents=False, exist_ok=False)
        self._events_root = self.root / "events"
        self._events_root.mkdir()
        self._event_sequence = 0
        self._next_episode_index = 0
        self._episode_index: int | None = None
        self._step_index = 0
        self._last_current_index: int | None = None
        self._episode_open = False
        self._poisoned = False
        self._write_failed = False
        self._latest_step_reference: PortfolioEpisodeEventReference | None = None

        base_env = cast(SharedPortfolioGymEnv, self.unwrapped)
        manifest = {
            "schema": PORTFOLIO_EPISODE_JOURNAL_SCHEMA,
            "declared_context_digest": declared_context_digest,
            "context_digest_semantics": "caller-declared; not independently attested",
            "environment_context": _environment_context(base_env),
            "recorded_content": [
                "submitted_action",
                "source_action_codes",
                "proposal_and_approved_targets_with_native_execution_events",
                "reward",
                "termination_flags",
                "native_interval_and_terminal_facts",
            ],
        }
        atomic_write_bytes(self.root / "manifest.json", canonical_json_bytes(manifest))

    @property
    def latest_step_reference(self) -> PortfolioEpisodeEventReference | None:
        return self._latest_step_reference

    def _append_event(
        self, event: str, data: dict[str, object]
    ) -> PortfolioEpisodeEventReference:
        episode_index = self._episode_index
        sequence = self._event_sequence
        record = {
            "schema": PORTFOLIO_EPISODE_JOURNAL_SCHEMA,
            "sequence": sequence,
            "episode_index": episode_index,
            "event": event,
            "data": data,
        }
        path = self._events_root / f"event-{sequence:08d}.json"
        try:
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"journal event path already exists: {path}")
            payload = canonical_json_bytes(record)
            atomic_write_bytes(path, payload)
        except BaseException:
            self._write_failed = True
            self._poisoned = True
            raise
        self._event_sequence += 1
        return PortfolioEpisodeEventReference(
            sequence=sequence,
            relative_path=path.relative_to(self.root).as_posix(),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def _require_writable(self) -> None:
        if self._poisoned or self._write_failed:
            raise RuntimeError("episode journal wrapper is poisoned")

    def _episode_cutoff(self, *, reason: str) -> None:
        if not self._episode_open:
            return
        self._append_event(
            "episode_interrupted",
            {
                "reason": reason,
                "last_recorded_current_index": self._last_current_index,
                "completed": False,
                "settlement_claimed": False,
            },
        )
        self._episode_open = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        self._require_writable()
        if self._episode_open:
            self._episode_cutoff(reason="manual_reset")

        self._episode_index = self._next_episode_index
        self._next_episode_index += 1
        self._step_index = 0
        self._last_current_index = None
        self._episode_open = True
        self._append_event(
            "episode_started",
            {"seed": _action_descriptor(seed), "options": _action_descriptor(options)},
        )
        try:
            observation, info = self.env.reset(seed=seed, options=options)
        except BaseException as error:
            self._record_call_failure("episode_reset_failed", error)
            raise

        try:
            current_index = _strict_int(
                info.get("current_index"), field="current_index"
            )
            reset_facts = {
                "current_index": current_index,
                "observation_schema": info.get("observation_schema"),
                "execution_seed": info.get("execution_seed"),
            }
            canonical_json_bytes(reset_facts)
            self._append_event("episode_reset", reset_facts)
        except BaseException:
            self._poisoned = True
            raise
        self._last_current_index = current_index
        return observation, info

    def _record_call_failure(self, event: str, error: BaseException) -> None:
        self._poisoned = True
        failure = {
            "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "error_message": str(error),
            "native_result_returned": False,
            "native_state_outcome": "unknown",
        }
        try:
            self._append_event(event, failure)
        except BaseException as journal_error:
            error.add_note(
                "The journal could not persist the secondary failure record: "
                f"{type(journal_error).__name__}: {journal_error}"
            )

    @staticmethod
    def _result_facts(
        reward: object,
        terminated: object,
        truncated: object,
        info: Mapping[str, object],
    ) -> tuple[float, bool, bool, dict[str, object]]:
        resolved_reward = _strict_finite(reward, field="reward")
        resolved_terminated = _strict_bool(terminated, field="terminated")
        resolved_truncated = _strict_bool(truncated, field="truncated")
        raw_action = info.get("raw_action")
        if type(raw_action) not in (tuple, list) or any(
            type(value) is not int for value in cast(Sequence[object], raw_action)
        ):
            raise TypeError("source raw_action must be an integer sequence")
        current_index = _strict_int(info.get("current_index"), field="current_index")
        elapsed_intervals = _strict_int(
            info.get("elapsed_intervals"), field="elapsed_intervals"
        )
        settlement_complete = _strict_bool(
            info.get("settlement_complete"), field="settlement_complete"
        )
        endpoint_valid = _strict_bool(
            info.get("objective_endpoint_valid"),
            field="objective_endpoint_valid",
        )
        terminal_state = info.get("terminal_state")
        if terminal_state is not None and type(terminal_state) is not dict:
            raise TypeError("terminal_state must be a dict or None")
        raw_target_weights = info.get("target_weights")
        if type(raw_target_weights) not in (tuple, list):
            raise TypeError("source target_weights must be a sequence")
        raw_effective_holds = info.get("effective_hold_symbols")
        if type(raw_effective_holds) not in (tuple, list):
            raise TypeError("source effective_hold_symbols must be a sequence")
        raw_decision_trace = info.get("decision_trace")
        if type(raw_decision_trace) is not dict:
            raise TypeError("source decision_trace must be a dict")
        if raw_decision_trace.get("schema") != _DECISION_TRACE_SCHEMA:
            raise ValueError("source decision_trace schema is unsupported")
        decision_trace = json.loads(canonical_json_bytes(raw_decision_trace))
        if type(decision_trace) is not dict:
            raise TypeError("source decision_trace must canonicalize to a dict")
        if decision_trace.get("raw_action_codes") != list(
            cast(Sequence[int], raw_action)
        ):
            raise ValueError("decision_trace action codes differ from source action")
        facts = {
            "raw_action_codes": list(cast(Sequence[int], raw_action)),
            "decision_trace": decision_trace,
            "current_index": current_index,
            "elapsed_intervals": elapsed_intervals,
            "target_weights": [
                _strict_finite(value, field="target_weights")
                for value in cast(Sequence[object], raw_target_weights)
            ],
            "effective_hold_symbols": [
                _strict_int(value, field="effective_hold_symbols")
                for value in cast(Sequence[object], raw_effective_holds)
            ],
            "settlement_complete": settlement_complete,
            "objective_endpoint_valid": endpoint_valid,
            "terminal_state": terminal_state,
        }
        if decision_trace.get("proposal_weights") != facts["target_weights"]:
            raise ValueError("decision_trace proposal differs from source target")
        canonical_json_bytes(facts)
        return resolved_reward, resolved_terminated, resolved_truncated, facts

    def step(
        self, action: object
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        self._require_writable()
        if not self._episode_open:
            raise RuntimeError("reset must be called before step")
        submitted_action = _action_descriptor(action)
        step_index = self._step_index
        self._step_index += 1
        self._append_event(
            "step_started",
            {"step_index": step_index, "submitted_action": submitted_action},
        )
        try:
            observation, reward, terminated, truncated, info = self.env.step(
                cast(np.ndarray, action)
            )
        except BaseException as error:
            self._record_call_failure(
                "step_failed",
                error,
            )
            raise

        self._poisoned = True
        try:
            if not isinstance(info, Mapping):
                raise TypeError("source step info must be a mapping")
            resolved_reward, resolved_terminated, resolved_truncated, facts = (
                self._result_facts(reward, terminated, truncated, info)
            )
            data = {
                "step_index": step_index,
                "submitted_action": submitted_action,
                "reward": resolved_reward,
                "terminated": resolved_terminated,
                "truncated": resolved_truncated,
                **facts,
            }
            self._latest_step_reference = self._append_event("step_result", data)
        except BaseException:
            self._poisoned = True
            raise

        self._poisoned = False
        self._last_current_index = cast(int, facts["current_index"])
        if resolved_terminated or resolved_truncated:
            self._episode_open = False
        return (
            observation,
            resolved_reward,
            resolved_terminated,
            resolved_truncated,
            info,
        )

    def close(self) -> None:
        if not self._write_failed and self._episode_open:
            try:
                self._episode_cutoff(reason="close")
            except BaseException:
                self._poisoned = True
                raise
        super().close()


__all__ = [
    "PORTFOLIO_EPISODE_JOURNAL_SCHEMA",
    "PortfolioEpisodeEventReference",
    "PortfolioEpisodeJournalWrapper",
]
