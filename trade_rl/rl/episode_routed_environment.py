"""Gym facade that routes complete episodes across isolated single-symbol envs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.instrument_episode_routing import (
    GENERIC_INSTRUMENT_ACTION_NAMES,
    GENERIC_INSTRUMENT_SYMBOLS,
    DeterministicBalancedInstrumentRouter,
    InstrumentDatasetBinding,
    InstrumentEpisodeBinding,
)

EPISODE_ROUTED_SINGLE_INSTRUMENT_ENVIRONMENT_SCHEMA = (
    "episode_routed_single_instrument_environment_v1"
)
_GENERIC_ACTION_SPEC_SCHEMA = "generic_single_instrument_action_v1"
_CHILD_RESET_SEED_SCHEMA = "episode_routed_child_reset_seed_v1"
_FORBIDDEN_RESET_OPTION_KEYS = frozenset(
    {
        "concrete_symbol",
        "dataset_id",
        "instrument",
        "symbol",
    }
)

Observation: TypeAlias = np.ndarray | dict[str, np.ndarray]


class _DatasetContract(Protocol):
    symbols: tuple[str, ...]
    dataset_id: str


class SingleInstrumentChildEnvironment(Protocol):
    """Runtime surface required from one isolated concrete-symbol child."""

    dataset: _DatasetContract
    action_spec: ActionSpec
    action_names: tuple[str, ...]
    action_spec_digest: str
    action_space: gym.spaces.Space[np.ndarray]
    observation_space: gym.spaces.Space[Any]
    observation_schema: str
    observation_contract_digest: str
    environment_digest: str

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]: ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


def _non_negative_info_index(info: Mapping[str, Any], key: str) -> int:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"child reset info {key} must be a non-negative integer")
    return value


def _same_space(
    left: gym.spaces.Space[Any],
    right: gym.spaces.Space[Any],
) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


class EpisodeRoutedSingleInstrumentEnv(gym.Env[Observation, np.ndarray]):
    """Expose one generic slot while selecting one concrete child per episode."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        child_environments: Mapping[str, SingleInstrumentChildEnvironment],
        *,
        bindings: Sequence[InstrumentDatasetBinding],
        run_seed: int,
        environment_index: int,
        partition_digest: str,
    ) -> None:
        super().__init__()
        children = dict(child_environments)
        if not children:
            raise ValueError("child environment mapping must not be empty")
        if any(not isinstance(symbol, str) or not symbol for symbol in children):
            raise ValueError("child environment symbols must be non-empty strings")
        if len({id(child) for child in children.values()}) != len(children):
            raise ValueError(
                "each concrete symbol requires an isolated child environment"
            )

        self.router = DeterministicBalancedInstrumentRouter(
            bindings,
            run_seed=run_seed,
            environment_index=environment_index,
            partition_digest=partition_digest,
        )
        binding_by_symbol = {
            binding.concrete_symbol: binding for binding in self.router.bindings
        }
        if frozenset(children) != frozenset(binding_by_symbol):
            raise ValueError("child environment and binding symbol closure mismatch")

        first_symbol = self.router.bindings[0].concrete_symbol
        first = children[first_symbol]
        self._validate_child(
            first_symbol,
            first,
            binding_by_symbol[first_symbol],
        )
        observation_space = first.observation_space
        observation_schema = first.observation_schema
        observation_contract_digest = first.observation_contract_digest
        action_space = first.action_space

        for binding in self.router.bindings[1:]:
            symbol = binding.concrete_symbol
            child = children[symbol]
            self._validate_child(symbol, child, binding)
            if not _same_space(observation_space, child.observation_space):
                raise ValueError("child observation spaces do not match")
            if observation_schema != child.observation_schema:
                raise ValueError("child observation schemas do not match")
            if observation_contract_digest != child.observation_contract_digest:
                raise ValueError("child observation contract digests do not match")
            if not _same_space(action_space, child.action_space):
                raise ValueError("child action spaces do not match")

        self._children = children
        self._bindings = binding_by_symbol
        self.symbols = GENERIC_INSTRUMENT_SYMBOLS
        self.action_names = GENERIC_INSTRUMENT_ACTION_NAMES
        self.action_spec = ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=1,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        )
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_schema = observation_schema
        self.observation_contract_digest = observation_contract_digest
        self.action_spec_digest = content_digest(
            {
                "schema_version": _GENERIC_ACTION_SPEC_SCHEMA,
                "mode": ActionMode.TARGET_WEIGHT.value,
                "symbols": self.symbols,
                "action_names": self.action_names,
                "shape": (1,),
                "validation_mode": ActionValidationMode.FAIL_CLOSED.value,
            }
        )
        self._environment_digest = content_digest(
            {
                "schema_version": (EPISODE_ROUTED_SINGLE_INSTRUMENT_ENVIRONMENT_SCHEMA),
                "router_digest": self.router.digest,
                "generic_symbols": self.symbols,
                "generic_action_names": self.action_names,
                "action_spec_digest": self.action_spec_digest,
                "observation_schema": self.observation_schema,
                "observation_contract_digest": self.observation_contract_digest,
                "child_runtime_identities": tuple(
                    (
                        binding.digest,
                        children[binding.concrete_symbol].environment_digest,
                    )
                    for binding in self.router.bindings
                ),
            }
        )
        self._completed_episode_count = 0
        self._active_child: SingleInstrumentChildEnvironment | None = None
        self._active_episode_binding: InstrumentEpisodeBinding | None = None
        self._episode_active = False
        self._failed_symbol: str | None = None
        self._closed = False

    @staticmethod
    def _validate_child(
        symbol: str,
        child: SingleInstrumentChildEnvironment,
        binding: InstrumentDatasetBinding,
    ) -> None:
        dataset = getattr(child, "dataset", None)
        if dataset is None:
            raise ValueError("child environment dataset is missing")
        if tuple(getattr(dataset, "symbols", ())) != (symbol,):
            raise ValueError("child dataset must contain exactly its concrete symbol")
        if getattr(dataset, "dataset_id", None) != binding.symbol_dataset_digest:
            raise ValueError("child dataset identity does not match its binding")
        action_spec = getattr(child, "action_spec", None)
        if not isinstance(action_spec, ActionSpec):
            raise ValueError("child action specification is missing")
        if action_spec.mode is not ActionMode.TARGET_WEIGHT:
            raise ValueError("child must use direct target-weight action mode")
        if action_spec.target_weight_count != 1 or action_spec.size != 1:
            raise ValueError("child target-weight action must have shape (1,)")
        expected_action_names = (f"target_weight:{symbol}",)
        if tuple(getattr(child, "action_names", ())) != expected_action_names:
            raise ValueError("child concrete action names do not match its symbol")
        action_space = getattr(child, "action_space", None)
        if getattr(action_space, "shape", None) != (1,):
            raise ValueError("child action space must have shape (1,)")
        for field_name in (
            "action_spec_digest",
            "environment_digest",
            "observation_contract_digest",
            "observation_schema",
            "observation_space",
        ):
            if getattr(child, field_name, None) is None:
                raise ValueError(f"child {field_name} is missing")

    @property
    def environment_digest(self) -> str:
        return self._environment_digest

    @property
    def completed_episode_count(self) -> int:
        return self._completed_episode_count

    @property
    def active_episode_binding(self) -> InstrumentEpisodeBinding:
        binding = self._active_episode_binding
        if binding is None:
            raise RuntimeError("environment has no active or completed episode binding")
        return binding

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("episode-routed environment is closed")
        if self._failed_symbol is not None:
            raise RuntimeError(
                "episode-routed environment is fail-closed after child failure for "
                f"{self._failed_symbol}"
            )

    def _child_reset_seed(self, *, user_seed: int | None) -> int:
        route = self.router.route(self._completed_episode_count)
        if user_seed is not None and (
            isinstance(user_seed, bool)
            or not isinstance(user_seed, int)
            or user_seed < 0
        ):
            raise ValueError("reset seed must be a non-negative integer or null")
        digest = content_digest(
            {
                "schema_version": _CHILD_RESET_SEED_SCHEMA,
                "router_digest": self.router.digest,
                "binding_digest": route.binding.digest,
                "completed_episode_count": route.completed_episode_count,
                "user_seed": user_seed,
            }
        )
        return int(digest[:8], 16)

    @staticmethod
    def _validate_reset_options(options: Mapping[str, Any]) -> None:
        forbidden = sorted(_FORBIDDEN_RESET_OPTION_KEYS.intersection(options))
        if forbidden:
            raise ValueError(
                "reset options cannot select or override concrete symbol fields: "
                f"{forbidden}"
            )

    @staticmethod
    def _enrich_info(
        info: Mapping[str, Any],
        binding: InstrumentEpisodeBinding,
        *,
        terminal: bool,
    ) -> dict[str, Any]:
        payload = binding.to_json_dict()
        enriched = dict(info)
        enriched.update(
            {
                "generic_symbols": GENERIC_INSTRUMENT_SYMBOLS,
                "generic_action_names": GENERIC_INSTRUMENT_ACTION_NAMES,
                "instrument_episode_binding": payload,
            }
        )
        if terminal:
            enriched["terminal_instrument_episode_binding"] = payload
        return enriched

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        self._ensure_open()
        if self._episode_active:
            raise RuntimeError("cannot reset while an instrument episode is active")
        resolved_options = dict(options or {})
        self._validate_reset_options(resolved_options)
        super().reset(seed=seed)
        route = self.router.route(self._completed_episode_count)
        symbol = route.binding.concrete_symbol
        child = self._children[symbol]
        child_seed = self._child_reset_seed(user_seed=seed)
        try:
            observation, child_info = child.reset(
                seed=child_seed,
                options=resolved_options or None,
            )
        except Exception:
            self._failed_symbol = symbol
            raise
        if not isinstance(child_info, Mapping):
            self._failed_symbol = symbol
            raise ValueError("child reset info must be a mapping")
        episode_start = _non_negative_info_index(child_info, "start_index")
        episode_stop = _non_negative_info_index(child_info, "end_index")
        try:
            binding = InstrumentEpisodeBinding.from_route(
                route,
                episode_start=episode_start,
                episode_stop=episode_stop,
            )
        except Exception:
            self._failed_symbol = symbol
            raise
        self._active_child = child
        self._active_episode_binding = binding
        self._episode_active = True
        return observation, self._enrich_info(child_info, binding, terminal=False)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._ensure_open()
        if (
            not self._episode_active
            or self._active_child is None
            or self._active_episode_binding is None
        ):
            raise RuntimeError("reset must start an instrument episode before step")
        vector = np.asarray(action)
        if vector.shape != (1,):
            raise ValueError("episode-routed action must have exact shape (1,)")
        parsed = self.action_spec.parse(
            vector,
            mode=ActionValidationMode.FAIL_CLOSED,
        )
        maintained_action = parsed.as_array()
        try:
            observation, reward, terminated, truncated, child_info = (
                self._active_child.step(maintained_action)
            )
        except Exception:
            self._failed_symbol = (
                self._active_episode_binding.dataset_binding.concrete_symbol
            )
            self._episode_active = False
            raise
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            self._failed_symbol = (
                self._active_episode_binding.dataset_binding.concrete_symbol
            )
            self._episode_active = False
            raise ValueError("child terminal flags must be booleans")
        if terminated and truncated:
            self._failed_symbol = (
                self._active_episode_binding.dataset_binding.concrete_symbol
            )
            self._episode_active = False
            raise RuntimeError("terminated and truncated must remain exclusive")
        if not isinstance(child_info, Mapping):
            self._failed_symbol = (
                self._active_episode_binding.dataset_binding.concrete_symbol
            )
            self._episode_active = False
            raise ValueError("child step info must be a mapping")
        terminal = terminated or truncated
        info = self._enrich_info(
            child_info,
            self._active_episode_binding,
            terminal=terminal,
        )
        if terminal:
            self._episode_active = False
            self._completed_episode_count += 1
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        for child in self._children.values():
            child.close()
        self._closed = True
        self._episode_active = False


__all__ = [
    "EPISODE_ROUTED_SINGLE_INSTRUMENT_ENVIRONMENT_SCHEMA",
    "EpisodeRoutedSingleInstrumentEnv",
    "SingleInstrumentChildEnvironment",
]
