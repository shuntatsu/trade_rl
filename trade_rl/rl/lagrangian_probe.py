"""Diagnostic canonical-action probe for constrained-policy training."""

from __future__ import annotations

import math
import multiprocessing as mp
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.actions import ActionMode, ActionSpec
from trade_rl.rl.environment_constraints import ConstraintCostVector
from trade_rl.rl.lagrangian import (
    CompletedEpisodeCostAccumulator,
    LagrangianSchema,
)
from trade_rl.rl.lagrangian_episode import classify_episode_completion

_TOLERANCE = 1e-12
_EVIDENCE_SCHEMA_VERSION = "canonical_action_probe_evidence_v1"


@dataclass(frozen=True, slots=True)
class _ProbeEpisodeResult:
    action_semantic: CanonicalActionSemantic
    action: np.ndarray
    numerators: Mapping[str, float]
    denominators: Mapping[str, int]
    censored_episode_count: int


_FORK_ENVIRONMENT_FACTORY: Callable[[], Any] | None = None
_FORK_SCHEMA: LagrangianSchema | None = None
_FORK_MAXIMUM_STEPS = 0


class CanonicalActionSemantic(str, Enum):
    """Meaning of the all-zero action under the maintained action contract."""

    TARGET_WEIGHT_CASH = "target_weight_cash"
    RESIDUAL_BASELINE = "residual_baseline"


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _finite_non_negative_mapping(
    values: Mapping[str, object],
    *,
    field_name: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    result: dict[str, float] = {}
    for name, raw_value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field_name} names must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(
            raw_value,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(f"{field_name} values must be numeric")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field_name} values must be finite and non-negative")
        result[name] = value
    return result


def _positive_denominator_mapping(
    values: Mapping[str, object],
) -> dict[str, int]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("denominators must be a non-empty mapping")
    result: dict[str, int] = {}
    for name, raw_value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("denominator names must be non-empty strings")
        result[name] = _positive_integer(raw_value, field_name=f"denominator {name}")
    return result


@dataclass(frozen=True, slots=True)
class CanonicalActionProbeEvidence:
    """Immutable warning evidence from stepping one canonical action."""

    action_semantic: CanonicalActionSemantic
    action: np.ndarray
    estimates: Mapping[str, float]
    denominators: Mapping[str, int]
    budgets: Mapping[str, float]
    violated_costs: tuple[str, ...]
    completed_episode_count: int
    censored_episode_count: int
    episode_count: int
    max_steps_per_episode: int
    warning: bool

    def __post_init__(self) -> None:
        try:
            semantic = CanonicalActionSemantic(self.action_semantic)
        except ValueError as error:
            raise ValueError("canonical action semantic is unsupported") from error
        action = np.asarray(self.action).reshape(-1).copy()
        if action.size == 0 or not np.issubdtype(action.dtype, np.floating):
            raise ValueError("canonical probe action must be a floating vector")
        if not np.isfinite(action).all():
            raise ValueError("canonical probe action must be finite")
        action.setflags(write=False)

        estimates = _finite_non_negative_mapping(
            self.estimates,
            field_name="estimates",
        )
        denominators = _positive_denominator_mapping(self.denominators)
        budgets = _finite_non_negative_mapping(self.budgets, field_name="budgets")
        if tuple(estimates) != tuple(denominators) or tuple(estimates) != tuple(
            budgets
        ):
            raise ValueError("probe evidence cost order is inconsistent")
        violated = tuple(self.violated_costs)
        if len(set(violated)) != len(violated) or any(
            name not in estimates for name in violated
        ):
            raise ValueError("violated costs must be unique known constraints")
        expected_violations = tuple(
            name for name in estimates if estimates[name] > budgets[name] + _TOLERANCE
        )
        if violated != expected_violations:
            raise ValueError("violated costs do not match estimates and budgets")
        completed = _positive_integer(
            self.completed_episode_count,
            field_name="completed_episode_count",
        )
        configured = _positive_integer(self.episode_count, field_name="episode_count")
        maximum_steps = _positive_integer(
            self.max_steps_per_episode,
            field_name="max_steps_per_episode",
        )
        if completed != configured:
            raise ValueError("completed probe episodes must equal configured episodes")
        if (
            isinstance(self.censored_episode_count, bool)
            or not isinstance(self.censored_episode_count, int)
            or self.censored_episode_count < 0
        ):
            raise ValueError("censored_episode_count must be a non-negative integer")
        if not isinstance(self.warning, bool) or self.warning != bool(violated):
            raise ValueError("warning must equal whether any constraint was violated")

        object.__setattr__(self, "action_semantic", semantic)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "denominators", denominators)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "violated_costs", violated)
        object.__setattr__(self, "completed_episode_count", completed)
        object.__setattr__(self, "episode_count", configured)
        object.__setattr__(self, "max_steps_per_episode", maximum_steps)

    def digest_payload(self) -> dict[str, object]:
        """Return deterministic JSON-compatible probe evidence."""

        return {
            "action": self.action.tolist(),
            "action_semantic": self.action_semantic.value,
            "budgets": dict(self.budgets),
            "censored_episode_count": self.censored_episode_count,
            "completed_episode_count": self.completed_episode_count,
            "denominators": dict(self.denominators),
            "episode_count": self.episode_count,
            "estimates": dict(self.estimates),
            "max_steps_per_episode": self.max_steps_per_episode,
            "schema_version": _EVIDENCE_SCHEMA_VERSION,
            "violated_costs": list(self.violated_costs),
            "warning": self.warning,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


def _resolve_canonical_action(
    environment: Any,
) -> tuple[CanonicalActionSemantic, np.ndarray]:
    unwrapped = getattr(environment, "unwrapped", environment)
    action_spec = getattr(unwrapped, "action_spec", None)
    if not isinstance(action_spec, ActionSpec):
        raise ValueError("canonical probe environment must expose an ActionSpec")
    action_space = getattr(environment, "action_space", None)
    if not isinstance(action_space, spaces.Box):
        raise ValueError("canonical probe requires a Box action space")
    shape = action_space.shape
    if shape is None or len(shape) != 1 or shape != (action_spec.size,):
        raise ValueError("canonical probe action space does not match ActionSpec")
    if not np.issubdtype(action_space.dtype, np.floating):
        raise ValueError("canonical probe action space must use a floating dtype")
    action = np.zeros(shape, dtype=action_space.dtype)
    semantic = (
        CanonicalActionSemantic.TARGET_WEIGHT_CASH
        if action_spec.mode is ActionMode.TARGET_WEIGHT
        else CanonicalActionSemantic.RESIDUAL_BASELINE
    )
    return semantic, action


def _elapsed_hours(info: Mapping[str, object], costs: ConstraintCostVector) -> float:
    explicit = info.get("transition_elapsed_hours")
    vector_elapsed = costs.transition_elapsed_hours
    if explicit is None and vector_elapsed is None:
        raise ValueError("info is missing transition_elapsed_hours")
    raw_value = explicit if explicit is not None else vector_elapsed
    if isinstance(raw_value, bool) or not isinstance(
        raw_value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("transition_elapsed_hours must be finite and positive")
    elapsed = float(raw_value)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("transition_elapsed_hours must be finite and positive")
    if vector_elapsed is not None and not math.isclose(
        elapsed,
        float(vector_elapsed),
        rel_tol=0.0,
        abs_tol=_TOLERANCE,
    ):
        raise ValueError("transition elapsed metadata mismatch")
    return elapsed


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a boolean")
    return bool(value)


def _run_probe_episode(
    environment_factory: Callable[[], Any],
    schema: LagrangianSchema,
    *,
    episode_index: int,
    maximum_steps: int,
) -> _ProbeEpisodeResult:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=schema)
    steps_for_episode = 0
    censored_count = 0
    canonical_semantic: CanonicalActionSemantic | None = None
    canonical_action: np.ndarray | None = None

    while steps_for_episode < maximum_steps:
        environment = environment_factory()
        if environment is None:
            raise ValueError("environment_factory returned null")
        try:
            semantic, action = _resolve_canonical_action(environment)
            if canonical_semantic is None:
                canonical_semantic = semantic
                canonical_action = action.copy()
            elif (
                semantic is not canonical_semantic
                or canonical_action is None
                or not np.array_equal(action, canonical_action)
            ):
                raise ValueError("canonical probe environment contract changed")

            environment.reset(seed=episode_index)
            attempt_finished = False
            while steps_for_episode < maximum_steps:
                transition = environment.step(action.copy())
                if not isinstance(transition, tuple) or len(transition) != 5:
                    raise ValueError(
                        "canonical probe step returned an invalid transition"
                    )
                _, _, raw_terminated, raw_truncated, raw_info = transition
                terminated = _boolean(raw_terminated, field_name="terminated")
                truncated = _boolean(raw_truncated, field_name="truncated")
                if not isinstance(raw_info, Mapping):
                    raise ValueError("canonical probe info must be a mapping")
                info = raw_info
                raw_costs = info.get("constraint_costs")
                if not isinstance(raw_costs, ConstraintCostVector):
                    raise ValueError(
                        "canonical probe info is missing valid constraint_costs"
                    )
                elapsed = _elapsed_hours(info, raw_costs)
                raw_time_limit = info.get("TimeLimit.truncated", truncated)
                time_limit = _boolean(
                    raw_time_limit,
                    field_name="TimeLimit.truncated",
                )
                completion_kind = classify_episode_completion(
                    terminated=terminated,
                    truncated=truncated,
                    time_limit_truncated=time_limit,
                    termination_reason=info.get("termination_reason"),
                )
                values = raw_costs.constraint_dict()
                cost_row = np.asarray(
                    [values[name] for name in schema.names],
                    dtype=np.float64,
                )
                batch = accumulator.ingest_rollout(
                    costs=cost_row.reshape(1, 1, -1),
                    elapsed_hours=np.asarray([[elapsed]], dtype=np.float64),
                    completion_kinds=np.asarray(
                        [[int(completion_kind)]],
                        dtype=np.int8,
                    ),
                )
                steps_for_episode += 1
                censored_count += batch.censored_episode_count
                if batch.completed_episode_count:
                    numerators: dict[str, float] = {}
                    denominators: dict[str, int] = {}
                    for name in schema.names:
                        estimate = batch.estimates[name]
                        if estimate is None:
                            raise RuntimeError(
                                "completed canonical probe estimate is missing"
                            )
                        numerators[name] = estimate.numerator
                        denominators[name] = estimate.denominator
                    if canonical_semantic is None or canonical_action is None:
                        raise RuntimeError("canonical probe did not resolve an action")
                    return _ProbeEpisodeResult(
                        action_semantic=canonical_semantic,
                        action=canonical_action,
                        numerators=numerators,
                        denominators=denominators,
                        censored_episode_count=censored_count,
                    )
                if batch.censored_episode_count:
                    attempt_finished = True
                    break
            if not attempt_finished and steps_for_episode >= maximum_steps:
                raise ValueError(
                    "canonical probe episode did not complete within the step limit"
                )
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()

    raise ValueError(
        "canonical probe did not obtain a valid completed episode within the step limit"
    )


def _run_forked_probe_episode(episode_index: int) -> _ProbeEpisodeResult:
    if _FORK_ENVIRONMENT_FACTORY is None or _FORK_SCHEMA is None:
        raise RuntimeError("forked canonical probe worker is not initialized")
    return _run_probe_episode(
        _FORK_ENVIRONMENT_FACTORY,
        _FORK_SCHEMA,
        episode_index=episode_index,
        maximum_steps=_FORK_MAXIMUM_STEPS,
    )


def _probe_episode_results(
    environment_factory: Callable[[], Any],
    schema: LagrangianSchema,
    *,
    episode_count: int,
    maximum_steps: int,
    max_workers: int,
) -> tuple[_ProbeEpisodeResult, ...]:
    episode_indices = tuple(range(episode_count))
    worker_count = min(max_workers, episode_count)
    if "fork" not in mp.get_all_start_methods():
        return tuple(
            _run_probe_episode(
                environment_factory,
                schema,
                episode_index=episode_index,
                maximum_steps=maximum_steps,
            )
            for episode_index in episode_indices
        )

    global _FORK_ENVIRONMENT_FACTORY, _FORK_MAXIMUM_STEPS, _FORK_SCHEMA
    _FORK_ENVIRONMENT_FACTORY = environment_factory
    _FORK_SCHEMA = schema
    _FORK_MAXIMUM_STEPS = maximum_steps
    try:
        context = mp.get_context("fork")
        # A full-market environment dirties enough inherited Python/NumPy pages
        # that running episodes in the parent or reusing a child steadily grows
        # RSS. Even with one worker, isolate and recycle every episode so its
        # allocator/COW state is returned to the OS.
        with context.Pool(processes=worker_count, maxtasksperchild=1) as pool:
            # map preserves episode/seed order, keeping floating-point aggregation
            # and the evidence digest identical to the serial implementation.
            return tuple(pool.map(_run_forked_probe_episode, episode_indices))
    finally:
        _FORK_ENVIRONMENT_FACTORY = None
        _FORK_SCHEMA = None
        _FORK_MAXIMUM_STEPS = 0


def run_canonical_action_feasibility_probe(
    *,
    environment_factory: Callable[[], Any],
    schema: LagrangianSchema,
    episode_count: int,
    max_steps_per_episode: int,
    max_workers: int = 1,
) -> CanonicalActionProbeEvidence:
    """Step the canonical zero action and return warning-only feasibility evidence."""

    if not callable(environment_factory):
        raise TypeError("environment_factory must be callable")
    if not isinstance(schema, LagrangianSchema):
        raise TypeError("schema must be a LagrangianSchema")
    required_episodes = _positive_integer(episode_count, field_name="episode_count")
    maximum_steps = _positive_integer(
        max_steps_per_episode,
        field_name="max_steps_per_episode",
    )
    worker_count = _positive_integer(max_workers, field_name="max_workers")
    results = _probe_episode_results(
        environment_factory,
        schema,
        episode_count=required_episodes,
        maximum_steps=maximum_steps,
        max_workers=worker_count,
    )
    pooled_numerators = {name: 0.0 for name in schema.names}
    pooled_denominators = {name: 0 for name in schema.names}
    censored_count = 0
    canonical_semantic: CanonicalActionSemantic | None = None
    canonical_action: np.ndarray | None = None
    for result in results:
        if canonical_semantic is None:
            canonical_semantic = result.action_semantic
            canonical_action = result.action.copy()
        elif (
            result.action_semantic is not canonical_semantic
            or canonical_action is None
            or not np.array_equal(result.action, canonical_action)
        ):
            raise ValueError("canonical probe environment contract changed")
        for name in schema.names:
            pooled_numerators[name] += result.numerators[name]
            pooled_denominators[name] += result.denominators[name]
        censored_count += result.censored_episode_count

    if canonical_semantic is None or canonical_action is None:
        raise RuntimeError("canonical probe did not resolve an action")
    estimates = {
        name: pooled_numerators[name] / pooled_denominators[name]
        for name in schema.names
    }
    budgets = {spec.name: spec.budget for spec in schema.specs}
    violated = tuple(
        name for name in schema.names if estimates[name] > budgets[name] + _TOLERANCE
    )
    return CanonicalActionProbeEvidence(
        action_semantic=canonical_semantic,
        action=canonical_action,
        estimates=estimates,
        denominators=pooled_denominators,
        budgets=budgets,
        violated_costs=violated,
        completed_episode_count=len(results),
        censored_episode_count=censored_count,
        episode_count=required_episodes,
        max_steps_per_episode=maximum_steps,
        warning=bool(violated),
    )


__all__ = [
    "CanonicalActionProbeEvidence",
    "CanonicalActionSemantic",
    "run_canonical_action_feasibility_probe",
]
