"""Typed Lagrangian constraint schemas and completed-episode statistics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.constraint_contracts import (
    CONSTRAINT_COST_NAMES,
    ConstraintAggregation,
    canonical_constraint_aggregation,
    canonical_constraint_unit,
)
from trade_rl.domain.constraint_contracts import (
    validate_constraint_name as _validate_constraint_name,
)
from trade_rl.rl.lagrangian_episode import EpisodeCompletionKind


@dataclass(frozen=True, slots=True)
class LagrangianConstraintSpec:
    """Independent budget and stabilized dual-update settings for one cost."""

    name: str
    aggregation: ConstraintAggregation
    budget: float
    dual_learning_rate: float
    ema_beta: float
    initial_multiplier: float
    max_multiplier: float
    warmup_rollouts: int
    update_interval_rollouts: int

    def __post_init__(self) -> None:
        _validate_constraint_name(self.name)
        if not isinstance(self.aggregation, ConstraintAggregation):
            raise ValueError("aggregation must be a ConstraintAggregation")
        expected_aggregation = canonical_constraint_aggregation(self.name)
        if self.aggregation is not expected_aggregation:
            raise ValueError(
                f"aggregation mismatch for {self.name}: "
                f"expected {expected_aggregation.value}"
            )

        for field_name in (
            "budget",
            "dual_learning_rate",
            "ema_beta",
            "initial_multiplier",
            "max_multiplier",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)

        if self.budget < 0.0:
            raise ValueError("budget must be non-negative")
        if self.dual_learning_rate <= 0.0:
            raise ValueError("dual_learning_rate must be positive")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("ema_beta must be within [0, 1)")
        if self.initial_multiplier < 0.0:
            raise ValueError("initial_multiplier must be non-negative")
        if self.max_multiplier <= 0.0:
            raise ValueError("max_multiplier must be positive")
        if self.initial_multiplier > self.max_multiplier:
            raise ValueError("initial_multiplier cannot exceed max_multiplier")

        for field_name, minimum in (
            ("warmup_rollouts", 0),
            ("update_interval_rollouts", 1),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{field_name} must be a {qualifier} integer")

    def digest_payload(self) -> dict[str, object]:
        return {
            "aggregation": self.aggregation.value,
            "budget": self.budget,
            "dual_learning_rate": self.dual_learning_rate,
            "ema_beta": self.ema_beta,
            "initial_multiplier": self.initial_multiplier,
            "max_multiplier": self.max_multiplier,
            "name": self.name,
            "unit": canonical_constraint_unit(self.name),
            "update_interval_rollouts": self.update_interval_rollouts,
            "warmup_rollouts": self.warmup_rollouts,
        }


@dataclass(frozen=True, slots=True)
class LagrangianSchema:
    """Ordered constraint schema included in training and checkpoint identity."""

    specs: tuple[LagrangianConstraintSpec, ...]

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        if not specs:
            raise ValueError("Lagrangian schema must not be empty")
        if any(not isinstance(spec, LagrangianConstraintSpec) for spec in specs):
            raise ValueError("Lagrangian schema requires constraint specs")
        names = tuple(spec.name for spec in specs)
        if len(set(names)) != len(names):
            raise ValueError("Lagrangian schema contains duplicate constraint names")
        canonical_indices = tuple(CONSTRAINT_COST_NAMES.index(name) for name in names)
        if canonical_indices != tuple(sorted(canonical_indices)):
            raise ValueError("Lagrangian schema must preserve canonical order")
        object.__setattr__(self, "specs", specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def __getitem__(self, name: str) -> LagrangianConstraintSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def digest_payload(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "specs": [spec.digest_payload() for spec in self.specs],
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


@dataclass(frozen=True, slots=True)
class ConstraintEstimate:
    """One pooled completed-episode estimate for a maintained constraint."""

    name: str
    numerator: float
    denominator: int

    def __post_init__(self) -> None:
        _validate_constraint_name(self.name)
        numerator = float(self.numerator)
        if not math.isfinite(numerator) or numerator < 0.0:
            raise ValueError(
                "constraint estimate numerator must be finite and non-negative"
            )
        if (
            isinstance(self.denominator, bool)
            or not isinstance(self.denominator, int)
            or self.denominator <= 0
        ):
            raise ValueError("constraint estimate denominator must be positive")
        object.__setattr__(self, "numerator", numerator)

    @property
    def value(self) -> float:
        value = self.numerator / self.denominator
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("constraint estimate value became invalid")
        return value


@dataclass(frozen=True, slots=True)
class CompletedEpisodeBatch:
    """Completed and censored episode statistics produced by one rollout."""

    estimates: dict[str, ConstraintEstimate | None]
    completed_episode_count: int
    censored_episode_count: int

    def __post_init__(self) -> None:
        estimates = dict(self.estimates)
        if not estimates:
            raise ValueError("completed episode estimates must not be empty")
        for name, estimate in estimates.items():
            _validate_constraint_name(name)
            if estimate is not None and not isinstance(estimate, ConstraintEstimate):
                raise TypeError("completed episode estimate has an invalid type")
            if estimate is not None and estimate.name != name:
                raise ValueError(f"completed episode estimate name mismatch for {name}")
        for field_name in ("completed_episode_count", "censored_episode_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.completed_episode_count == 0 and any(
            estimate is not None for estimate in estimates.values()
        ):
            raise ValueError("zero completed episodes cannot produce estimates")
        if self.completed_episode_count > 0 and any(
            estimate is None for estimate in estimates.values()
        ):
            raise ValueError("completed episodes require every constraint estimate")
        object.__setattr__(self, "estimates", estimates)


class CompletedEpisodeCostAccumulator:
    """Aggregate time-aware completed episodes across rollout boundaries."""

    _STATE_VERSION = "completed_episode_cost_accumulator_v2"
    _EVENT_TOLERANCE = 1e-12

    def __init__(self, *, n_envs: int, schema: LagrangianSchema) -> None:
        if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs <= 0:
            raise ValueError("n_envs must be a positive integer")
        if not isinstance(schema, LagrangianSchema):
            raise TypeError("schema must be a LagrangianSchema")
        self.n_envs = n_envs
        self.schema = schema
        shape = (n_envs, len(schema.names))
        self._episode_cost_sums = np.zeros(shape, dtype=np.float64)
        self._episode_time_weighted_sums = np.zeros(shape, dtype=np.float64)
        self._episode_elapsed_hours = np.zeros(n_envs, dtype=np.float64)
        self._episode_step_counts = np.zeros(n_envs, dtype=np.int64)

    @property
    def state_version(self) -> str:
        """Return the checkpoint schema version for unfinished episodes."""

        return self._STATE_VERSION

    def _validated_rollout(
        self,
        *,
        costs: np.ndarray,
        elapsed_hours: np.ndarray,
        completion_kinds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cost_array = np.asarray(costs, dtype=np.float64)
        expected_suffix = (self.n_envs, len(self.schema.names))
        if cost_array.ndim != 3 or cost_array.shape[1:] != expected_suffix:
            raise ValueError(
                "cost rollout shape must be [steps, n_envs, n_costs] "
                f"with suffix {expected_suffix}"
            )
        if cost_array.shape[0] <= 0:
            raise ValueError("cost rollout must contain at least one step")
        if not np.all(np.isfinite(cost_array)) or np.any(cost_array < 0.0):
            raise ValueError("cost rollout values must be finite and non-negative")

        metadata_shape = (cost_array.shape[0], self.n_envs)
        elapsed_array = np.asarray(elapsed_hours, dtype=np.float64)
        if elapsed_array.shape != metadata_shape:
            raise ValueError(f"elapsed hours must have shape {metadata_shape}")
        if not np.all(np.isfinite(elapsed_array)) or np.any(elapsed_array <= 0.0):
            raise ValueError("elapsed hours must be finite and positive")

        raw_kinds = np.asarray(completion_kinds)
        if raw_kinds.shape != metadata_shape:
            raise ValueError(f"completion kinds must have shape {metadata_shape}")
        if not np.issubdtype(raw_kinds.dtype, np.integer):
            raise ValueError("completion kind values must be integers")
        kind_array = raw_kinds.astype(np.int8, copy=False)
        valid_kinds = {int(kind) for kind in EpisodeCompletionKind}
        if any(int(value) not in valid_kinds for value in kind_array.flat):
            raise ValueError("completion kind value is unknown")

        for index, spec in enumerate(self.schema.specs):
            if spec.aggregation is ConstraintAggregation.EPISODE_EVENT_RATE and np.any(
                cost_array[:, :, index] > 1.0 + self._EVENT_TOLERANCE
            ):
                raise ValueError(f"event cost {spec.name} must be within [0, 1]")
        return cost_array, elapsed_array, kind_array

    def _validate_event_totals(
        self,
        cost_sums: np.ndarray,
        *,
        env_index: int,
    ) -> None:
        for cost_index, spec in enumerate(self.schema.specs):
            if spec.aggregation is not ConstraintAggregation.EPISODE_EVENT_RATE:
                continue
            if cost_sums[env_index, cost_index] > 1.0 + self._EVENT_TOLERANCE:
                raise ValueError(
                    f"event cost {spec.name} occurred more than once within one episode"
                )

    @staticmethod
    def _clear_environment(
        *,
        env_index: int,
        cost_sums: np.ndarray,
        time_weighted_sums: np.ndarray,
        elapsed_hours: np.ndarray,
        step_counts: np.ndarray,
    ) -> None:
        cost_sums[env_index].fill(0.0)
        time_weighted_sums[env_index].fill(0.0)
        elapsed_hours[env_index] = 0.0
        step_counts[env_index] = 0

    def ingest_rollout(
        self,
        *,
        costs: np.ndarray,
        elapsed_hours: np.ndarray,
        completion_kinds: np.ndarray,
    ) -> CompletedEpisodeBatch:
        """Consume aligned rollout metadata and estimate valid completed episodes."""

        cost_array, elapsed_array, kind_array = self._validated_rollout(
            costs=costs,
            elapsed_hours=elapsed_hours,
            completion_kinds=completion_kinds,
        )
        cost_sums = self._episode_cost_sums.copy()
        time_weighted_sums = self._episode_time_weighted_sums.copy()
        episode_elapsed = self._episode_elapsed_hours.copy()
        step_counts = self._episode_step_counts.copy()
        numerators = np.zeros(len(self.schema.names), dtype=np.float64)
        completed_episode_count = 0
        censored_episode_count = 0

        for step in range(cost_array.shape[0]):
            for env_index in range(self.n_envs):
                transition_costs = cost_array[step, env_index]
                transition_hours = float(elapsed_array[step, env_index])
                cost_sums[env_index] += transition_costs
                time_weighted_sums[env_index] += transition_costs * (
                    transition_hours / 24.0
                )
                episode_elapsed[env_index] += transition_hours
                step_counts[env_index] += 1

                kind = EpisodeCompletionKind(int(kind_array[step, env_index]))
                if kind is EpisodeCompletionKind.NONE:
                    continue
                self._validate_event_totals(cost_sums, env_index=env_index)
                if kind is EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION:
                    censored_episode_count += 1
                    self._clear_environment(
                        env_index=env_index,
                        cost_sums=cost_sums,
                        time_weighted_sums=time_weighted_sums,
                        elapsed_hours=episode_elapsed,
                        step_counts=step_counts,
                    )
                    continue

                episode_steps = int(step_counts[env_index])
                episode_hours = float(episode_elapsed[env_index])
                if episode_steps <= 0 or episode_hours <= 0.0:
                    raise RuntimeError("completed episode has invalid support")
                episode_days = episode_hours / 24.0
                for cost_index, spec in enumerate(self.schema.specs):
                    raw_sum = float(cost_sums[env_index, cost_index])
                    weighted_sum = float(time_weighted_sums[env_index, cost_index])
                    if spec.aggregation is ConstraintAggregation.EPISODE_TIME_AREA:
                        contribution = weighted_sum
                    elif (
                        spec.aggregation is ConstraintAggregation.EPISODE_DECISION_MEAN
                    ):
                        contribution = raw_sum / episode_steps
                    elif (
                        spec.aggregation
                        is ConstraintAggregation.EPISODE_TIME_WEIGHTED_MEAN
                    ):
                        contribution = weighted_sum / episode_days
                    elif spec.aggregation is ConstraintAggregation.EPISODE_SUM:
                        contribution = raw_sum
                    else:
                        contribution = min(raw_sum, 1.0)
                    if not math.isfinite(contribution) or contribution < 0.0:
                        raise RuntimeError("constraint aggregation became invalid")
                    numerators[cost_index] += contribution

                completed_episode_count += 1
                self._clear_environment(
                    env_index=env_index,
                    cost_sums=cost_sums,
                    time_weighted_sums=time_weighted_sums,
                    elapsed_hours=episode_elapsed,
                    step_counts=step_counts,
                )

        self._episode_cost_sums = cost_sums
        self._episode_time_weighted_sums = time_weighted_sums
        self._episode_elapsed_hours = episode_elapsed
        self._episode_step_counts = step_counts
        estimates: dict[str, ConstraintEstimate | None]
        if completed_episode_count == 0:
            estimates = {name: None for name in self.schema.names}
        else:
            estimates = {
                name: ConstraintEstimate(
                    name=name,
                    numerator=float(numerators[index]),
                    denominator=completed_episode_count,
                )
                for index, name in enumerate(self.schema.names)
            }
        return CompletedEpisodeBatch(
            estimates=estimates,
            completed_episode_count=completed_episode_count,
            censored_episode_count=censored_episode_count,
        )

    def state_dict(self) -> dict[str, object]:
        """Return JSON-compatible unfinished-episode sufficient statistics."""

        return {
            "cost_names": list(self.schema.names),
            "episode_cost_sums": self._episode_cost_sums.tolist(),
            "episode_elapsed_hours": self._episode_elapsed_hours.tolist(),
            "episode_step_counts": self._episode_step_counts.tolist(),
            "episode_time_weighted_sums": self._episode_time_weighted_sums.tolist(),
            "n_envs": self.n_envs,
            "schema_digest": self.schema.digest,
            "schema_version": self._STATE_VERSION,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore unfinished episodes only when identity and statistics match."""

        if state.get("schema_version") != self._STATE_VERSION:
            raise ValueError("accumulator state schema version mismatch")
        raw_cost_names = state.get("cost_names")
        if not isinstance(raw_cost_names, (list, tuple)) or not all(
            isinstance(name, str) for name in raw_cost_names
        ):
            raise ValueError("accumulator state schema mismatch")
        if (
            state.get("schema_digest") != self.schema.digest
            or tuple(raw_cost_names) != self.schema.names
        ):
            raise ValueError("accumulator state schema mismatch")
        if state.get("n_envs") != self.n_envs:
            raise ValueError("accumulator state environment count mismatch")

        try:
            cost_sums = np.asarray(state["episode_cost_sums"], dtype=np.float64)
            weighted_sums = np.asarray(
                state["episode_time_weighted_sums"], dtype=np.float64
            )
            elapsed_hours = np.asarray(state["episode_elapsed_hours"], dtype=np.float64)
            raw_step_counts = np.asarray(state["episode_step_counts"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("accumulator state payload is invalid") from error

        expected_matrix_shape = self._episode_cost_sums.shape
        expected_vector_shape = self._episode_step_counts.shape
        if cost_sums.shape != expected_matrix_shape:
            raise ValueError("accumulator state cost shape mismatch")
        if weighted_sums.shape != expected_matrix_shape:
            raise ValueError("accumulator state weighted-cost shape mismatch")
        if elapsed_hours.shape != expected_vector_shape:
            raise ValueError("accumulator state elapsed-hours shape mismatch")
        if raw_step_counts.shape != expected_vector_shape:
            raise ValueError("accumulator state step shape mismatch")
        for values, message in (
            (cost_sums, "accumulator state costs"),
            (weighted_sums, "accumulator state weighted costs"),
            (elapsed_hours, "accumulator state elapsed hours"),
        ):
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{message} must be finite and non-negative")
        if not np.all(np.isfinite(raw_step_counts)):
            raise ValueError("accumulator state steps must be finite")
        step_counts = raw_step_counts.astype(np.int64)
        if np.any(step_counts < 0) or not np.array_equal(raw_step_counts, step_counts):
            raise ValueError("accumulator state steps must be non-negative integers")
        for index, spec in enumerate(self.schema.specs):
            if spec.aggregation is ConstraintAggregation.EPISODE_EVENT_RATE and np.any(
                cost_sums[:, index] > 1.0 + self._EVENT_TOLERANCE
            ):
                raise ValueError(f"accumulator event cost {spec.name} is invalid")
        empty = step_counts == 0
        if np.any(np.abs(cost_sums[empty]) > self._EVENT_TOLERANCE):
            raise ValueError("accumulator empty episodes must have zero cost state")
        if np.any(np.abs(weighted_sums[empty]) > self._EVENT_TOLERANCE):
            raise ValueError("accumulator empty episodes must have zero weighted state")
        if np.any(np.abs(elapsed_hours[empty]) > self._EVENT_TOLERANCE):
            raise ValueError("accumulator empty episodes must have zero elapsed time")
        if np.any((step_counts > 0) & (elapsed_hours <= 0.0)):
            raise ValueError(
                "accumulator active episodes require positive elapsed time"
            )

        self._episode_cost_sums = cost_sums.copy()
        self._episode_time_weighted_sums = weighted_sums.copy()
        self._episode_elapsed_hours = elapsed_hours.copy()
        self._episode_step_counts = step_counts.copy()


__all__ = [
    "CompletedEpisodeBatch",
    "CompletedEpisodeCostAccumulator",
    "ConstraintAggregation",
    "ConstraintEstimate",
    "LagrangianConstraintSpec",
    "LagrangianSchema",
    "canonical_constraint_aggregation",
    "canonical_constraint_unit",
]
