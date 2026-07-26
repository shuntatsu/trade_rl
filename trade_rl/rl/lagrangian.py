"""Typed constraint aggregation and stabilized dual-optimization state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES


class ConstraintAggregation(str, Enum):
    """Completed-episode aggregation used by one maintained constraint."""

    EPISODE_SUM = "episode_sum"
    EPISODE_MEAN = "episode_mean"
    EPISODE_EVENT_RATE = "episode_event_rate"


_CANONICAL_AGGREGATIONS: dict[str, ConstraintAggregation] = {
    "drawdown_excess": ConstraintAggregation.EPISODE_SUM,
    "drawdown_stop_event": ConstraintAggregation.EPISODE_EVENT_RATE,
    "margin_deficit_fraction": ConstraintAggregation.EPISODE_SUM,
    "forced_liquidation_event": ConstraintAggregation.EPISODE_EVENT_RATE,
    "gross_exposure_request_excess": ConstraintAggregation.EPISODE_MEAN,
    "daily_turnover": ConstraintAggregation.EPISODE_MEAN,
    "execution_cost_fraction": ConstraintAggregation.EPISODE_SUM,
}


def canonical_constraint_aggregation(name: str) -> ConstraintAggregation:
    """Return the maintained aggregation for a canonical constraint cost."""

    if name not in CONSTRAINT_COST_NAMES:
        raise ValueError(f"unknown constraint cost: {name}")
    return _CANONICAL_AGGREGATIONS[name]


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
        if self.name not in CONSTRAINT_COST_NAMES:
            raise ValueError(f"unknown constraint cost: {self.name}")
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
    """One completed-episode rollout estimate for a maintained constraint."""

    name: str
    numerator: float
    denominator: int

    def __post_init__(self) -> None:
        if self.name not in CONSTRAINT_COST_NAMES:
            raise ValueError(f"unknown constraint cost: {self.name}")
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


class CompletedEpisodeCostAccumulator:
    """Aggregate completed episodes without losing cross-rollout partial state."""

    _STATE_VERSION = "completed_episode_cost_accumulator_v1"
    _EVENT_TOLERANCE = 1e-12

    def __init__(self, *, n_envs: int, schema: LagrangianSchema) -> None:
        if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs <= 0:
            raise ValueError("n_envs must be a positive integer")
        if not isinstance(schema, LagrangianSchema):
            raise TypeError("schema must be a LagrangianSchema")
        self.n_envs = n_envs
        self.schema = schema
        self._episode_cost_sums = np.zeros(
            (n_envs, len(schema.names)),
            dtype=np.float64,
        )
        self._episode_step_counts = np.zeros(n_envs, dtype=np.int64)

    def _validated_rollout(
        self,
        *,
        costs: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
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

        done_shape = (cost_array.shape[0], self.n_envs)
        terminated_array = np.asarray(terminated, dtype=np.bool_)
        truncated_array = np.asarray(truncated, dtype=np.bool_)
        if terminated_array.shape != done_shape or truncated_array.shape != done_shape:
            raise ValueError(f"termination arrays must have shape {done_shape}")
        if np.any(terminated_array & truncated_array):
            raise ValueError("a transition cannot be both terminated and truncated")

        for index, spec in enumerate(self.schema.specs):
            if spec.aggregation is ConstraintAggregation.EPISODE_EVENT_RATE and np.any(
                cost_array[:, :, index] > 1.0 + self._EVENT_TOLERANCE
            ):
                raise ValueError(f"event cost {spec.name} must be within [0, 1]")
        return cost_array, terminated_array, truncated_array

    def ingest_rollout(
        self,
        *,
        costs: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> dict[str, ConstraintEstimate | None]:
        """Consume one aligned rollout and estimate only completed episodes."""

        cost_array, terminated_array, truncated_array = self._validated_rollout(
            costs=costs,
            terminated=terminated,
            truncated=truncated,
        )
        episode_cost_sums = self._episode_cost_sums.copy()
        episode_step_counts = self._episode_step_counts.copy()
        numerators = np.zeros(len(self.schema.names), dtype=np.float64)
        completed_episode_count = 0

        for step in range(cost_array.shape[0]):
            for env_index in range(self.n_envs):
                episode_cost_sums[env_index] += cost_array[step, env_index]
                episode_step_counts[env_index] += 1
                if not (
                    terminated_array[step, env_index]
                    or truncated_array[step, env_index]
                ):
                    continue

                episode_steps = int(episode_step_counts[env_index])
                if episode_steps <= 0:
                    raise RuntimeError("completed episode has no steps")
                for cost_index, spec in enumerate(self.schema.specs):
                    episode_value = float(episode_cost_sums[env_index, cost_index])
                    if spec.aggregation is ConstraintAggregation.EPISODE_SUM:
                        contribution = episode_value
                    elif spec.aggregation is ConstraintAggregation.EPISODE_MEAN:
                        contribution = episode_value / episode_steps
                    else:
                        if episode_value > 1.0 + self._EVENT_TOLERANCE:
                            raise ValueError(
                                f"event cost {spec.name} occurred more than once "
                                "within one episode"
                            )
                        contribution = min(episode_value, 1.0)
                    if not math.isfinite(contribution) or contribution < 0.0:
                        raise RuntimeError("constraint aggregation became invalid")
                    numerators[cost_index] += contribution

                completed_episode_count += 1
                episode_cost_sums[env_index].fill(0.0)
                episode_step_counts[env_index] = 0

        self._episode_cost_sums = episode_cost_sums
        self._episode_step_counts = episode_step_counts
        if completed_episode_count == 0:
            return {name: None for name in self.schema.names}
        return {
            name: ConstraintEstimate(
                name=name,
                numerator=float(numerators[index]),
                denominator=completed_episode_count,
            )
            for index, name in enumerate(self.schema.names)
        }

    def state_dict(self) -> dict[str, object]:
        """Return JSON-compatible unfinished-episode state."""

        return {
            "cost_names": list(self.schema.names),
            "episode_cost_sums": self._episode_cost_sums.tolist(),
            "episode_step_counts": self._episode_step_counts.tolist(),
            "n_envs": self.n_envs,
            "schema_digest": self.schema.digest,
            "schema_version": self._STATE_VERSION,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore unfinished episodes only when identity and shapes match."""

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
            raw_step_counts = np.asarray(
                state["episode_step_counts"],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("accumulator state payload is invalid") from error
        if cost_sums.shape != self._episode_cost_sums.shape:
            raise ValueError("accumulator state cost shape mismatch")
        if raw_step_counts.shape != self._episode_step_counts.shape:
            raise ValueError("accumulator state step shape mismatch")
        if not np.all(np.isfinite(cost_sums)) or np.any(cost_sums < 0.0):
            raise ValueError("accumulator state costs must be finite and non-negative")
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

        self._episode_cost_sums = cost_sums.copy()
        self._episode_step_counts = step_counts.copy()


@dataclass(frozen=True, slots=True)
class DualUpdateReport:
    """One per-cost dual decision after a completed rollout."""

    name: str
    raw_estimate: float | None
    ema_estimate: float | None
    budget: float
    multiplier_before: float
    multiplier_after: float
    updated: bool
    skip_reason: str | None
    saturated: bool
    denominator: int | None
    rollout_count: int
    update_count: int


class LagrangianDualController:
    """Maintain independent EMA-smoothed, capped Lagrange multipliers."""

    _STATE_VERSION = "lagrangian_dual_controller_v1"
    _BOUNDARY_TOLERANCE = 1e-12

    def __init__(self, schema: LagrangianSchema) -> None:
        if not isinstance(schema, LagrangianSchema):
            raise TypeError("schema must be a LagrangianSchema")
        self.schema = schema
        self._multipliers = np.asarray(
            [spec.initial_multiplier for spec in schema.specs],
            dtype=np.float64,
        )
        self._ema_estimates = np.zeros(len(schema.names), dtype=np.float64)
        self._ema_initialized = np.zeros(len(schema.names), dtype=np.bool_)
        self._update_counts = np.zeros(len(schema.names), dtype=np.int64)
        self._rollout_count = 0

    @property
    def rollout_count(self) -> int:
        return self._rollout_count

    def begin_rollout(self) -> np.ndarray:
        """Return a read-only multiplier snapshot frozen for the rollout."""

        snapshot = self._multipliers.copy()
        snapshot.flags.writeable = False
        return snapshot

    def _validated_estimates(
        self,
        estimates: Mapping[str, ConstraintEstimate | None],
    ) -> tuple[ConstraintEstimate | None, ...]:
        if not isinstance(estimates, Mapping):
            raise TypeError("estimates must be a mapping")
        keys = tuple(estimates.keys())
        if len(keys) != len(self.schema.names) or set(keys) != set(self.schema.names):
            raise ValueError("estimate mapping constraint names do not match schema")
        ordered: list[ConstraintEstimate | None] = []
        for name in self.schema.names:
            estimate = estimates[name]
            if estimate is not None and not isinstance(estimate, ConstraintEstimate):
                raise TypeError(
                    "constraint estimate must be a ConstraintEstimate or null"
                )
            if estimate is not None and estimate.name != name:
                raise ValueError(f"estimate name mismatch for {name}")
            if estimate is not None:
                value = estimate.value
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "constraint estimate must be finite and non-negative"
                    )
            ordered.append(estimate)
        return tuple(ordered)

    def update_after_rollout(
        self,
        estimates: Mapping[str, ConstraintEstimate | None],
    ) -> dict[str, DualUpdateReport]:
        """Apply at most one eligible dual update per cost after one rollout."""

        ordered_estimates = self._validated_estimates(estimates)
        next_rollout_count = self._rollout_count + 1
        multipliers = self._multipliers.copy()
        ema_estimates = self._ema_estimates.copy()
        ema_initialized = self._ema_initialized.copy()
        update_counts = self._update_counts.copy()
        reports: dict[str, DualUpdateReport] = {}

        for index, (spec, estimate) in enumerate(
            zip(self.schema.specs, ordered_estimates, strict=True)
        ):
            multiplier_before = float(self._multipliers[index])
            previous_ema = (
                float(self._ema_estimates[index])
                if self._ema_initialized[index]
                else None
            )
            raw_estimate = estimate.value if estimate is not None else None
            denominator = estimate.denominator if estimate is not None else None
            skip_reason: str | None = None
            if next_rollout_count <= spec.warmup_rollouts:
                skip_reason = "warmup"
            elif (
                next_rollout_count - spec.warmup_rollouts
            ) % spec.update_interval_rollouts != 0:
                skip_reason = "update_interval"
            elif estimate is None:
                skip_reason = "missing_estimate"

            if skip_reason is not None:
                reports[spec.name] = DualUpdateReport(
                    name=spec.name,
                    raw_estimate=raw_estimate,
                    ema_estimate=previous_ema,
                    budget=spec.budget,
                    multiplier_before=multiplier_before,
                    multiplier_after=multiplier_before,
                    updated=False,
                    skip_reason=skip_reason,
                    saturated=False,
                    denominator=denominator,
                    rollout_count=next_rollout_count,
                    update_count=int(self._update_counts[index]),
                )
                continue

            if raw_estimate is None:
                raise RuntimeError("eligible constraint estimate unexpectedly missing")
            ema_after = (
                raw_estimate
                if previous_ema is None
                else spec.ema_beta * previous_ema + (1.0 - spec.ema_beta) * raw_estimate
            )
            if not math.isfinite(ema_after) or ema_after < 0.0:
                raise ValueError("EMA constraint estimate became invalid")
            proposed_multiplier = multiplier_before + spec.dual_learning_rate * (
                ema_after - spec.budget
            )
            if not math.isfinite(proposed_multiplier):
                raise ValueError("Lagrange multiplier update became non-finite")
            multiplier_after = float(
                np.clip(proposed_multiplier, 0.0, spec.max_multiplier)
            )
            if not math.isfinite(multiplier_after):
                raise ValueError("Lagrange multiplier became non-finite")
            saturated = (
                multiplier_after <= self._BOUNDARY_TOLERANCE
                or multiplier_after >= spec.max_multiplier - self._BOUNDARY_TOLERANCE
            )
            multipliers[index] = multiplier_after
            ema_estimates[index] = ema_after
            ema_initialized[index] = True
            update_counts[index] += 1
            reports[spec.name] = DualUpdateReport(
                name=spec.name,
                raw_estimate=raw_estimate,
                ema_estimate=ema_after,
                budget=spec.budget,
                multiplier_before=multiplier_before,
                multiplier_after=multiplier_after,
                updated=True,
                skip_reason=None,
                saturated=saturated,
                denominator=denominator,
                rollout_count=next_rollout_count,
                update_count=int(update_counts[index]),
            )

        self._multipliers = multipliers
        self._ema_estimates = ema_estimates
        self._ema_initialized = ema_initialized
        self._update_counts = update_counts
        self._rollout_count = next_rollout_count
        return reports

    def state_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible dual state."""

        return {
            "cost_names": list(self.schema.names),
            "ema_estimates": [
                float(self._ema_estimates[index])
                if self._ema_initialized[index]
                else None
                for index in range(len(self.schema.names))
            ],
            "multipliers": self._multipliers.tolist(),
            "rollout_count": self._rollout_count,
            "schema_digest": self.schema.digest,
            "schema_version": self._STATE_VERSION,
            "update_counts": self._update_counts.tolist(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore dual state only when schema identity and values are valid."""

        if state.get("schema_version") != self._STATE_VERSION:
            raise ValueError("dual state schema version mismatch")
        raw_cost_names = state.get("cost_names")
        if not isinstance(raw_cost_names, (list, tuple)) or not all(
            isinstance(name, str) for name in raw_cost_names
        ):
            raise ValueError("dual state schema mismatch")
        if (
            state.get("schema_digest") != self.schema.digest
            or tuple(raw_cost_names) != self.schema.names
        ):
            raise ValueError("dual state schema mismatch")

        try:
            multipliers = np.asarray(state["multipliers"], dtype=np.float64)
            raw_update_counts = np.asarray(
                state["update_counts"],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("dual state payload is invalid") from error
        expected_shape = (len(self.schema.names),)
        if multipliers.shape != expected_shape:
            raise ValueError("dual state multiplier shape mismatch")
        if raw_update_counts.shape != expected_shape:
            raise ValueError("dual state update-count shape mismatch")
        if not np.all(np.isfinite(multipliers)) or np.any(multipliers < 0.0):
            raise ValueError("dual state multipliers must be finite and non-negative")
        for index, spec in enumerate(self.schema.specs):
            if multipliers[index] > spec.max_multiplier + self._BOUNDARY_TOLERANCE:
                raise ValueError(f"dual state multiplier exceeds cap for {spec.name}")
        if not np.all(np.isfinite(raw_update_counts)):
            raise ValueError("dual state update counts must be finite")
        update_counts = raw_update_counts.astype(np.int64)
        if np.any(update_counts < 0) or not np.array_equal(
            raw_update_counts, update_counts
        ):
            raise ValueError("dual state update counts must be non-negative integers")

        raw_ema_estimates = state.get("ema_estimates")
        if not isinstance(raw_ema_estimates, (list, tuple)) or len(
            raw_ema_estimates
        ) != len(self.schema.names):
            raise ValueError("dual state EMA shape mismatch")
        ema_estimates = np.zeros(len(self.schema.names), dtype=np.float64)
        ema_initialized = np.zeros(len(self.schema.names), dtype=np.bool_)
        for index, raw_value in enumerate(raw_ema_estimates):
            if raw_value is None:
                continue
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError("dual state EMA values must be numeric or null")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "dual state EMA values must be finite and non-negative"
                )
            ema_estimates[index] = value
            ema_initialized[index] = True

        raw_rollout_count = state.get("rollout_count")
        if (
            isinstance(raw_rollout_count, bool)
            or not isinstance(raw_rollout_count, int)
            or raw_rollout_count < 0
        ):
            raise ValueError("dual state rollout count must be a non-negative integer")

        self._multipliers = multipliers.copy()
        self._ema_estimates = ema_estimates
        self._ema_initialized = ema_initialized
        self._update_counts = update_counts.copy()
        self._rollout_count = raw_rollout_count


_T = TypeVar("_T")


def _validated_vector(
    values: tuple[_T, ...],
    *,
    expected_length: int,
    field_name: str,
) -> tuple[_T, ...]:
    result = tuple(values)
    if len(result) != expected_length:
        raise ValueError(f"{field_name} must contain exactly {expected_length} values")
    return result


def canonical_lagrangian_schema(
    *,
    names: tuple[str, ...],
    budgets: tuple[float, ...],
    dual_learning_rates: tuple[float, ...],
    ema_betas: tuple[float, ...],
    initial_multipliers: tuple[float, ...],
    max_multipliers: tuple[float, ...],
    warmup_rollouts: tuple[int, ...],
    update_interval_rollouts: tuple[int, ...],
) -> LagrangianSchema:
    """Build an explicit schema from canonical-order configuration vectors."""

    ordered_names = tuple(names)
    if not ordered_names:
        raise ValueError("names must not be empty")
    expected_length = len(ordered_names)
    vectors = (
        _validated_vector(
            budgets,
            expected_length=expected_length,
            field_name="budgets",
        ),
        _validated_vector(
            dual_learning_rates,
            expected_length=expected_length,
            field_name="dual_learning_rates",
        ),
        _validated_vector(
            ema_betas,
            expected_length=expected_length,
            field_name="ema_betas",
        ),
        _validated_vector(
            initial_multipliers,
            expected_length=expected_length,
            field_name="initial_multipliers",
        ),
        _validated_vector(
            max_multipliers,
            expected_length=expected_length,
            field_name="max_multipliers",
        ),
        _validated_vector(
            warmup_rollouts,
            expected_length=expected_length,
            field_name="warmup_rollouts",
        ),
        _validated_vector(
            update_interval_rollouts,
            expected_length=expected_length,
            field_name="update_interval_rollouts",
        ),
    )
    return LagrangianSchema(
        tuple(
            LagrangianConstraintSpec(
                name=name,
                aggregation=canonical_constraint_aggregation(name),
                budget=budget,
                dual_learning_rate=learning_rate,
                ema_beta=ema_beta,
                initial_multiplier=initial_multiplier,
                max_multiplier=max_multiplier,
                warmup_rollouts=warmup,
                update_interval_rollouts=update_interval,
            )
            for (
                name,
                budget,
                learning_rate,
                ema_beta,
                initial_multiplier,
                max_multiplier,
                warmup,
                update_interval,
            ) in zip(ordered_names, *vectors, strict=True)
        )
    )


__all__ = [
    "CompletedEpisodeCostAccumulator",
    "ConstraintAggregation",
    "ConstraintEstimate",
    "DualUpdateReport",
    "LagrangianConstraintSpec",
    "LagrangianDualController",
    "LagrangianSchema",
    "canonical_constraint_aggregation",
    "canonical_lagrangian_schema",
]
