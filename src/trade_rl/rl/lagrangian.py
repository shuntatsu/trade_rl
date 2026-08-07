"""Stabilized Lagrangian dual optimization and completed-episode exports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from trade_rl.rl.lagrangian_statistics import (
    CompletedEpisodeBatch,
    CompletedEpisodeCostAccumulator,
    ConstraintAggregation,
    ConstraintEstimate,
    LagrangianSchema,
    canonical_constraint_aggregation,
    canonical_constraint_unit,
)
from trade_rl.rl.lagrangian_statistics import (
    LagrangianConstraintSpec as BaseLagrangianConstraintSpec,
)


@dataclass(frozen=True, slots=True)
class LagrangianConstraintSpec(BaseLagrangianConstraintSpec):
    """Constraint settings including support required for one dual update."""

    minimum_completed_episodes: int

    def __post_init__(self) -> None:
        BaseLagrangianConstraintSpec.__post_init__(self)
        value = self.minimum_completed_episodes
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("minimum_completed_episodes must be a positive integer")

    def digest_payload(self) -> dict[str, object]:
        payload = BaseLagrangianConstraintSpec.digest_payload(self)
        payload["minimum_completed_episodes"] = self.minimum_completed_episodes
        return payload


@dataclass(frozen=True, slots=True)
class DualUpdateReport:
    """One per-cost estimator and dual-actuator decision."""

    name: str
    raw_estimate: float | None
    ema_estimate: float | None
    budget: float
    multiplier_before: float
    multiplier_after: float
    updated: bool
    skip_reason: str | None
    denominator: int | None
    pending_numerator_before: float
    pending_denominator_before: int
    consumed_denominator: int
    censored_episode_count: int
    constraint_residual: float | None
    at_lower_bound: bool
    at_upper_cap: bool
    rollout_count: int
    update_count: int

    @property
    def saturated(self) -> bool:
        """Compatibility alias for upper-cap saturation only."""

        return self.at_upper_cap


class LagrangianDualController:
    """Pool completed episodes before scheduled integral dual updates."""

    _STATE_VERSION = "lagrangian_dual_controller_v2"
    _BOUNDARY_TOLERANCE = 1e-12

    def __init__(self, schema: LagrangianSchema) -> None:
        if not isinstance(schema, LagrangianSchema):
            raise TypeError("schema must be a LagrangianSchema")
        if any(not isinstance(spec, LagrangianConstraintSpec) for spec in schema.specs):
            raise TypeError(
                "schema constraint specs must include minimum completed episode support"
            )
        self.schema = schema
        size = len(schema.names)
        self._multipliers = np.asarray(
            [spec.initial_multiplier for spec in schema.specs],
            dtype=np.float64,
        )
        self._ema_estimates = np.zeros(size, dtype=np.float64)
        self._ema_initialized = np.zeros(size, dtype=np.bool_)
        self._update_counts = np.zeros(size, dtype=np.int64)
        self._pending_numerators = np.zeros(size, dtype=np.float64)
        self._pending_denominators = np.zeros(size, dtype=np.int64)
        self._censored_episode_count = 0
        self._rollout_count = 0

    @property
    def rollout_count(self) -> int:
        return self._rollout_count

    @property
    def state_version(self) -> str:
        return self._STATE_VERSION

    def begin_rollout(self) -> np.ndarray:
        """Return a read-only multiplier snapshot frozen for one rollout."""

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

    @staticmethod
    def _validated_non_negative_integer(
        value: object,
        *,
        field_name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
        return value

    @classmethod
    def _at_lower_bound(cls, multiplier: float) -> bool:
        return multiplier <= cls._BOUNDARY_TOLERANCE

    @classmethod
    def _at_upper_cap(cls, multiplier: float, cap: float) -> bool:
        return multiplier >= cap - cls._BOUNDARY_TOLERANCE

    def update_after_rollout(
        self,
        estimates: Mapping[str, ConstraintEstimate | None],
        *,
        censored_episode_count: int,
    ) -> dict[str, DualUpdateReport]:
        """Retain observations before applying schedule and support gates."""

        ordered_estimates = self._validated_estimates(estimates)
        censored_increment = self._validated_non_negative_integer(
            censored_episode_count,
            field_name="censored_episode_count",
        )
        next_rollout_count = self._rollout_count + 1
        next_censored_count = self._censored_episode_count + censored_increment

        multipliers = self._multipliers.copy()
        ema_estimates = self._ema_estimates.copy()
        ema_initialized = self._ema_initialized.copy()
        update_counts = self._update_counts.copy()
        pending_numerators = self._pending_numerators.copy()
        pending_denominators = self._pending_denominators.copy()
        reports: dict[str, DualUpdateReport] = {}

        for index, (raw_spec, estimate) in enumerate(
            zip(self.schema.specs, ordered_estimates, strict=True)
        ):
            if not isinstance(raw_spec, LagrangianConstraintSpec):
                raise TypeError("Lagrangian constraint support metadata is missing")
            spec = raw_spec
            multiplier_before = float(self._multipliers[index])
            previous_ema = (
                float(self._ema_estimates[index])
                if self._ema_initialized[index]
                else None
            )
            pending_numerator_before = float(self._pending_numerators[index])
            pending_denominator_before = int(self._pending_denominators[index])
            current_denominator = estimate.denominator if estimate is not None else None

            if estimate is not None:
                pending_numerators[index] += estimate.numerator
                pending_denominators[index] += estimate.denominator
            pooled_numerator = float(pending_numerators[index])
            pooled_denominator = int(pending_denominators[index])
            if not math.isfinite(pooled_numerator) or pooled_numerator < 0.0:
                raise ValueError("pending constraint numerator became invalid")
            raw_estimate = (
                pooled_numerator / pooled_denominator
                if pooled_denominator > 0
                else None
            )
            if raw_estimate is not None and (
                not math.isfinite(raw_estimate) or raw_estimate < 0.0
            ):
                raise ValueError("pooled constraint estimate became invalid")

            skip_reason: str | None = None
            if next_rollout_count <= spec.warmup_rollouts:
                skip_reason = "warmup"
            elif (
                next_rollout_count - spec.warmup_rollouts
            ) % spec.update_interval_rollouts != 0:
                skip_reason = "update_interval"
            elif 0 < pooled_denominator < spec.minimum_completed_episodes:
                skip_reason = "insufficient_completed_episodes"
            elif pooled_denominator == 0:
                skip_reason = "missing_estimate_or_pending_support"

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
                    denominator=current_denominator,
                    pending_numerator_before=pending_numerator_before,
                    pending_denominator_before=pending_denominator_before,
                    consumed_denominator=0,
                    censored_episode_count=next_censored_count,
                    constraint_residual=None,
                    at_lower_bound=self._at_lower_bound(multiplier_before),
                    at_upper_cap=self._at_upper_cap(
                        multiplier_before,
                        spec.max_multiplier,
                    ),
                    rollout_count=next_rollout_count,
                    update_count=int(self._update_counts[index]),
                )
                continue

            if raw_estimate is None:
                raise RuntimeError("eligible pooled constraint estimate is missing")
            consumed_denominator = pooled_denominator
            beta_effective = spec.ema_beta**consumed_denominator
            ema_after = (
                raw_estimate
                if previous_ema is None
                else beta_effective * previous_ema
                + (1.0 - beta_effective) * raw_estimate
            )
            if not math.isfinite(ema_after) or ema_after < 0.0:
                raise ValueError("EMA constraint estimate became invalid")
            residual = ema_after - spec.budget
            if not math.isfinite(residual):
                raise ValueError("constraint residual became non-finite")
            proposed_multiplier = multiplier_before + spec.dual_learning_rate * residual
            if not math.isfinite(proposed_multiplier):
                raise ValueError("Lagrange multiplier update became non-finite")
            multiplier_after = float(
                np.clip(proposed_multiplier, 0.0, spec.max_multiplier)
            )
            if not math.isfinite(multiplier_after):
                raise ValueError("Lagrange multiplier became non-finite")

            multipliers[index] = multiplier_after
            ema_estimates[index] = ema_after
            ema_initialized[index] = True
            update_counts[index] += 1
            pending_numerators[index] = 0.0
            pending_denominators[index] = 0
            reports[spec.name] = DualUpdateReport(
                name=spec.name,
                raw_estimate=raw_estimate,
                ema_estimate=ema_after,
                budget=spec.budget,
                multiplier_before=multiplier_before,
                multiplier_after=multiplier_after,
                updated=True,
                skip_reason=None,
                denominator=current_denominator,
                pending_numerator_before=pending_numerator_before,
                pending_denominator_before=pending_denominator_before,
                consumed_denominator=consumed_denominator,
                censored_episode_count=next_censored_count,
                constraint_residual=residual,
                at_lower_bound=self._at_lower_bound(multiplier_after),
                at_upper_cap=self._at_upper_cap(
                    multiplier_after,
                    spec.max_multiplier,
                ),
                rollout_count=next_rollout_count,
                update_count=int(update_counts[index]),
            )

        self._multipliers = multipliers
        self._ema_estimates = ema_estimates
        self._ema_initialized = ema_initialized
        self._update_counts = update_counts
        self._pending_numerators = pending_numerators
        self._pending_denominators = pending_denominators
        self._censored_episode_count = next_censored_count
        self._rollout_count = next_rollout_count
        return reports

    def state_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible estimator and dual state."""

        return {
            "censored_episode_count": self._censored_episode_count,
            "cost_names": list(self.schema.names),
            "ema_estimates": [
                float(self._ema_estimates[index])
                if self._ema_initialized[index]
                else None
                for index in range(len(self.schema.names))
            ],
            "multipliers": self._multipliers.tolist(),
            "pending_denominators": self._pending_denominators.tolist(),
            "pending_numerators": self._pending_numerators.tolist(),
            "rollout_count": self._rollout_count,
            "schema_digest": self.schema.digest,
            "schema_version": self._STATE_VERSION,
            "update_counts": self._update_counts.tolist(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore estimator and controller state after complete validation."""

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
            pending_numerators = np.asarray(
                state["pending_numerators"],
                dtype=np.float64,
            )
            raw_pending_denominators = np.asarray(
                state["pending_denominators"],
                dtype=np.float64,
            )
            raw_update_counts = np.asarray(
                state["update_counts"],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("dual state payload is invalid") from error

        expected_shape = (len(self.schema.names),)
        for values, message in (
            (multipliers, "dual state multiplier shape mismatch"),
            (pending_numerators, "dual state pending-numerator shape mismatch"),
            (
                raw_pending_denominators,
                "dual state pending-denominator shape mismatch",
            ),
            (raw_update_counts, "dual state update-count shape mismatch"),
        ):
            if values.shape != expected_shape:
                raise ValueError(message)

        if not np.all(np.isfinite(multipliers)) or np.any(multipliers < 0.0):
            raise ValueError("dual state multipliers must be finite and non-negative")
        for index, raw_spec in enumerate(self.schema.specs):
            if multipliers[index] > raw_spec.max_multiplier + self._BOUNDARY_TOLERANCE:
                raise ValueError(
                    f"dual state multiplier exceeds cap for {raw_spec.name}"
                )
        if not np.all(np.isfinite(pending_numerators)) or np.any(
            pending_numerators < 0.0
        ):
            raise ValueError(
                "dual state pending numerators must be finite and non-negative"
            )

        if not np.all(np.isfinite(raw_pending_denominators)):
            raise ValueError("dual state pending denominators must be finite")
        pending_denominators = raw_pending_denominators.astype(np.int64)
        if np.any(pending_denominators < 0) or not np.array_equal(
            raw_pending_denominators,
            pending_denominators,
        ):
            raise ValueError(
                "dual state pending denominators must be non-negative integers"
            )
        if np.any(
            (pending_denominators == 0)
            & (pending_numerators > self._BOUNDARY_TOLERANCE)
        ):
            raise ValueError("dual state pending support is inconsistent")

        if not np.all(np.isfinite(raw_update_counts)):
            raise ValueError("dual state update counts must be finite")
        update_counts = raw_update_counts.astype(np.int64)
        if np.any(update_counts < 0) or not np.array_equal(
            raw_update_counts,
            update_counts,
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

        rollout_count = self._validated_non_negative_integer(
            state.get("rollout_count"),
            field_name="dual state rollout count",
        )
        censored_count = self._validated_non_negative_integer(
            state.get("censored_episode_count"),
            field_name="dual state censored episode count",
        )

        self._multipliers = multipliers.copy()
        self._ema_estimates = ema_estimates
        self._ema_initialized = ema_initialized
        self._update_counts = update_counts.copy()
        self._pending_numerators = pending_numerators.copy()
        self._pending_denominators = pending_denominators.copy()
        self._censored_episode_count = censored_count
        self._rollout_count = rollout_count


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
    minimum_completed_episodes: tuple[int, ...],
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
        _validated_vector(
            minimum_completed_episodes,
            expected_length=expected_length,
            field_name="minimum_completed_episodes",
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
                minimum_completed_episodes=minimum_support,
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
                minimum_support,
            ) in zip(ordered_names, *vectors, strict=True)
        )
    )


__all__ = [
    "CompletedEpisodeBatch",
    "CompletedEpisodeCostAccumulator",
    "ConstraintAggregation",
    "ConstraintEstimate",
    "DualUpdateReport",
    "LagrangianConstraintSpec",
    "LagrangianDualController",
    "LagrangianSchema",
    "canonical_constraint_aggregation",
    "canonical_constraint_unit",
    "canonical_lagrangian_schema",
]
