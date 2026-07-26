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
    LagrangianConstraintSpec,
    LagrangianSchema,
    canonical_constraint_aggregation,
    canonical_constraint_unit,
)


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
