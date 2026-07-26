"""Typed constraint aggregation and dual-optimization identity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

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
    "ConstraintAggregation",
    "LagrangianConstraintSpec",
    "LagrangianSchema",
    "canonical_constraint_aggregation",
    "canonical_lagrangian_schema",
]
