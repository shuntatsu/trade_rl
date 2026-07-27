"""Framework-independent canonical constraint names, aggregations, and units."""

from __future__ import annotations

from enum import Enum

CONSTRAINT_COST_NAMES = (
    "drawdown_excess",
    "drawdown_stop_event",
    "margin_deficit_fraction",
    "forced_liquidation_event",
    "gross_exposure_request_excess",
    "daily_turnover",
    "execution_cost_fraction",
)


class ConstraintAggregation(str, Enum):
    """Completed-episode aggregation used by one maintained constraint."""

    EPISODE_TIME_AREA = "episode_time_area"
    EPISODE_DECISION_MEAN = "episode_decision_mean"
    EPISODE_TIME_WEIGHTED_MEAN = "episode_time_weighted_mean"
    EPISODE_EVENT_RATE = "episode_event_rate"
    EPISODE_SUM = "episode_sum"


_CANONICAL_AGGREGATIONS: dict[str, ConstraintAggregation] = {
    "drawdown_excess": ConstraintAggregation.EPISODE_TIME_AREA,
    "drawdown_stop_event": ConstraintAggregation.EPISODE_EVENT_RATE,
    "margin_deficit_fraction": ConstraintAggregation.EPISODE_TIME_AREA,
    "forced_liquidation_event": ConstraintAggregation.EPISODE_EVENT_RATE,
    "gross_exposure_request_excess": ConstraintAggregation.EPISODE_DECISION_MEAN,
    "daily_turnover": ConstraintAggregation.EPISODE_TIME_WEIGHTED_MEAN,
    "execution_cost_fraction": ConstraintAggregation.EPISODE_SUM,
}
_CANONICAL_UNITS: dict[str, str] = {
    "drawdown_excess": "drawdown_excess_area_days",
    "drawdown_stop_event": "event_per_episode",
    "margin_deficit_fraction": "margin_deficit_fraction_days",
    "forced_liquidation_event": "event_per_episode",
    "gross_exposure_request_excess": "excess_per_decision",
    "daily_turnover": "turnover_per_day",
    "execution_cost_fraction": "execution_cost_fraction_per_episode",
}


def validate_constraint_name(name: str) -> None:
    """Reject names outside the maintained canonical constraint schema."""

    if name not in CONSTRAINT_COST_NAMES:
        raise ValueError(f"unknown constraint cost: {name}")


def canonical_constraint_aggregation(name: str) -> ConstraintAggregation:
    """Return the maintained aggregation for a canonical constraint cost."""

    validate_constraint_name(name)
    return _CANONICAL_AGGREGATIONS[name]


def canonical_constraint_unit(name: str) -> str:
    """Return the evidence unit for a canonical completed-episode estimate."""

    validate_constraint_name(name)
    return _CANONICAL_UNITS[name]


__all__ = [
    "CONSTRAINT_COST_NAMES",
    "ConstraintAggregation",
    "canonical_constraint_aggregation",
    "canonical_constraint_unit",
    "validate_constraint_name",
]
