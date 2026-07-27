from __future__ import annotations

from trade_rl.domain.constraint_contracts import (
    CONSTRAINT_COST_NAMES,
    ConstraintAggregation,
    canonical_constraint_aggregation,
    canonical_constraint_unit,
)
from trade_rl.rl.environment_constraints import (
    CONSTRAINT_COST_NAMES as ENVIRONMENT_CONSTRAINT_COST_NAMES,
)
from trade_rl.rl.lagrangian_statistics import (
    ConstraintAggregation as RLConstraintAggregation,
)
from trade_rl.rl.lagrangian_statistics import (
    canonical_constraint_aggregation as rl_canonical_constraint_aggregation,
)
from trade_rl.rl.lagrangian_statistics import (
    canonical_constraint_unit as rl_canonical_constraint_unit,
)


def test_constraint_metadata_has_one_domain_owned_identity() -> None:
    assert ENVIRONMENT_CONSTRAINT_COST_NAMES is CONSTRAINT_COST_NAMES
    assert RLConstraintAggregation is ConstraintAggregation
    assert rl_canonical_constraint_aggregation is canonical_constraint_aggregation
    assert rl_canonical_constraint_unit is canonical_constraint_unit


def test_constraint_metadata_preserves_the_canonical_order() -> None:
    assert CONSTRAINT_COST_NAMES == (
        "drawdown_excess",
        "drawdown_stop_event",
        "margin_deficit_fraction",
        "forced_liquidation_event",
        "gross_exposure_request_excess",
        "daily_turnover",
        "execution_cost_fraction",
    )
    assert tuple(
        canonical_constraint_aggregation(name) for name in CONSTRAINT_COST_NAMES
    ) == (
        ConstraintAggregation.EPISODE_TIME_AREA,
        ConstraintAggregation.EPISODE_EVENT_RATE,
        ConstraintAggregation.EPISODE_TIME_AREA,
        ConstraintAggregation.EPISODE_EVENT_RATE,
        ConstraintAggregation.EPISODE_DECISION_MEAN,
        ConstraintAggregation.EPISODE_TIME_WEIGHTED_MEAN,
        ConstraintAggregation.EPISODE_SUM,
    )
    assert tuple(canonical_constraint_unit(name) for name in CONSTRAINT_COST_NAMES) == (
        "drawdown_excess_area_days",
        "event_per_episode",
        "margin_deficit_fraction_days",
        "event_per_episode",
        "excess_per_decision",
        "turnover_per_day",
        "execution_cost_fraction_per_episode",
    )
