from __future__ import annotations

import pytest

from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian import (
    ConstraintAggregation,
    LagrangianConstraintSpec,
    LagrangianSchema,
    canonical_constraint_aggregation,
)


def _spec(
    name: str,
    *,
    aggregation: ConstraintAggregation | None = None,
    budget: float = 0.0,
    dual_learning_rate: float = 0.05,
    ema_beta: float = 0.9,
    initial_multiplier: float = 0.0,
    max_multiplier: float = 10.0,
    warmup_rollouts: int = 2,
    update_interval_rollouts: int = 3,
) -> LagrangianConstraintSpec:
    return LagrangianConstraintSpec(
        name=name,
        aggregation=(
            canonical_constraint_aggregation(name)
            if aggregation is None
            else aggregation
        ),
        budget=budget,
        dual_learning_rate=dual_learning_rate,
        ema_beta=ema_beta,
        initial_multiplier=initial_multiplier,
        max_multiplier=max_multiplier,
        warmup_rollouts=warmup_rollouts,
        update_interval_rollouts=update_interval_rollouts,
    )


def test_lagrangian_schema_preserves_canonical_order_and_identity() -> None:
    schema = LagrangianSchema(
        (
            _spec("drawdown_excess"),
            _spec("drawdown_stop_event"),
        )
    )

    assert schema.names == ("drawdown_excess", "drawdown_stop_event")
    assert schema["drawdown_excess"].aggregation is ConstraintAggregation.EPISODE_SUM
    assert (
        schema["drawdown_stop_event"].aggregation
        is ConstraintAggregation.EPISODE_EVENT_RATE
    )
    assert len(schema.digest) == 64
    assert schema.digest_payload()["names"] == list(schema.names)


def test_canonical_constraint_aggregations_cover_every_cost() -> None:
    expected = {
        "drawdown_excess": ConstraintAggregation.EPISODE_SUM,
        "drawdown_stop_event": ConstraintAggregation.EPISODE_EVENT_RATE,
        "margin_deficit_fraction": ConstraintAggregation.EPISODE_SUM,
        "forced_liquidation_event": ConstraintAggregation.EPISODE_EVENT_RATE,
        "gross_exposure_request_excess": ConstraintAggregation.EPISODE_MEAN,
        "daily_turnover": ConstraintAggregation.EPISODE_MEAN,
        "execution_cost_fraction": ConstraintAggregation.EPISODE_SUM,
    }

    assert tuple(expected) == CONSTRAINT_COST_NAMES
    assert {
        name: canonical_constraint_aggregation(name) for name in CONSTRAINT_COST_NAMES
    } == expected


def test_lagrangian_schema_digest_changes_with_dual_semantics() -> None:
    baseline = LagrangianSchema((_spec("drawdown_excess"),))
    changed_budget = LagrangianSchema((_spec("drawdown_excess", budget=0.01),))
    changed_rate = LagrangianSchema(
        (_spec("drawdown_excess", dual_learning_rate=0.1),)
    )
    changed_ema = LagrangianSchema((_spec("drawdown_excess", ema_beta=0.95),))
    changed_cap = LagrangianSchema((_spec("drawdown_excess", max_multiplier=20.0),))
    changed_schedule = LagrangianSchema(
        (_spec("drawdown_excess", warmup_rollouts=1, update_interval_rollouts=1),)
    )

    assert len(
        {
            baseline.digest,
            changed_budget.digest,
            changed_rate.digest,
            changed_ema.digest,
            changed_cap.digest,
            changed_schedule.digest,
        }
    ) == 6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"budget": -0.01}, "budget"),
        ({"dual_learning_rate": 0.0}, "dual_learning_rate"),
        ({"dual_learning_rate": float("nan")}, "dual_learning_rate"),
        ({"ema_beta": -0.1}, "ema_beta"),
        ({"ema_beta": 1.0}, "ema_beta"),
        ({"initial_multiplier": -0.1}, "initial_multiplier"),
        (
            {"initial_multiplier": 11.0, "max_multiplier": 10.0},
            "initial_multiplier",
        ),
        ({"max_multiplier": 0.0}, "max_multiplier"),
        ({"warmup_rollouts": -1}, "warmup_rollouts"),
        ({"update_interval_rollouts": 0}, "update_interval_rollouts"),
    ],
)
def test_lagrangian_constraint_spec_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _spec("drawdown_excess", **kwargs)  # type: ignore[arg-type]


def test_lagrangian_constraint_spec_rejects_unknown_cost() -> None:
    with pytest.raises(ValueError, match="unknown constraint cost"):
        LagrangianConstraintSpec(
            name="unknown",
            aggregation=ConstraintAggregation.EPISODE_SUM,
            budget=0.0,
            dual_learning_rate=0.05,
            ema_beta=0.9,
            initial_multiplier=0.0,
            max_multiplier=10.0,
            warmup_rollouts=0,
            update_interval_rollouts=1,
        )


def test_lagrangian_constraint_spec_rejects_wrong_aggregation() -> None:
    with pytest.raises(ValueError, match="aggregation mismatch"):
        _spec(
            "drawdown_stop_event",
            aggregation=ConstraintAggregation.EPISODE_SUM,
        )


def test_lagrangian_schema_rejects_duplicates_and_reordering() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        LagrangianSchema((_spec("drawdown_excess"), _spec("drawdown_excess")))

    with pytest.raises(ValueError, match="canonical order"):
        LagrangianSchema(
            (
                _spec("margin_deficit_fraction"),
                _spec("drawdown_stop_event"),
            )
        )


def test_lagrangian_schema_requires_at_least_one_constraint() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        LagrangianSchema(())
