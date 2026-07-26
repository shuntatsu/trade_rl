from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian import (
    CompletedEpisodeCostAccumulator,
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


def _aggregation_schema() -> LagrangianSchema:
    return LagrangianSchema(
        (
            _spec("drawdown_excess"),
            _spec("drawdown_stop_event"),
            _spec("gross_exposure_request_excess"),
        )
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


def test_canonical_constraint_aggregation_rejects_unknown_cost() -> None:
    with pytest.raises(ValueError, match="unknown constraint cost"):
        canonical_constraint_aggregation("unknown")


def test_lagrangian_schema_digest_changes_with_dual_semantics() -> None:
    baseline = LagrangianSchema((_spec("drawdown_excess"),))
    changed_budget = LagrangianSchema((_spec("drawdown_excess", budget=0.01),))
    changed_rate = LagrangianSchema((_spec("drawdown_excess", dual_learning_rate=0.1),))
    changed_ema = LagrangianSchema((_spec("drawdown_excess", ema_beta=0.95),))
    changed_cap = LagrangianSchema((_spec("drawdown_excess", max_multiplier=20.0),))
    changed_schedule = LagrangianSchema(
        (_spec("drawdown_excess", warmup_rollouts=1, update_interval_rollouts=1),)
    )

    assert (
        len(
            {
                baseline.digest,
                changed_budget.digest,
                changed_rate.digest,
                changed_ema.digest,
                changed_cap.digest,
                changed_schedule.digest,
            }
        )
        == 6
    )


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


def test_completed_episode_costs_aggregate_sum_mean_and_event_rate() -> None:
    accumulator = CompletedEpisodeCostAccumulator(
        n_envs=2,
        schema=_aggregation_schema(),
    )
    costs = np.asarray(
        [
            [[0.01, 0.0, 0.10], [0.02, 0.0, 0.20]],
            [[0.02, 1.0, 0.30], [0.01, 0.0, 0.40]],
            [[0.03, 0.0, 0.20], [0.03, 0.0, 0.60]],
        ],
        dtype=np.float64,
    )
    terminated = np.asarray(
        [[False, False], [True, False], [False, False]],
        dtype=np.bool_,
    )
    truncated = np.asarray(
        [[False, False], [False, False], [False, True]],
        dtype=np.bool_,
    )

    estimates = accumulator.ingest_rollout(
        costs=costs,
        terminated=terminated,
        truncated=truncated,
    )

    drawdown = estimates["drawdown_excess"]
    event = estimates["drawdown_stop_event"]
    gross = estimates["gross_exposure_request_excess"]
    assert drawdown is not None
    assert event is not None
    assert gross is not None
    assert drawdown.numerator == pytest.approx(0.09)
    assert drawdown.denominator == 2
    assert drawdown.value == pytest.approx(0.045)
    assert event.numerator == pytest.approx(1.0)
    assert event.denominator == 2
    assert event.value == pytest.approx(0.5)
    assert gross.numerator == pytest.approx(0.6)
    assert gross.denominator == 2
    assert gross.value == pytest.approx(0.3)


def test_completed_episode_accumulator_carries_unfinished_state_across_rollouts() -> (
    None
):
    accumulator = CompletedEpisodeCostAccumulator(
        n_envs=2,
        schema=_aggregation_schema(),
    )
    first = accumulator.ingest_rollout(
        costs=np.asarray(
            [[[0.03, 0.0, 0.20], [0.01, 0.0, 0.10]]],
            dtype=np.float64,
        ),
        terminated=np.zeros((1, 2), dtype=np.bool_),
        truncated=np.zeros((1, 2), dtype=np.bool_),
    )
    assert all(value is None for value in first.values())

    second = accumulator.ingest_rollout(
        costs=np.asarray(
            [[[0.04, 0.0, 0.40], [0.02, 0.0, 0.30]]],
            dtype=np.float64,
        ),
        terminated=np.asarray([[True, False]], dtype=np.bool_),
        truncated=np.zeros((1, 2), dtype=np.bool_),
    )

    drawdown = second["drawdown_excess"]
    gross = second["gross_exposure_request_excess"]
    assert drawdown is not None
    assert gross is not None
    assert drawdown.value == pytest.approx(0.07)
    assert gross.value == pytest.approx(0.30)

    state = accumulator.state_dict()
    restored = CompletedEpisodeCostAccumulator(
        n_envs=2,
        schema=_aggregation_schema(),
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    original_next = accumulator.ingest_rollout(
        costs=np.asarray(
            [[[0.03, 0.0, 0.50], [0.03, 0.0, 0.50]]],
            dtype=np.float64,
        ),
        terminated=np.asarray([[False, True]], dtype=np.bool_),
        truncated=np.zeros((1, 2), dtype=np.bool_),
    )
    restored_next = restored.ingest_rollout(
        costs=np.asarray(
            [[[0.03, 0.0, 0.50], [0.03, 0.0, 0.50]]],
            dtype=np.float64,
        ),
        terminated=np.asarray([[False, True]], dtype=np.bool_),
        truncated=np.zeros((1, 2), dtype=np.bool_),
    )
    assert restored_next == original_next


def test_completed_episode_accumulator_rejects_invalid_rollouts() -> None:
    accumulator = CompletedEpisodeCostAccumulator(
        n_envs=2,
        schema=_aggregation_schema(),
    )
    valid_costs = np.zeros((1, 2, 3), dtype=np.float64)
    valid_done = np.zeros((1, 2), dtype=np.bool_)

    with pytest.raises(ValueError, match="shape"):
        accumulator.ingest_rollout(
            costs=np.zeros((1, 2, 2), dtype=np.float64),
            terminated=valid_done,
            truncated=valid_done,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        invalid = valid_costs.copy()
        invalid[0, 0, 0] = -0.01
        accumulator.ingest_rollout(
            costs=invalid,
            terminated=valid_done,
            truncated=valid_done,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        invalid = valid_costs.copy()
        invalid[0, 0, 0] = np.nan
        accumulator.ingest_rollout(
            costs=invalid,
            terminated=valid_done,
            truncated=valid_done,
        )
    with pytest.raises(ValueError, match="both terminated and truncated"):
        both = np.asarray([[True, False]], dtype=np.bool_)
        accumulator.ingest_rollout(
            costs=valid_costs,
            terminated=both,
            truncated=both,
        )


def test_completed_episode_accumulator_rejects_multiple_event_hits_per_episode() -> (
    None
):
    accumulator = CompletedEpisodeCostAccumulator(
        n_envs=1,
        schema=_aggregation_schema(),
    )

    with pytest.raises(ValueError, match="event cost"):
        accumulator.ingest_rollout(
            costs=np.asarray(
                [[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]],
                dtype=np.float64,
            ),
            terminated=np.asarray([[False], [True]], dtype=np.bool_),
            truncated=np.zeros((2, 1), dtype=np.bool_),
        )


def test_completed_episode_accumulator_rejects_mismatched_state() -> None:
    source = CompletedEpisodeCostAccumulator(
        n_envs=2,
        schema=_aggregation_schema(),
    )
    state = source.state_dict()

    wrong_env_count = CompletedEpisodeCostAccumulator(
        n_envs=1,
        schema=_aggregation_schema(),
    )
    with pytest.raises(ValueError, match="environment count"):
        wrong_env_count.load_state_dict(state)

    wrong_schema = CompletedEpisodeCostAccumulator(
        n_envs=2,
        schema=LagrangianSchema((_spec("drawdown_excess"),)),
    )
    with pytest.raises(ValueError, match="schema"):
        wrong_schema.load_state_dict(state)
