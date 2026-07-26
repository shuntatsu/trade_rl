from __future__ import annotations

import copy

import numpy as np
import pytest

from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian import (
    CompletedEpisodeBatch,
    CompletedEpisodeCostAccumulator,
    ConstraintAggregation,
    LagrangianSchema,
    canonical_constraint_aggregation,
    canonical_constraint_unit,
    canonical_lagrangian_schema,
)
from trade_rl.rl.lagrangian_episode import (
    EpisodeCompletionKind,
    classify_episode_completion,
)


@pytest.mark.parametrize(
    ("terminated", "truncated", "time_limit", "reason", "expected"),
    [
        (False, False, False, None, EpisodeCompletionKind.NONE),
        (
            True,
            False,
            False,
            "margin_call",
            EpisodeCompletionKind.ECONOMIC_TERMINATION,
        ),
        (
            False,
            True,
            True,
            None,
            EpisodeCompletionKind.TIME_LIMIT_COMPLETION,
        ),
        (
            False,
            True,
            True,
            "shadow_minimum_equity",
            EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION,
        ),
    ],
)
def test_episode_completion_classification(
    terminated: bool,
    truncated: bool,
    time_limit: bool,
    reason: str | None,
    expected: EpisodeCompletionKind,
) -> None:
    assert (
        classify_episode_completion(
            terminated=terminated,
            truncated=truncated,
            time_limit_truncated=time_limit,
            termination_reason=reason,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("terminated", "truncated", "time_limit", "reason", "message"),
    [
        (True, True, True, "margin_call", "both terminated and truncated"),
        (False, True, False, None, "TimeLimit.truncated"),
        (False, True, True, "manual_reset", "unknown truncation reason"),
        (False, False, False, "shadow_minimum_equity", "shadow"),
        (True, False, False, "shadow_minimum_equity", "shadow"),
        (False, False, True, None, "time-limit flag"),
    ],
)
def test_episode_completion_classification_fails_closed(
    terminated: bool,
    truncated: bool,
    time_limit: bool,
    reason: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        classify_episode_completion(
            terminated=terminated,
            truncated=truncated,
            time_limit_truncated=time_limit,
            termination_reason=reason,
        )


def test_episode_completion_accepts_enum_like_reason() -> None:
    class _Reason:
        value = "margin_call"

    assert (
        classify_episode_completion(
            terminated=True,
            truncated=False,
            time_limit_truncated=False,
            termination_reason=_Reason(),
        )
        is EpisodeCompletionKind.ECONOMIC_TERMINATION
    )


def _schema() -> LagrangianSchema:
    count = len(CONSTRAINT_COST_NAMES)
    return canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=(0.0,) * count,
        dual_learning_rates=(0.1,) * count,
        ema_betas=(0.9,) * count,
        initial_multipliers=(0.0,) * count,
        max_multipliers=(10.0,) * count,
        warmup_rollouts=(0,) * count,
        update_interval_rollouts=(1,) * count,
        minimum_completed_episodes=(1,) * count,
    )


def _cost_rollout(**columns: list[float]) -> np.ndarray:
    steps = len(columns["drawdown_excess"])
    matrix = np.column_stack(
        [columns.get(name, [0.0] * steps) for name in CONSTRAINT_COST_NAMES]
    )
    return matrix[:, None, :].astype(np.float64)


def test_canonical_constraint_aggregation_and_units_are_explicit() -> None:
    expected_aggregations = {
        "drawdown_excess": ConstraintAggregation.EPISODE_TIME_AREA,
        "drawdown_stop_event": ConstraintAggregation.EPISODE_EVENT_RATE,
        "margin_deficit_fraction": ConstraintAggregation.EPISODE_TIME_AREA,
        "forced_liquidation_event": ConstraintAggregation.EPISODE_EVENT_RATE,
        "gross_exposure_request_excess": ConstraintAggregation.EPISODE_DECISION_MEAN,
        "daily_turnover": ConstraintAggregation.EPISODE_TIME_WEIGHTED_MEAN,
        "execution_cost_fraction": ConstraintAggregation.EPISODE_SUM,
    }
    expected_units = {
        "drawdown_excess": "drawdown_excess_area_days",
        "drawdown_stop_event": "event_per_episode",
        "margin_deficit_fraction": "margin_deficit_fraction_days",
        "forced_liquidation_event": "event_per_episode",
        "gross_exposure_request_excess": "excess_per_decision",
        "daily_turnover": "turnover_per_day",
        "execution_cost_fraction": "execution_cost_fraction_per_episode",
    }

    assert {
        name: canonical_constraint_aggregation(name) for name in CONSTRAINT_COST_NAMES
    } == expected_aggregations
    assert {
        name: canonical_constraint_unit(name) for name in CONSTRAINT_COST_NAMES
    } == expected_units
    for spec in _schema().specs:
        assert spec.digest_payload()["unit"] == expected_units[spec.name]


def test_completed_episode_statistics_are_time_aware() -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    result = accumulator.ingest_rollout(
        costs=_cost_rollout(
            drawdown_excess=[0.10, 0.20],
            drawdown_stop_event=[0.0, 1.0],
            margin_deficit_fraction=[0.04, 0.08],
            gross_exposure_request_excess=[0.30, 0.10],
            daily_turnover=[2.0, 4.0],
            execution_cost_fraction=[0.001, 0.002],
        ),
        elapsed_hours=np.asarray([[6.0], [18.0]], dtype=np.float64),
        completion_kinds=np.asarray(
            [
                [EpisodeCompletionKind.NONE],
                [EpisodeCompletionKind.TIME_LIMIT_COMPLETION],
            ],
            dtype=np.int8,
        ),
    )

    assert isinstance(result, CompletedEpisodeBatch)
    assert result.completed_episode_count == 1
    assert result.censored_episode_count == 0
    expected = {
        "drawdown_excess": 0.175,
        "drawdown_stop_event": 1.0,
        "margin_deficit_fraction": 0.07,
        "gross_exposure_request_excess": 0.20,
        "daily_turnover": 3.5,
        "execution_cost_fraction": 0.003,
    }
    for name, value in expected.items():
        estimate = result.estimates[name]
        assert estimate is not None
        assert estimate.value == pytest.approx(value)


def test_shadow_truncation_clears_state_without_safe_denominator() -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    result = accumulator.ingest_rollout(
        costs=_cost_rollout(
            drawdown_excess=[0.10, 0.20],
            drawdown_stop_event=[0.0, 1.0],
            daily_turnover=[3.0, 4.0],
        ),
        elapsed_hours=np.asarray([[1.0], [1.0]], dtype=np.float64),
        completion_kinds=np.asarray(
            [
                [EpisodeCompletionKind.NONE],
                [EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION],
            ],
            dtype=np.int8,
        ),
    )

    assert result.completed_episode_count == 0
    assert result.censored_episode_count == 1
    assert all(value is None for value in result.estimates.values())
    state = accumulator.state_dict()
    assert state["episode_step_counts"] == [0]
    assert state["episode_elapsed_hours"] == [0.0]
    assert np.asarray(state["episode_cost_sums"]).sum() == pytest.approx(0.0)
    assert np.asarray(state["episode_time_weighted_sums"]).sum() == pytest.approx(0.0)


def test_valid_completions_are_isolated_per_environment() -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=2, schema=_schema())
    costs = np.zeros((1, 2, len(CONSTRAINT_COST_NAMES)), dtype=np.float64)
    drawdown_index = CONSTRAINT_COST_NAMES.index("drawdown_excess")
    costs[0, :, drawdown_index] = [0.10, 0.20]

    first = accumulator.ingest_rollout(
        costs=costs,
        elapsed_hours=np.asarray([[6.0, 6.0]], dtype=np.float64),
        completion_kinds=np.asarray(
            [
                [
                    EpisodeCompletionKind.ECONOMIC_TERMINATION,
                    EpisodeCompletionKind.NONE,
                ]
            ],
            dtype=np.int8,
        ),
    )
    assert first.completed_episode_count == 1
    state = accumulator.state_dict()
    assert state["episode_step_counts"] == [0, 1]
    assert state["episode_elapsed_hours"] == [0.0, 6.0]

    second_costs = np.zeros_like(costs)
    second_costs[0, :, drawdown_index] = [0.0, 0.40]
    second = accumulator.ingest_rollout(
        costs=second_costs,
        elapsed_hours=np.asarray([[1.0, 18.0]], dtype=np.float64),
        completion_kinds=np.asarray(
            [
                [
                    EpisodeCompletionKind.NONE,
                    EpisodeCompletionKind.TIME_LIMIT_COMPLETION,
                ]
            ],
            dtype=np.int8,
        ),
    )
    estimate = second.estimates["drawdown_excess"]
    assert estimate is not None
    assert estimate.value == pytest.approx(0.20 * 6.0 / 24.0 + 0.40 * 18.0 / 24.0)


def test_rollout_boundary_state_round_trip_preserves_elapsed_time() -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    first = accumulator.ingest_rollout(
        costs=_cost_rollout(
            drawdown_excess=[0.10],
            gross_exposure_request_excess=[0.30],
            daily_turnover=[2.0],
        ),
        elapsed_hours=np.asarray([[6.0]], dtype=np.float64),
        completion_kinds=np.asarray([[EpisodeCompletionKind.NONE]], dtype=np.int8),
    )
    assert all(value is None for value in first.estimates.values())

    state = accumulator.state_dict()
    restored = CompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    costs = _cost_rollout(
        drawdown_excess=[0.20],
        gross_exposure_request_excess=[0.10],
        daily_turnover=[4.0],
    )
    elapsed = np.asarray([[18.0]], dtype=np.float64)
    kinds = np.asarray([[EpisodeCompletionKind.TIME_LIMIT_COMPLETION]], dtype=np.int8)
    assert restored.ingest_rollout(
        costs=costs,
        elapsed_hours=elapsed,
        completion_kinds=kinds,
    ) == accumulator.ingest_rollout(
        costs=costs,
        elapsed_hours=elapsed,
        completion_kinds=kinds,
    )


def test_old_accumulator_state_version_fails_closed() -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    state = copy.deepcopy(accumulator.state_dict())
    state["schema_version"] = "completed_episode_cost_accumulator_v1"

    with pytest.raises(ValueError, match="schema version"):
        accumulator.load_state_dict(state)


@pytest.mark.parametrize("elapsed", [0.0, -1.0, np.nan, np.inf])
def test_completed_episode_accumulator_rejects_invalid_elapsed_hours(
    elapsed: float,
) -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    with pytest.raises(ValueError, match="elapsed hours"):
        accumulator.ingest_rollout(
            costs=_cost_rollout(drawdown_excess=[0.0]),
            elapsed_hours=np.asarray([[elapsed]], dtype=np.float64),
            completion_kinds=np.asarray([[EpisodeCompletionKind.NONE]], dtype=np.int8),
        )


def test_completed_episode_accumulator_rejects_unknown_completion_kind() -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    with pytest.raises(ValueError, match="completion kind"):
        accumulator.ingest_rollout(
            costs=_cost_rollout(drawdown_excess=[0.0]),
            elapsed_hours=np.asarray([[1.0]], dtype=np.float64),
            completion_kinds=np.asarray([[99]], dtype=np.int8),
        )


def test_completed_episode_accumulator_rejects_multiple_event_hits() -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    with pytest.raises(ValueError, match="occurred more than once"):
        accumulator.ingest_rollout(
            costs=_cost_rollout(
                drawdown_excess=[0.0, 0.0],
                drawdown_stop_event=[1.0, 1.0],
            ),
            elapsed_hours=np.asarray([[1.0], [1.0]], dtype=np.float64),
            completion_kinds=np.asarray(
                [
                    [EpisodeCompletionKind.NONE],
                    [EpisodeCompletionKind.ECONOMIC_TERMINATION],
                ],
                dtype=np.int8,
            ),
        )
