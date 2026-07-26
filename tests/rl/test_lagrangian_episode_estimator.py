from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian import LagrangianSchema, canonical_lagrangian_schema
from trade_rl.rl.lagrangian_episode_estimator import (
    CompletionKind,
    TimeAwareCompletedEpisodeCostAccumulator,
    classify_completion_kind,
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
    )


def test_completion_classification_distinguishes_policy_completion_and_censoring() -> (
    None
):
    assert (
        classify_completion_kind(
            terminated=False,
            truncated=False,
            truncation_reason=None,
        )
        is CompletionKind.NONE
    )
    assert (
        classify_completion_kind(
            terminated=True,
            truncated=False,
            truncation_reason=None,
        )
        is CompletionKind.ECONOMIC_TERMINATION
    )
    assert (
        classify_completion_kind(
            terminated=False,
            truncated=True,
            truncation_reason="time_limit",
        )
        is CompletionKind.TIME_LIMIT_COMPLETION
    )
    assert (
        classify_completion_kind(
            terminated=False,
            truncated=True,
            truncation_reason="shadow_drawdown_stop",
        )
        is CompletionKind.CENSORED_EXTERNAL_TRUNCATION
    )


@pytest.mark.parametrize(
    ("terminated", "truncated", "reason"),
    [
        (True, True, None),
        (False, True, None),
        (False, True, "unknown_reason"),
        (True, False, "time_limit"),
        (False, False, "shadow_drawdown_stop"),
    ],
)
def test_completion_classification_fails_closed_on_inconsistent_metadata(
    terminated: bool,
    truncated: bool,
    reason: str | None,
) -> None:
    with pytest.raises(ValueError, match="completion"):
        classify_completion_kind(
            terminated=terminated,
            truncated=truncated,
            truncation_reason=reason,
        )


def test_time_aware_episode_aggregation_uses_canonical_units() -> None:
    accumulator = TimeAwareCompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    costs = np.asarray(
        [
            [[0.12, 0.0, 0.30, 0.0, 0.20, 1.0, 0.001]],
            [[0.24, 1.0, 0.10, 0.0, 0.60, 3.0, 0.002]],
        ],
        dtype=np.float64,
    )
    elapsed_hours = np.asarray([[6.0], [18.0]], dtype=np.float64)
    completion_kinds = np.asarray(
        [[CompletionKind.NONE], [CompletionKind.ECONOMIC_TERMINATION]],
        dtype=object,
    )

    batch = accumulator.ingest_rollout(
        costs=costs,
        transition_elapsed_hours=elapsed_hours,
        completion_kinds=completion_kinds,
    )

    assert batch.completed_episode_count == 1
    assert batch.censored_episode_count == 0
    expected = {
        "drawdown_excess": 0.12 * 6.0 / 24.0 + 0.24 * 18.0 / 24.0,
        "drawdown_stop_event": 1.0,
        "margin_deficit_fraction": 0.30 * 6.0 / 24.0 + 0.10 * 18.0 / 24.0,
        "forced_liquidation_event": 0.0,
        "gross_exposure_request_excess": 0.40,
        "daily_turnover": (1.0 * 6.0 + 3.0 * 18.0) / 24.0,
        "execution_cost_fraction": 0.003,
    }
    for name, value in expected.items():
        estimate = batch.estimates[name]
        assert estimate is not None
        assert estimate.denominator == 1
        assert estimate.value == pytest.approx(value)


def test_censored_episode_clears_environment_without_counting_safe_completion() -> None:
    accumulator = TimeAwareCompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    first = accumulator.ingest_rollout(
        costs=np.asarray([[[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        transition_elapsed_hours=np.asarray([[12.0]]),
        completion_kinds=np.asarray(
            [[CompletionKind.CENSORED_EXTERNAL_TRUNCATION]],
            dtype=object,
        ),
    )
    assert first.completed_episode_count == 0
    assert first.censored_episode_count == 1
    assert all(value is None for value in first.estimates.values())

    second = accumulator.ingest_rollout(
        costs=np.asarray([[[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        transition_elapsed_hours=np.asarray([[24.0]]),
        completion_kinds=np.asarray(
            [[CompletionKind.TIME_LIMIT_COMPLETION]],
            dtype=object,
        ),
    )
    estimate = second.estimates["drawdown_excess"]
    assert estimate is not None
    assert estimate.value == pytest.approx(0.2)
    assert accumulator.censored_episode_count == 1


def test_vector_environments_keep_unfinished_episode_state_isolated() -> None:
    accumulator = TimeAwareCompletedEpisodeCostAccumulator(n_envs=2, schema=_schema())
    batch = accumulator.ingest_rollout(
        costs=np.asarray(
            [
                [
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ),
        transition_elapsed_hours=np.asarray([[12.0, 6.0], [12.0, 18.0]]),
        completion_kinds=np.asarray(
            [
                [CompletionKind.NONE, CompletionKind.CENSORED_EXTERNAL_TRUNCATION],
                [CompletionKind.ECONOMIC_TERMINATION, CompletionKind.NONE],
            ],
            dtype=object,
        ),
    )
    estimate = batch.estimates["drawdown_excess"]
    assert estimate is not None
    assert estimate.denominator == 1
    assert estimate.value == pytest.approx(0.15)
    assert batch.censored_episode_count == 1

    next_batch = accumulator.ingest_rollout(
        costs=np.asarray(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        transition_elapsed_hours=np.asarray([[1.0, 24.0]]),
        completion_kinds=np.asarray(
            [[CompletionKind.NONE, CompletionKind.TIME_LIMIT_COMPLETION]],
            dtype=object,
        ),
    )
    second_estimate = next_batch.estimates["drawdown_excess"]
    assert second_estimate is not None
    assert second_estimate.value == pytest.approx(0.625)


@pytest.mark.parametrize("elapsed", [0.0, -1.0, float("nan"), float("inf")])
def test_episode_estimator_rejects_invalid_elapsed_time(elapsed: float) -> None:
    accumulator = TimeAwareCompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    with pytest.raises(ValueError, match="elapsed"):
        accumulator.ingest_rollout(
            costs=np.zeros((1, 1, len(CONSTRAINT_COST_NAMES))),
            transition_elapsed_hours=np.asarray([[elapsed]]),
            completion_kinds=np.asarray([[CompletionKind.NONE]], dtype=object),
        )


def test_episode_estimator_state_roundtrip_reproduces_next_completion() -> None:
    accumulator = TimeAwareCompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    accumulator.ingest_rollout(
        costs=np.asarray([[[0.4, 0.0, 0.2, 0.0, 0.3, 2.0, 0.001]]]),
        transition_elapsed_hours=np.asarray([[6.0]]),
        completion_kinds=np.asarray([[CompletionKind.NONE]], dtype=object),
    )
    restored = TimeAwareCompletedEpisodeCostAccumulator(n_envs=1, schema=_schema())
    restored.load_state_dict(accumulator.state_dict())

    kwargs = {
        "costs": np.asarray([[[0.2, 0.0, 0.1, 0.0, 0.5, 4.0, 0.002]]]),
        "transition_elapsed_hours": np.asarray([[18.0]]),
        "completion_kinds": np.asarray(
            [[CompletionKind.ECONOMIC_TERMINATION]],
            dtype=object,
        ),
    }
    assert restored.ingest_rollout(**kwargs) == accumulator.ingest_rollout(**kwargs)
