from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.learning.evaluation import (
    CAUSAL_GENERALIZATION_SCOPE,
    ORACLE_DIAGNOSTIC_SCOPE,
    OracleTeacherEvaluation,
    PathPerformanceMetrics,
    evaluate_behavior_cloning_holdout,
    evaluate_path_performance,
    write_learning_evaluation,
)


def _path(
    net: tuple[float, ...],
    *,
    gross: tuple[float, ...] | None = None,
    rewards: tuple[float, ...] | None = None,
    turnover: tuple[float, ...] | None = None,
    costs: tuple[float, ...] | None = None,
    closed_trade_pnls: tuple[float, ...] | None = None,
) -> PathPerformanceMetrics:
    count = len(net)
    return evaluate_path_performance(
        gross_step_returns=net if gross is None else gross,
        net_step_returns=net,
        rewards=net if rewards is None else rewards,
        turnover=(1.0,) * count if turnover is None else turnover,
        costs=(0.0,) * count if costs is None else costs,
        closed_trade_pnls=closed_trade_pnls,
    )


def test_path_metrics_cover_return_reward_trading_cost_and_drawdown() -> None:
    metrics = _path(
        (0.08, -0.10, 0.05),
        gross=(0.10, -0.08, 0.06),
        rewards=(1.0, -2.0, 0.5),
        turnover=(0.4, 0.0, 0.2),
        costs=(0.01, 0.0, 0.02),
        closed_trade_pnls=(12.0, -3.0, 0.0),
    )

    assert metrics.gross_return == pytest.approx(1.10 * 0.92 * 1.06 - 1.0)
    assert metrics.net_return == pytest.approx(1.08 * 0.90 * 1.05 - 1.0)
    assert metrics.reward_total == pytest.approx(-0.5)
    assert metrics.reward_mean == pytest.approx(-1 / 6)
    assert metrics.traded_step_count == 2
    assert metrics.trade_count == 3
    assert metrics.trade_win_rate == pytest.approx(1 / 3)
    assert metrics.positive_step_rate == pytest.approx(2 / 3)
    assert metrics.turnover_total == pytest.approx(0.6)
    assert metrics.cost_total == pytest.approx(0.03)
    assert metrics.maximum_drawdown == pytest.approx(0.10)


def test_no_trade_path_reports_null_trade_win_rate() -> None:
    metrics = _path((0.0, 0.0), turnover=(0.0, 0.0))

    assert metrics.trade_count == 0
    assert metrics.traded_step_count == 0
    assert metrics.trade_win_rate is None


def test_path_metrics_accept_closed_trade_tracker_aggregate_counts() -> None:
    metrics = evaluate_path_performance(
        gross_step_returns=(0.01, -0.02),
        net_step_returns=(0.009, -0.021),
        rewards=(0.1, -0.2),
        turnover=(0.2, 0.3),
        costs=(0.001, 0.001),
        closed_trade_count=3,
        winning_trade_count=2,
    )

    assert metrics.trade_count == 3
    assert metrics.trade_win_rate == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "extra",
    [
        {"closed_trade_count": 2},
        {"closed_trade_count": 1, "winning_trade_count": 2},
        {
            "closed_trade_pnls": (1.0,),
            "closed_trade_count": 1,
            "winning_trade_count": 1,
        },
    ],
)
def test_path_metrics_reject_invalid_closed_trade_aggregates(
    extra: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="trade"):
        evaluate_path_performance(
            gross_step_returns=(0.0,),
            net_step_returns=(0.0,),
            rewards=(0.0,),
            turnover=(0.0,),
            costs=(0.0,),
            **extra,
        )


def test_bc_holdout_keeps_hindsight_oracle_separate_from_causal_policy() -> None:
    oracle_performance = _path((0.10, -0.02, 0.04))
    policy_performance = _path((0.04, -0.03, 0.01))
    result = evaluate_behavior_cloning_holdout(
        teacher_train_range=(0, 5),
        heldout_range=(5, 9),
        oracle_actions=np.array([[0.4], [0.4], [0.0]], dtype=np.float32),
        policy_actions=np.array([[0.39], [0.2], [0.01]], dtype=np.float32),
        oracle_performance=oracle_performance,
        causal_policy_performance=policy_performance,
        action_tolerance=0.05,
    )

    assert result.action_agreement_rate == pytest.approx(2 / 3)
    assert result.action_mae == pytest.approx((0.01 + 0.20 + 0.01) / 3)
    assert result.heldout_oracle_regret == pytest.approx(
        oracle_performance.net_return - policy_performance.net_return
    )
    assert result.oracle_reference.scope == ORACLE_DIAGNOSTIC_SCOPE
    assert result.oracle_reference.uses_future_information is True
    assert result.oracle_reference.eligible_for_production_generalization is False
    assert result.policy_scope == CAUSAL_GENERALIZATION_SCOPE
    assert result.policy_uses_future_information is False
    assert result.oracle_comparison_is_production_evidence is False


def test_bc_holdout_rejects_training_overlap_and_wrong_action_range() -> None:
    performance = _path((0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="overlap"):
        evaluate_behavior_cloning_holdout(
            teacher_train_range=(0, 6),
            heldout_range=(4, 8),
            oracle_actions=np.zeros((3, 1)),
            policy_actions=np.zeros((3, 1)),
            oracle_performance=performance,
            causal_policy_performance=performance,
        )
    with pytest.raises(ValueError, match="exact heldout range"):
        evaluate_behavior_cloning_holdout(
            teacher_train_range=(0, 5),
            heldout_range=(5, 9),
            oracle_actions=np.zeros((2, 1)),
            policy_actions=np.zeros((2, 1)),
            oracle_performance=performance,
            causal_policy_performance=performance,
        )


def test_bc_holdout_allows_shared_boundary_bar_without_decision_overlap() -> None:
    performance = _path((0.01, 0.0, -0.01))

    result = evaluate_behavior_cloning_holdout(
        teacher_train_range=(0, 6),
        heldout_range=(5, 9),
        oracle_actions=np.zeros((3, 1)),
        policy_actions=np.zeros((3, 1)),
        oracle_performance=performance,
        causal_policy_performance=performance,
    )

    assert result.teacher_train_range == (0, 6)
    assert result.heldout_range == (5, 9)


def test_evaluation_json_is_atomic_digest_bound_and_scope_explicit(
    tmp_path: Path,
) -> None:
    performance = _path((0.02, -0.01))
    evaluation = OracleTeacherEvaluation(
        evaluation_range=(4, 7),
        performance=performance,
    )
    output = tmp_path / "oracle-evaluation.json"

    digest = write_learning_evaluation(output, evaluation)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["artifact_digest"] == digest
    assert payload["scope"] == ORACLE_DIAGNOSTIC_SCOPE
    assert payload["uses_future_information"] is True
    assert payload["eligible_for_production_generalization"] is False
    assert not (tmp_path / ".oracle-evaluation.json.tmp").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("gross_step_returns", (0.0, float("nan"))),
        ("net_step_returns", (0.0, -1.0)),
        ("turnover", (0.0, -0.1)),
        ("costs", (0.0, -0.1)),
    ],
)
def test_path_metrics_reject_invalid_accounting_inputs(field: str, value: object) -> None:
    values: dict[str, object] = {
        "gross_step_returns": (0.0, 0.0),
        "net_step_returns": (0.0, 0.0),
        "rewards": (0.0, 0.0),
        "turnover": (0.0, 0.0),
        "costs": (0.0, 0.0),
    }
    values[field] = value

    with pytest.raises(ValueError):
        evaluate_path_performance(
            gross_step_returns=values["gross_step_returns"],
            net_step_returns=values["net_step_returns"],
            rewards=values["rewards"],
            turnover=values["turnover"],
            costs=values["costs"],
        )
