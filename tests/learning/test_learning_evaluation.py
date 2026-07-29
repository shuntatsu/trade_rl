from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.learning.evaluation import (
    CAUSAL_GENERALIZATION_SCOPE,
    ORACLE_DIAGNOSTIC_SCOPE,
    BehaviorCloningGateThresholds,
    BehaviorCloningHoldoutEvaluation,
    OracleTeacherEvaluation,
    PathPerformanceMetrics,
    evaluate_behavior_cloning_holdout,
    evaluate_path_performance,
    write_learning_evaluation,
)
from trade_rl.learning.hierarchical_bc_metrics import (
    HierarchicalBehaviorCloningMetrics,
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
def test_path_metrics_reject_invalid_accounting_inputs(
    field: str, value: object
) -> None:
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


def _hierarchical_metrics(
    *,
    positive_support: int = 8,
    predicted_positive_support: int = 8,
    precision: float = 0.8,
    recall: float = 0.8,
    target_rmse: float | None = 0.05,
    activity_ratio: float | None = 1.0,
    constant: bool = False,
    all_hold: bool = False,
    all_trade: bool = False,
) -> HierarchicalBehaviorCloningMetrics:
    return HierarchicalBehaviorCloningMetrics(
        active_support=16,
        positive_support=positive_support,
        predicted_positive_support=predicted_positive_support,
        gate_precision=precision,
        gate_recall=recall,
        gate_f1=0.8,
        active_target_rmse=target_rmse,
        composed_rmse=0.04,
        teacher_activity_rate=positive_support / 16,
        policy_activity_rate=predicted_positive_support / 16,
        activity_ratio=activity_ratio,
        event_recalls=(0.8, 0.8, 0.8, 0.8),
        constant_action_collapse=constant,
        all_hold_collapse=all_hold,
        all_trade_collapse=all_trade,
        insufficient_target_support=positive_support == 0,
    )


def _gate_thresholds() -> BehaviorCloningGateThresholds:
    return BehaviorCloningGateThresholds(
        minimum_composed_loss_relative_improvement=0.1,
        minimum_gate_precision=0.6,
        minimum_gate_recall=0.6,
        maximum_active_target_rmse=0.1,
        minimum_activity_ratio=0.5,
        maximum_activity_ratio=1.5,
        minimum_teacher_positive_support=2,
        minimum_causal_holdout_trades=1,
        maximum_causal_holdout_regret=0.2,
    )


def _holdout_with_evidence(
    *,
    submitted: int,
    executed: int,
    constant: bool,
    net: tuple[float, ...] = (0.01, 0.0, -0.005),
) -> BehaviorCloningHoldoutEvaluation:
    from trade_rl.learning.evaluation import ActionPathCollapseEvidence

    oracle = _path((0.02, 0.01, 0.0))
    policy = _path(
        net,
        turnover=tuple(0.2 if index < executed else 0.0 for index in range(3)),
    )
    evidence = ActionPathCollapseEvidence(
        decision_count=3,
        action_dimension_count=1,
        active_dimension_count=3,
        inactive_dimension_count=0,
        proposal_distance_count=submitted,
        submitted_change_count=submitted,
        downstream_no_trade_suppression_count=max(0, submitted - executed),
        execution_rejection_count=0,
        executed_change_count=executed,
        trade_count=policy.trade_count,
        constant_submitted_actions=constant,
    )
    return evaluate_behavior_cloning_holdout(
        teacher_train_range=(0, 5),
        heldout_range=(5, 9),
        oracle_actions=np.array([[0.5], [-0.5], [0.0]], dtype=np.float32),
        policy_actions=np.array([[0.2], [0.2], [0.2]], dtype=np.float32),
        oracle_performance=oracle,
        causal_policy_performance=policy,
        causal_policy_evidence=evidence,
    )


def test_mandatory_bc_gates_pass_without_requiring_oracle_agreement() -> None:
    from trade_rl.learning.evaluation import evaluate_behavior_cloning_gates

    holdout = _holdout_with_evidence(submitted=2, executed=2, constant=False)
    assert holdout.action_agreement_rate == 0.0

    gates = evaluate_behavior_cloning_gates(
        initial_composed_loss=0.2,
        final_composed_loss=0.1,
        reconstruction_metrics=_hierarchical_metrics(),
        holdout=holdout,
        thresholds=_gate_thresholds(),
    )

    assert gates.teacher_reconstruction_gate.required is True
    assert gates.causal_non_collapse_gate.required is True
    assert gates.passed is True
    gates.require_passed()


def test_mandatory_bc_gate_rejects_zero_trade_causal_holdout() -> None:
    from trade_rl.learning.evaluation import evaluate_behavior_cloning_gates

    gates = evaluate_behavior_cloning_gates(
        initial_composed_loss=0.2,
        final_composed_loss=0.1,
        reconstruction_metrics=_hierarchical_metrics(),
        holdout=_holdout_with_evidence(submitted=0, executed=0, constant=True),
        thresholds=_gate_thresholds(),
    )

    assert gates.teacher_reconstruction_gate.passed is True
    assert gates.causal_non_collapse_gate.passed is False
    with pytest.raises(RuntimeError, match="zero-trade collapse"):
        gates.require_passed()


def test_mandatory_bc_gate_reports_insufficient_teacher_support_as_failure() -> None:
    from trade_rl.learning.evaluation import evaluate_behavior_cloning_gates

    gates = evaluate_behavior_cloning_gates(
        initial_composed_loss=0.2,
        final_composed_loss=0.1,
        reconstruction_metrics=_hierarchical_metrics(
            positive_support=0,
            predicted_positive_support=0,
            precision=0.0,
            recall=0.0,
            target_rmse=None,
            activity_ratio=1.0,
            constant=True,
            all_hold=True,
        ),
        holdout=_holdout_with_evidence(submitted=0, executed=0, constant=True),
        thresholds=_gate_thresholds(),
    )

    statuses = {metric.status for metric in gates.teacher_reconstruction_gate.metrics}
    assert "insufficient_support" in statuses
    assert gates.passed is False


def test_mandatory_bc_gate_rejects_catastrophic_cash_baseline_regret() -> None:
    from trade_rl.learning.evaluation import evaluate_behavior_cloning_gates

    gates = evaluate_behavior_cloning_gates(
        initial_composed_loss=0.2,
        final_composed_loss=0.1,
        reconstruction_metrics=_hierarchical_metrics(),
        holdout=_holdout_with_evidence(
            submitted=2,
            executed=2,
            constant=False,
            net=(-0.1, -0.1, -0.1),
        ),
        thresholds=_gate_thresholds(),
    )

    metric = next(
        item
        for item in gates.causal_non_collapse_gate.metrics
        if item.name == "cash_baseline_after_cost_regret"
    )
    assert metric.status == "failed"
    assert gates.passed is False
