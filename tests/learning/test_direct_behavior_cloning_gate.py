from __future__ import annotations

from types import SimpleNamespace

import pytest

from trade_rl.learning.direct_bc_evaluation import (
    evaluate_direct_behavior_cloning_gates,
)
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    BehaviorCloningGateThresholds,
)


def _thresholds() -> BehaviorCloningGateThresholds:
    return BehaviorCloningGateThresholds(
        minimum_composed_loss_relative_improvement=0.1,
        minimum_gate_precision=0.0,
        minimum_gate_recall=0.0,
        maximum_active_target_rmse=1.0,
        minimum_activity_ratio=0.0,
        maximum_activity_ratio=4.0,
        minimum_teacher_positive_support=1,
        minimum_causal_holdout_trades=2,
        maximum_causal_holdout_regret=0.2,
        minimum_causal_holdout_episodes=1,
        maximum_causal_holdout_regret_upper_bound=0.2,
    )


def _holdout(
    *,
    executed_changes: int,
    submitted_changes: int,
    constant_actions: bool,
    net_return: float,
) -> SimpleNamespace:
    evidence = ActionPathCollapseEvidence(
        decision_count=10,
        action_dimension_count=2,
        active_dimension_count=20,
        inactive_dimension_count=0,
        proposal_distance_count=10,
        submitted_change_count=submitted_changes,
        downstream_no_trade_suppression_count=0,
        execution_rejection_count=0,
        executed_change_count=executed_changes,
        trade_count=executed_changes,
        constant_submitted_actions=constant_actions,
    )
    return SimpleNamespace(
        causal_policy_evidence=evidence,
        causal_policy_performance=SimpleNamespace(net_return=net_return),
        heldout_oracle_regret=max(0.0, -net_return),
        records=(),
    )


def test_direct_behavior_cloning_gate_rejects_zero_trade_collapse() -> None:
    evaluation = evaluate_direct_behavior_cloning_gates(
        initial_mse=1.0,
        final_mse=0.5,
        teacher_change_support=10,
        holdout=_holdout(
            executed_changes=0,
            submitted_changes=0,
            constant_actions=True,
            net_return=0.0,
        ),
        thresholds=_thresholds(),
    )

    assert evaluation.teacher_reconstruction_gate.passed
    assert not evaluation.causal_non_collapse_gate.passed
    with pytest.raises(RuntimeError, match="zero-trade collapse"):
        evaluation.require_passed()


def test_direct_behavior_cloning_gate_accepts_reconstruction_and_causal_support() -> (
    None
):
    evaluation = evaluate_direct_behavior_cloning_gates(
        initial_mse=1.0,
        final_mse=0.5,
        teacher_change_support=10,
        holdout=_holdout(
            executed_changes=4,
            submitted_changes=5,
            constant_actions=False,
            net_return=0.01,
        ),
        thresholds=_thresholds(),
    )

    assert evaluation.passed
    assert tuple(
        metric.name for metric in evaluation.teacher_reconstruction_gate.metrics
    ) == ("action_mse_relative_improvement",)
    evaluation.require_passed()
