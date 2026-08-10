from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    BehaviorCloningGateThresholds,
    PathPerformanceMetrics,
    deterministic_bootstrap_lower_bound,
    evaluate_behavior_cloning_gates,
)
from trade_rl.learning.hierarchical_bc_metrics import (
    HierarchicalBehaviorCloningMetrics,
)


def _performance(net_return: float) -> PathPerformanceMetrics:
    return PathPerformanceMetrics(
        step_count=3,
        traded_step_count=2,
        trade_count=0,
        gross_return=net_return,
        net_return=net_return,
        reward_total=net_return,
        reward_mean=net_return / 3.0,
        trade_win_rate=None,
        positive_step_rate=2.0 / 3.0,
        turnover_total=0.4,
        turnover_mean=0.4 / 3.0,
        cost_total=0.01,
        cost_mean=0.01 / 3.0,
        maximum_drawdown=max(0.0, -net_return),
    )


def _evidence() -> ActionPathCollapseEvidence:
    return ActionPathCollapseEvidence(
        decision_count=3,
        action_dimension_count=1,
        active_dimension_count=3,
        inactive_dimension_count=0,
        proposal_distance_count=2,
        submitted_change_count=2,
        downstream_no_trade_suppression_count=0,
        execution_rejection_count=0,
        executed_change_count=2,
        trade_count=0,
        constant_submitted_actions=False,
    )


def _reconstruction_metrics() -> HierarchicalBehaviorCloningMetrics:
    return HierarchicalBehaviorCloningMetrics(
        active_support=16,
        positive_support=8,
        predicted_positive_support=8,
        gate_precision=0.8,
        gate_recall=0.8,
        gate_f1=0.8,
        active_target_rmse=0.05,
        composed_rmse=0.04,
        teacher_activity_rate=0.5,
        policy_activity_rate=0.5,
        activity_ratio=1.0,
        event_recalls=(0.8, 0.8, 0.8, 0.8),
        constant_action_collapse=False,
        all_hold_collapse=False,
        all_trade_collapse=False,
        insufficient_target_support=False,
    )


def _thresholds(
    *,
    minimum_episodes: int,
    minimum_net_return_lower_bound: float,
) -> BehaviorCloningGateThresholds:
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
        minimum_causal_holdout_episodes=minimum_episodes,
        maximum_causal_holdout_regret_upper_bound=0.2,
        minimum_causal_holdout_net_return_lower_bound=(
            minimum_net_return_lower_bound
        ),
    )


def _episode_holdout(
    *,
    episode_returns: tuple[float, ...],
    lower_bound: float,
) -> Any:
    records = tuple(
        SimpleNamespace(causal_policy_performance=_performance(value))
        for value in episode_returns
    )
    return SimpleNamespace(
        records=records,
        causal_policy_performance=_performance(min(episode_returns)),
        causal_policy_evidence=_evidence(),
        causal_net_return_lower_confidence_bound=lower_bound,
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=2_000,
        heldout_oracle_regret=0.0,
    )


def _metric(gates: Any, name: str) -> Any:
    return next(
        metric
        for metric in gates.causal_non_collapse_gate.metrics
        if metric.name == name
    )


def _evaluate(holdout: Any, thresholds: BehaviorCloningGateThresholds) -> Any:
    return evaluate_behavior_cloning_gates(
        initial_composed_loss=0.2,
        final_composed_loss=0.1,
        reconstruction_metrics=_reconstruction_metrics(),
        holdout=holdout,
        thresholds=thresholds,
    )


def test_bootstrap_lower_bound_is_deterministic_one_sided_and_accepts_losses() -> None:
    values = np.asarray([-0.08, -0.02, 0.01, 0.04, 0.07], dtype=np.float64)

    first = deterministic_bootstrap_lower_bound(
        values,
        confidence_level=0.95,
        resamples=2_000,
        seed_material="a" * 64,
    )
    second = deterministic_bootstrap_lower_bound(
        values,
        confidence_level=0.95,
        resamples=2_000,
        seed_material="a" * 64,
    )

    assert first == second
    assert first <= float(np.mean(values))


def test_bc_gate_rejects_insufficient_complete_episode_support() -> None:
    gates = _evaluate(
        _episode_holdout(episode_returns=(0.01, 0.02), lower_bound=0.0),
        _thresholds(minimum_episodes=3, minimum_net_return_lower_bound=-0.05),
    )

    metric = _metric(gates, "causal_net_return_lower_confidence_bound")
    assert metric.status == "insufficient_support"
    assert metric.support == 2
    assert metric.minimum_support == 3
    assert gates.passed is False


def test_bc_gate_rejects_after_cost_lower_bound_below_floor() -> None:
    gates = _evaluate(
        _episode_holdout(episode_returns=(0.01, 0.02), lower_bound=-0.08),
        _thresholds(minimum_episodes=2, minimum_net_return_lower_bound=-0.05),
    )

    metric = _metric(gates, "causal_net_return_lower_confidence_bound")
    assert metric.status == "failed"
    assert metric.observed == pytest.approx(-0.08)
    assert gates.passed is False
    with pytest.raises(RuntimeError, match="lower confidence bound"):
        gates.require_passed()


def test_legacy_single_path_holdout_uses_observed_net_return_as_bound() -> None:
    holdout = SimpleNamespace(
        causal_policy_performance=_performance(0.01),
        causal_policy_evidence=_evidence(),
        heldout_oracle_regret=0.0,
    )

    gates = _evaluate(
        holdout,
        _thresholds(minimum_episodes=1, minimum_net_return_lower_bound=0.0),
    )

    metric = _metric(gates, "causal_net_return_lower_confidence_bound")
    assert metric.status == "passed"
    assert metric.observed == pytest.approx(0.01)
    assert metric.support == 1
    assert gates.passed is True
