from __future__ import annotations

import numpy as np

from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4TargetConfig,
    causal_alpha_v4_target_path,
)
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    PathPerformanceMetrics,
)
from trade_rl.learning.rollout_evaluation import ActionPathEvaluation
from trade_rl.workflows.universal_causal_alpha_v4_replay import (
    build_causal_alpha_v4_replay_metric,
)


def _digest(char: str) -> str:
    return char * 64


def _target_path(*, change: bool):
    rows = 2
    if change:
        p4 = np.full(rows, 0.20, dtype=np.float64)
        p24 = np.full(rows, 0.20, dtype=np.float64)
        p72 = np.full(rows, 0.60, dtype=np.float64)
        direction = np.ones(rows, dtype=np.float64)
    else:
        p4 = np.zeros(rows, dtype=np.float64)
        p24 = np.zeros(rows, dtype=np.float64)
        p72 = np.zeros(rows, dtype=np.float64)
        direction = np.zeros(rows, dtype=np.float64)
    zeros = np.zeros(rows, dtype=np.float64)
    ones = np.ones(rows, dtype=np.float64)
    return causal_alpha_v4_target_path(
        p4,
        p24,
        p72,
        direction_score_4h=direction,
        uncertainty_4h=zeros,
        uncertainty_24h=zeros,
        uncertainty_72h=zeros,
        one_way_cost_rates=zeros,
        liquidity_weight_caps=ones,
        config=CausalAlphaV4TargetConfig(),
        initial_weight=0.0,
    )


def _evaluation(
    *,
    submitted: int,
    executed: int,
    closed: int,
    turnover: float,
    suppression: int = 0,
    rejection_reason_counts: tuple[tuple[str, int], ...] = (),
    risk_reason_counts: tuple[tuple[str, int], ...] = (),
    hard_risk: bool = False,
) -> ActionPathEvaluation:
    steps = 2
    rejection_count = sum(count for _, count in rejection_reason_counts)
    proposal_count = max(submitted, rejection_count)
    performance = PathPerformanceMetrics(
        step_count=steps,
        traded_step_count=executed,
        trade_count=closed,
        gross_return=0.02,
        net_return=0.015,
        reward_total=1.0,
        reward_mean=0.5,
        trade_win_rate=None if closed == 0 else 1.0,
        positive_step_rate=0.5,
        turnover_total=turnover,
        turnover_mean=turnover / steps,
        cost_total=0.005,
        cost_mean=0.0025,
        maximum_drawdown=0.03,
    )
    collapse = ActionPathCollapseEvidence(
        decision_count=steps,
        action_dimension_count=1,
        active_dimension_count=steps,
        inactive_dimension_count=0,
        proposal_distance_count=proposal_count,
        submitted_change_count=submitted,
        downstream_no_trade_suppression_count=suppression,
        execution_rejection_count=rejection_count,
        executed_change_count=executed,
        trade_count=closed,
        constant_submitted_actions=submitted == 0,
        execution_rejection_reason_counts=rejection_reason_counts,
        risk_projection_reason_counts=risk_reason_counts,
        hard_risk_violation=hard_risk,
    )
    return ActionPathEvaluation(
        actions=np.zeros((steps, 1), dtype=np.float32),
        performance=performance,
        collapse_evidence=collapse,
    )


def _metric(*, evaluation: ActionPathEvaluation, change: bool = True):
    return build_causal_alpha_v4_replay_metric(
        run_manifest_digest=_digest("a"),
        v4_context_manifest_digest=_digest("b"),
        config_digest=_digest("c"),
        symbol="ETHUSDT",
        episode_index=3,
        contract_digest=_digest("d"),
        fit_digest=_digest("e"),
        forecast_digest=_digest("f"),
        target_path=_target_path(change=change),
        evaluation=evaluation,
        episode_hours=720.0,
    )


def test_positive_fill_without_closed_trade_is_meaningful_execution() -> None:
    metric = _metric(
        evaluation=_evaluation(
            submitted=1,
            executed=1,
            closed=0,
            turnover=0.25,
        )
    )

    assert metric.executed_change_count == 1
    assert metric.closed_trade_count == 0
    assert metric.has_meaningful_execution is True
    assert metric.turnover_per_day > 0.0


def test_no_submitted_change_is_distinct_from_no_fill_after_submission() -> None:
    no_submission = _metric(
        evaluation=_evaluation(
            submitted=0,
            executed=0,
            closed=0,
            turnover=0.0,
        ),
        change=False,
    )
    suppressed = _metric(
        evaluation=_evaluation(
            submitted=1,
            executed=0,
            closed=0,
            turnover=0.0,
            suppression=1,
        )
    )

    assert no_submission.submitted_change_count == 0
    assert no_submission.downstream_no_trade_suppression_count == 0
    assert no_submission.has_meaningful_execution is False
    assert suppressed.submitted_change_count == 1
    assert suppressed.downstream_no_trade_suppression_count == 1
    assert suppressed.has_meaningful_execution is False


def test_execution_rejection_is_separate_from_downstream_suppression() -> None:
    metric = _metric(
        evaluation=_evaluation(
            submitted=1,
            executed=0,
            closed=0,
            turnover=0.0,
            rejection_reason_counts=(("venue_rejected", 1),),
        )
    )

    assert metric.downstream_no_trade_suppression_count == 0
    assert metric.execution_rejection_reason_counts == (("venue_rejected", 1),)
    assert metric.has_meaningful_execution is False


def test_hard_risk_violation_and_reason_attribution_are_preserved() -> None:
    metric = _metric(
        evaluation=_evaluation(
            submitted=1,
            executed=1,
            closed=1,
            turnover=0.30,
            risk_reason_counts=(("drawdown_projection", 1),),
            hard_risk=True,
        )
    )

    assert metric.hard_risk_violation is True
    assert metric.risk_projection_reason_counts == (("drawdown_projection", 1),)
    assert metric.closed_trade_count == 1
    assert metric.has_meaningful_execution is True


def test_replay_uses_simulator_performance_without_reconstructing_accounting() -> None:
    evaluation = _evaluation(
        submitted=1,
        executed=1,
        closed=0,
        turnover=0.45,
    )
    metric = _metric(evaluation=evaluation)

    assert metric.gross_return == evaluation.performance.gross_return
    assert metric.net_return == evaluation.performance.net_return
    assert metric.total_execution_cost == evaluation.performance.cost_total
    assert metric.maximum_drawdown == evaluation.performance.maximum_drawdown
    assert np.isclose(
        metric.turnover_per_day,
        evaluation.performance.turnover_total / 30.0,
    )


def test_total_turnover_above_tolerance_is_meaningful_without_executed_step() -> None:
    aggregate_fill = _evaluation(
        submitted=1,
        executed=0,
        closed=0,
        turnover=0.25,
    )
    below_tolerance = _evaluation(
        submitted=1,
        executed=0,
        closed=0,
        turnover=5e-7,
    )

    assert _metric(evaluation=aggregate_fill).has_meaningful_execution is True
    assert _metric(evaluation=below_tolerance).has_meaningful_execution is False
