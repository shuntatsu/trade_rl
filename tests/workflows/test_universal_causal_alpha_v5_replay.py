from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v5 import CausalAlphaV5TargetPath
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    PathPerformanceMetrics,
)
from trade_rl.learning.rollout_evaluation import ActionPathEvaluation
from trade_rl.workflows.universal_causal_alpha_v5_replay import (
    build_causal_alpha_v5_replay_metric,
)


def _digest(char: str) -> str:
    return char * 64


def _target(
    weights: tuple[float, ...], *, active: tuple[bool, ...], initial: float = 0.0
):
    rows = len(weights)
    targets = np.asarray(weights)
    reasons: list[str] = []
    previous = initial
    for target in targets:
        if target == previous:
            reasons.append("hold_flat" if target == 0.0 else "hold_position")
        elif previous == 0.0:
            reasons.append("entry")
        elif target == 0.0:
            reasons.append("exit")
        elif previous * target < 0.0:
            reasons.append("flip")
        elif abs(target) > abs(previous):
            reasons.append("add")
        else:
            reasons.append("reduce")
        previous = float(target)
    zeros = np.zeros(rows)
    ones = np.ones(rows)
    return CausalAlphaV5TargetPath(
        initial_weight=initial,
        slow_anchors=targets,
        fast_deviations=zeros,
        targets=targets,
        slow_expected_returns=zeros,
        fast_expected_returns=zeros,
        slow_uncertainties=zeros,
        fast_uncertainties=zeros,
        liquidity_weight_caps=ones,
        risk_weight_caps=ones,
        slow_objectives=zeros,
        fast_objective_improvements=zeros,
        final_objectives=zeros,
        active_mask=np.asarray(active),
        reasons=tuple(reasons),
        reason_counts=tuple(
            sorted((reason, reasons.count(reason)) for reason in set(reasons))
        ),
        slow_anchor_change_count=sum(
            left != right for left, right in zip((initial, *weights), weights)
        ),
        fast_impulse_change_count=0,
        submitted_change_count=sum(
            left != right for left, right in zip((initial, *weights), weights)
        ),
        liquidity_deleveraging_count=0,
        risk_projection_count=0,
        sign_flip_count=sum(
            left * right < 0 for left, right in zip((initial, *weights), weights)
        ),
        selective_forecast_digest=_digest("a"),
        config_digest=_digest("b"),
    )


def _evaluation(
    rows: int, *, executed: int = 2, turnover: float = 0.4
) -> ActionPathEvaluation:
    performance = PathPerformanceMetrics(
        step_count=rows,
        traded_step_count=executed,
        trade_count=1,
        gross_return=0.04,
        net_return=0.03,
        reward_total=0.03,
        reward_mean=0.03 / rows,
        trade_win_rate=1.0,
        positive_step_rate=0.5,
        turnover_total=turnover,
        turnover_mean=turnover / rows,
        cost_total=0.01,
        cost_mean=0.01 / rows,
        maximum_drawdown=0.02,
    )
    collapse = ActionPathCollapseEvidence(
        decision_count=rows,
        action_dimension_count=1,
        active_dimension_count=rows,
        inactive_dimension_count=0,
        proposal_distance_count=2,
        submitted_change_count=2,
        downstream_no_trade_suppression_count=0,
        execution_rejection_count=0,
        executed_change_count=executed,
        trade_count=1,
        constant_submitted_actions=False,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        hard_risk_violation=False,
    )
    return ActionPathEvaluation(
        actions=np.zeros((rows, 1), dtype=np.float32),
        performance=performance,
        collapse_evidence=collapse,
    )


def _metric(
    path: CausalAlphaV5TargetPath, evaluation: ActionPathEvaluation, hours: float = 24.0
):
    return build_causal_alpha_v5_replay_metric(
        run_manifest_digest=_digest("1"),
        v4_context_manifest_digest=_digest("2"),
        config_digest=_digest("3"),
        symbol="BTCUSDT",
        episode_index=0,
        contract_digest=_digest("4"),
        fit_digest=_digest("5"),
        forecast_digest=_digest("6"),
        calibration_fit_digest=_digest("7"),
        target_path=path,
        evaluation=evaluation,
        episode_hours=hours,
    )


def test_v5_replay_uses_authoritative_simulator_values_and_attribution() -> None:
    path = _target((0.0, 0.2, 0.2, 0.0), active=(False, True, True, False))
    evaluation = _evaluation(4)
    metric = _metric(path, evaluation)
    assert metric.gross_return == evaluation.performance.gross_return
    assert metric.net_return == evaluation.performance.net_return
    assert metric.total_execution_cost == evaluation.performance.cost_total
    assert metric.maximum_drawdown == evaluation.performance.maximum_drawdown
    assert metric.turnover_per_day == evaluation.performance.turnover_total
    assert metric.active_coverage == 0.5
    assert metric.flat_time_fraction == 0.5
    assert metric.time_weighted_absolute_exposure == 0.1
    assert metric.completed_holding_durations_hours == (12.0,)
    assert not metric.has_unclosed_position
    assert dict(metric.target_reason_counts) == {
        "entry": 1,
        "exit": 1,
        "hold_flat": 1,
        "hold_position": 1,
    }


def test_v5_replay_all_flat_and_unclosed_position_semantics() -> None:
    flat_path = _target((0.0, 0.0), active=(False, False))
    flat = _metric(flat_path, _evaluation(2, executed=0, turnover=0.0))
    assert flat.flat_time_fraction == 1.0
    assert flat.completed_holding_durations_hours == ()
    assert not flat.has_unclosed_position
    open_path = _target((0.2, 0.2), active=(True, True))
    opened = _metric(open_path, _evaluation(2))
    assert opened.completed_holding_durations_hours == ()
    assert opened.has_unclosed_position


def test_v5_flip_closes_old_holding_and_opens_new_holding() -> None:
    path = _target((0.2, -0.2, 0.0), active=(True, True, False))
    metric = _metric(path, _evaluation(3), hours=18.0)
    assert metric.completed_holding_durations_hours == (6.0, 6.0)
    assert metric.sign_flip_count == 1
    assert not metric.has_unclosed_position


def test_v5_replay_rejects_zero_hours_length_drift_and_false_execution() -> None:
    path = _target((0.2, 0.0), active=(True, False))
    with pytest.raises(ValueError, match="episode_hours"):
        _metric(path, _evaluation(2), hours=0.0)
    with pytest.raises(ValueError, match="cover"):
        _metric(path, _evaluation(3))
    metric = _metric(path, _evaluation(2))
    with pytest.raises(ValueError, match="meaningful"):
        replace(metric, has_meaningful_execution=False, digest="")
    with pytest.raises(ValueError, match="malformed"):
        replace(metric, target_reason_counts=(("exit", 1), ("entry", 1)), digest="")
