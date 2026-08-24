from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetConfig,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    PathPerformanceMetrics,
)
from trade_rl.learning.rollout_evaluation import ActionPathEvaluation
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    build_causal_alpha_v6_replay_metric,
)


def _digest(char: str) -> str:
    return char * 64


def _target(
    weights: tuple[float, ...],
    *,
    initial: float = 0.0,
    candidate: CausalAlphaV6Candidate = CausalAlphaV6Candidate.FAST_ONLY,
) -> CausalAlphaV6TargetPath:
    targets = np.asarray(weights)
    rows = len(weights)
    previous = initial
    reasons: list[str] = []
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
    caps = np.full(rows, 0.25)
    previous_targets = np.asarray((initial, *weights[:-1]))
    return CausalAlphaV6TargetPath(
        candidate=candidate,
        initial_weight=initial,
        decision_indices=np.arange(rows),
        targets=targets,
        fast_proposals=targets,
        expected_returns_4h=zeros,
        expected_returns_24h=zeros,
        expected_returns_72h=zeros,
        direction_scores_4h=zeros,
        uncertainties_4h=zeros,
        one_way_cost_rates=zeros,
        liquidity_weight_caps=caps,
        risk_weight_caps=caps,
        objectives=zeros,
        confirmation_counts=np.zeros(rows, dtype=np.int64),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        slow_states=tuple(
            CausalAlphaV6SlowState.FLAT if target == 0.0 else CausalAlphaV6SlowState.MIXED
            for target in previous_targets
        ),
        reasons=tuple(reasons),
        reason_counts=tuple(
            sorted((reason, reasons.count(reason)) for reason in set(reasons))
        ),
        submitted_change_count=int(np.count_nonzero(targets != previous_targets)),
        sign_flip_count=int(np.count_nonzero(targets * previous_targets < 0.0)),
        liquidity_deleveraging_count=0,
        risk_projection_count=0,
        forecast_digest=_digest("a"),
        config_digest=CausalAlphaV6TargetConfig().digest,
    )


def _evaluation(
    rows: int,
    *,
    gross_return: float = 0.04,
    net_return: float = 0.03,
    reward_scale: float = 2.0,
    executed: int = 2,
    turnover: float = 0.4,
    rejections: tuple[tuple[str, int], ...] = (),
    risk: tuple[tuple[str, int], ...] = (),
    hard_risk: bool = False,
) -> ActionPathEvaluation:
    rejection_count = sum(count for _, count in rejections)
    performance = PathPerformanceMetrics(
        step_count=rows,
        traded_step_count=executed,
        trade_count=1 if executed else 0,
        gross_return=gross_return,
        net_return=net_return,
        reward_total=net_return * reward_scale,
        reward_mean=net_return * reward_scale / rows,
        trade_win_rate=1.0 if executed else 0.0,
        positive_step_rate=0.5,
        turnover_total=turnover,
        turnover_mean=turnover / rows,
        cost_total=max(0.0, gross_return - net_return),
        cost_mean=max(0.0, gross_return - net_return) / rows,
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
        execution_rejection_count=rejection_count,
        executed_change_count=executed,
        trade_count=performance.trade_count,
        constant_submitted_actions=False,
        execution_rejection_reason_counts=rejections,
        risk_projection_reason_counts=risk,
        hard_risk_violation=hard_risk,
    )
    return ActionPathEvaluation(
        actions=np.zeros((rows, 1), dtype=np.float32),
        performance=performance,
        collapse_evidence=collapse,
    )


def _metric(
    path: CausalAlphaV6TargetPath,
    evaluation: ActionPathEvaluation,
    *,
    hours: float = 24.0,
    reward_scale: float = 2.0,
):
    return build_causal_alpha_v6_replay_metric(
        run_manifest_digest=_digest("1"),
        v4_context_manifest_digest=_digest("2"),
        symbol="BTCUSDT",
        episode_index=0,
        contract_digest=_digest("3"),
        fit_digest=_digest("4"),
        forecast_digest=_digest("a"),
        target_path=path,
        evaluation=evaluation,
        episode_hours=hours,
        reward_scale=reward_scale,
    )


def test_v6_replay_preserves_authoritative_wealth_reward_and_holdings() -> None:
    path = _target((0.0, 0.2, 0.2, 0.0))
    metric = _metric(path, _evaluation(4))
    assert metric.gross_wealth == pytest.approx(math.exp(metric.gross_return))
    assert metric.net_wealth == pytest.approx(math.exp(metric.net_return))
    assert metric.reward_total == pytest.approx(metric.net_return * 2.0)
    assert metric.completed_holding_durations_hours == (12.0,)
    assert metric.open_holding_duration_hours == 0.0
    assert metric.turnover_per_day == 0.4
    assert metric.total_execution_cost == pytest.approx(0.01)


def test_v6_replay_all_flat_and_open_position_semantics() -> None:
    flat = _metric(_target((0.0, 0.0)), _evaluation(2, executed=0, turnover=0.0))
    assert flat.flat_time_fraction == 1.0
    assert flat.completed_holding_durations_hours == ()
    assert flat.open_holding_duration_hours == 0.0
    opened = _metric(_target((0.0, 0.0, 0.2)), _evaluation(3), hours=18.0)
    assert opened.completed_holding_durations_hours == ()
    assert opened.open_holding_duration_hours == 6.0


def test_v6_replay_flip_closes_and_opens_independent_holdings() -> None:
    metric = _metric(_target((0.2, -0.2, 0.0)), _evaluation(3), hours=18.0)
    assert metric.completed_holding_durations_hours == (6.0, 6.0)
    assert metric.sign_flip_count == 1
    assert metric.open_holding_duration_hours == 0.0


def test_v6_replay_preserves_rejection_and_hard_risk_evidence() -> None:
    evaluation = _evaluation(
        2,
        rejections=(("venue_rejected", 1),),
        risk=(("drawdown_projection", 1),),
        hard_risk=True,
    )
    metric = _metric(_target((0.2, 0.0)), evaluation)
    assert metric.execution_rejection_count == 1
    assert metric.risk_projection_count == 1
    assert metric.hard_risk_violation


def test_v6_replay_rejects_length_reward_and_identity_drift() -> None:
    path = _target((0.2, 0.0))
    with pytest.raises(ValueError, match="cover"):
        _metric(path, _evaluation(3))
    with pytest.raises(ValueError, match="reward"):
        _metric(path, _evaluation(2), reward_scale=1.0)
    with pytest.raises(ValueError, match="forecast"):
        build_causal_alpha_v6_replay_metric(
            run_manifest_digest=_digest("1"),
            v4_context_manifest_digest=_digest("2"),
            symbol="BTCUSDT",
            episode_index=0,
            contract_digest=_digest("3"),
            fit_digest=_digest("4"),
            forecast_digest=_digest("b"),
            target_path=path,
            evaluation=_evaluation(2),
            episode_hours=24.0,
            reward_scale=2.0,
        )
    metric = _metric(path, _evaluation(2))
    with pytest.raises(ValueError, match="wealth"):
        replace(metric, net_wealth=2.0, digest="")
