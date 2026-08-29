from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v8 import (
    CausalAlphaV8Candidate,
    CausalAlphaV8TargetConfig,
    CausalAlphaV8TargetPath,
)
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    evaluate_path_performance,
)
from trade_rl.learning.rollout_evaluation import (
    ActionPathEvaluation,
    ActionPathStepEconomics,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionBoundaries,
)
from trade_rl.workflows.universal_causal_alpha_v8_attribution import (
    build_causal_alpha_v8_attribution,
)


def _digest(char: str) -> str:
    return char * 64


def _target() -> CausalAlphaV8TargetPath:
    targets = np.asarray([0.10, 0.0, -0.10])
    reasons = ("hold_position", "exit", "entry")
    v6 = CausalAlphaV6TargetPath(
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        initial_weight=0.10,
        decision_indices=np.arange(10, 13),
        targets=targets,
        fast_proposals=targets,
        expected_returns_4h=np.asarray([0.03, 0.0, -0.03]),
        expected_returns_24h=np.zeros(3),
        expected_returns_72h=np.zeros(3),
        direction_scores_4h=np.asarray([1.0, 0.0, -1.0]),
        uncertainties_4h=np.full(3, 0.001),
        one_way_cost_rates=np.full(3, 0.0001),
        liquidity_weight_caps=np.full(3, 0.25),
        risk_weight_caps=np.full(3, 0.25),
        objectives=np.zeros(3),
        confirmation_counts=np.arange(3),
        actionable_mask=np.ones(3, dtype=np.bool_),
        slow_states=(
            CausalAlphaV6SlowState.MIXED,
            CausalAlphaV6SlowState.FLAT,
            CausalAlphaV6SlowState.MIXED,
        ),
        reasons=reasons,
        reason_counts=tuple(sorted((reason, reasons.count(reason)) for reason in set(reasons))),
        submitted_change_count=2,
        sign_flip_count=0,
        liquidity_deleveraging_count=0,
        risk_projection_count=0,
        forecast_digest=_digest("a"),
        config_digest=CausalAlphaV8TargetConfig().digest,
    )
    return CausalAlphaV8TargetPath(
        candidate=CausalAlphaV8Candidate.ROBUST_CONTRARIAN,
        v6_target_path=v6,
        source_forecast_digest=_digest("b"),
        calibration_fit_digest=_digest("c"),
        v8_config_digest=CausalAlphaV8TargetConfig().digest,
    )


def _evaluation() -> ActionPathEvaluation:
    gross = np.asarray([0.01, -0.01, 0.02])
    net = np.asarray([0.009, -0.011, 0.019])
    costs = np.full(3, 0.001)
    turnover = np.asarray([0.0, 0.1, 0.1])
    performance = evaluate_path_performance(
        gross_step_returns=gross,
        net_step_returns=net,
        rewards=100.0 * np.log1p(net),
        turnover=turnover,
        costs=costs,
        closed_trade_count=1,
        winning_trade_count=1,
    )
    collapse = ActionPathCollapseEvidence(
        decision_count=3,
        action_dimension_count=1,
        active_dimension_count=3,
        inactive_dimension_count=0,
        proposal_distance_count=2,
        submitted_change_count=2,
        downstream_no_trade_suppression_count=0,
        execution_rejection_count=0,
        executed_change_count=2,
        trade_count=1,
        constant_submitted_actions=False,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        hard_risk_violation=False,
    )
    return ActionPathEvaluation(
        actions=_target().v6_target_path.targets[:, None].astype(np.float32),
        performance=performance,
        collapse_evidence=collapse,
        step_economics=ActionPathStepEconomics(
            gross_returns=gross,
            net_returns=net,
            costs=costs,
            turnover=turnover,
            realized_weights=_target().v6_target_path.targets[:, None],
        ),
    )


def test_v8_attribution_rebinds_v8_identity_and_accepts_centered_liquidity() -> None:
    target = _target()
    evidence = build_causal_alpha_v8_attribution(
        target_path=target,
        evaluation=_evaluation(),
        confidence=np.asarray([0.2, 0.5, 0.8]),
        realized_volatility=np.asarray([0.01, 0.02, 0.03]),
        liquidity=np.asarray([-0.5, 0.0, 0.5]),
        boundaries=CausalAlphaV7AttributionBoundaries(
            confidence=(0.25, 0.50, 0.75),
            realized_volatility=(0.015, 0.025, 0.035),
            liquidity=(-0.25, 0.25, 0.75),
            calibration_range_digest=_digest("d"),
        ),
        step_hours=1.0,
    )

    assert evidence.candidate is CausalAlphaV8Candidate.ROBUST_CONTRARIAN
    assert evidence.target_path_digest == target.digest
    assert evidence.decision_count == 3
    assert set(evidence.dimensions) == {
        "confidence_quartile",
        "exposure",
        "liquidity_quartile",
        "slow_state",
        "transition",
        "volatility_quartile",
    }
    assert evidence.gross_log_return == pytest.approx(
        np.log1p(_evaluation().performance.gross_return)
    )
    assert evidence.net_log_return == pytest.approx(
        np.log1p(_evaluation().performance.net_return)
    )
