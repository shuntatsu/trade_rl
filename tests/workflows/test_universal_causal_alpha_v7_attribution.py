from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetConfig,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v7 import (
    CausalAlphaV7Candidate,
    CausalAlphaV7TargetPath,
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
    build_causal_alpha_v7_attribution,
)


def _digest(char: str) -> str:
    return char * 64


def _target() -> CausalAlphaV7TargetPath:
    targets = np.asarray([-0.10, -0.10, 0.0, 0.10, 0.10, 0.0])
    reasons = ("entry", "hold_position", "exit", "entry", "hold_position", "exit")
    v6 = CausalAlphaV6TargetPath(
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        initial_weight=0.0,
        decision_indices=np.arange(10, 16, dtype=np.int64),
        targets=targets,
        fast_proposals=targets,
        expected_returns_4h=np.linspace(-0.03, 0.03, 6),
        expected_returns_24h=np.full(6, 0.01),
        expected_returns_72h=np.full(6, 0.02),
        direction_scores_4h=np.sign(np.linspace(-0.03, 0.03, 6)),
        uncertainties_4h=np.full(6, 0.001),
        one_way_cost_rates=np.full(6, 0.0001),
        liquidity_weight_caps=np.full(6, 0.25),
        risk_weight_caps=np.full(6, 0.25),
        objectives=np.zeros(6),
        confirmation_counts=np.arange(6),
        actionable_mask=np.ones(6, dtype=np.bool_),
        slow_states=(
            CausalAlphaV6SlowState.FLAT,
            CausalAlphaV6SlowState.SUPPORTIVE,
            CausalAlphaV6SlowState.MIXED,
            CausalAlphaV6SlowState.OPPOSED,
            CausalAlphaV6SlowState.SUPPORTIVE,
            CausalAlphaV6SlowState.MIXED,
        ),
        reasons=reasons,
        reason_counts=tuple(
            sorted((reason, reasons.count(reason)) for reason in set(reasons))
        ),
        submitted_change_count=4,
        sign_flip_count=0,
        liquidity_deleveraging_count=0,
        risk_projection_count=0,
        forecast_digest=_digest("a"),
        config_digest=CausalAlphaV6TargetConfig().digest,
    )
    return CausalAlphaV7TargetPath(
        candidate=CausalAlphaV7Candidate.CAUSAL_CALIBRATED,
        v6_target_path=v6,
        source_forecast_digest=_digest("b"),
        calibration_fit_digest=_digest("c"),
    )


def _evaluation() -> ActionPathEvaluation:
    gross = np.asarray([0.01, -0.02, 0.0, 0.03, -0.01, 0.005])
    net = np.asarray([0.009, -0.021, -0.001, 0.029, -0.011, 0.004])
    costs = np.full(6, 0.001)
    turnover = np.asarray([0.1, 0.0, 0.1, 0.1, 0.0, 0.1])
    rewards = 100.0 * np.log1p(net)
    performance = evaluate_path_performance(
        gross_step_returns=gross,
        net_step_returns=net,
        rewards=rewards,
        turnover=turnover,
        costs=costs,
        closed_trade_count=2,
        winning_trade_count=1,
    )
    collapse = ActionPathCollapseEvidence(
        decision_count=6,
        action_dimension_count=1,
        active_dimension_count=6,
        inactive_dimension_count=0,
        proposal_distance_count=4,
        submitted_change_count=4,
        downstream_no_trade_suppression_count=0,
        execution_rejection_count=0,
        executed_change_count=4,
        trade_count=2,
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


def _boundaries() -> CausalAlphaV7AttributionBoundaries:
    return CausalAlphaV7AttributionBoundaries(
        confidence=(0.25, 0.50, 0.75),
        realized_volatility=(0.01, 0.02, 0.03),
        liquidity=(0.50, 1.00, 1.50),
        calibration_range_digest=_digest("d"),
    )


def test_v7_attribution_reconciles_every_fixed_dimension() -> None:
    evidence = build_causal_alpha_v7_attribution(
        target_path=_target(),
        evaluation=_evaluation(),
        confidence=np.asarray([0.1, 0.3, 0.6, 0.8, 0.9, 0.2]),
        realized_volatility=np.asarray([0.005, 0.015, 0.025, 0.035, 0.01, 0.02]),
        liquidity=np.asarray([0.25, 0.75, 1.25, 1.75, 1.0, 1.5]),
        boundaries=_boundaries(),
        step_hours=1.0,
    )

    assert evidence.dimensions == (
        "confidence_quartile",
        "exposure",
        "liquidity_quartile",
        "slow_state",
        "transition",
        "volatility_quartile",
    )
    for dimension in evidence.dimensions:
        cells = tuple(cell for cell in evidence.cells if cell.dimension == dimension)
        assert sum(cell.support for cell in cells) == evidence.decision_count
        assert sum(cell.gross_log_return for cell in cells) == pytest.approx(
            evidence.gross_log_return
        )
        assert sum(cell.net_log_return for cell in cells) == pytest.approx(
            evidence.net_log_return
        )
        assert sum(cell.execution_cost for cell in cells) == pytest.approx(
            evidence.total_execution_cost
        )
    assert {cell.key for cell in evidence.cells if cell.dimension == "exposure"} == {
        "flat",
        "long",
        "short",
    }


def test_v7_attribution_accepts_centered_relative_volume() -> None:
    evidence = build_causal_alpha_v7_attribution(
        target_path=_target(),
        evaluation=_evaluation(),
        confidence=np.asarray([0.1, 0.3, 0.6, 0.8, 0.9, 0.2]),
        realized_volatility=np.asarray([0.005, 0.015, 0.025, 0.035, 0.01, 0.02]),
        liquidity=np.asarray([-0.75, -0.25, 0.25, 0.75, 0.0, 0.5]),
        boundaries=CausalAlphaV7AttributionBoundaries(
            confidence=(0.25, 0.50, 0.75),
            realized_volatility=(0.01, 0.02, 0.03),
            liquidity=(-0.50, 0.0, 0.50),
            calibration_range_digest=_digest("d"),
        ),
        step_hours=1.0,
    )

    liquidity_cells = tuple(
        cell for cell in evidence.cells if cell.dimension == "liquidity_quartile"
    )
    assert tuple(cell.key for cell in liquidity_cells) == ("q1", "q2", "q3", "q4")
    assert sum(cell.support for cell in liquidity_cells) == evidence.decision_count


def test_v7_attribution_classifies_realized_exposure_not_requested_target() -> None:
    evaluation = _evaluation()
    economics = evaluation.step_economics
    assert economics is not None
    realized = economics.realized_weights.copy()
    realized[0, 0] = 0.0
    observed = replace(
        evaluation,
        step_economics=replace(economics, realized_weights=realized),
    )
    evidence = build_causal_alpha_v7_attribution(
        target_path=_target(),
        evaluation=observed,
        confidence=np.asarray([0.1, 0.3, 0.6, 0.8, 0.9, 0.2]),
        realized_volatility=np.asarray([0.005, 0.015, 0.025, 0.035, 0.01, 0.02]),
        liquidity=np.asarray([0.25, 0.75, 1.25, 1.75, 1.0, 1.5]),
        boundaries=_boundaries(),
        step_hours=1.0,
    )
    exposure = {
        cell.key: cell for cell in evidence.cells if cell.dimension == "exposure"
    }
    assert exposure["flat"].support == 3
    assert exposure["short"].support == 1


def test_v7_attribution_rejects_noncausal_bins_or_missing_step_economics() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(_boundaries(), confidence=(0.5, 0.5, 0.75), digest="")

    evaluation = _evaluation()
    without_steps = replace(evaluation, step_economics=None)
    with pytest.raises(ValueError, match="step economics"):
        build_causal_alpha_v7_attribution(
            target_path=_target(),
            evaluation=without_steps,
            confidence=np.ones(6),
            realized_volatility=np.ones(6),
            liquidity=np.ones(6),
            boundaries=_boundaries(),
            step_hours=1.0,
        )
