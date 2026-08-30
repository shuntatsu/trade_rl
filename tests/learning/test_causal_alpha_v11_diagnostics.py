from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v11_diagnostics import (
    build_causal_alpha_v11_diagnostics,
)
from trade_rl.learning.rollout_evaluation import (
    ActionPathLifecycleTrace,
    ActionPathStepTrace,
)


def _trace() -> tuple[ActionPathStepTrace, ActionPathLifecycleTrace]:
    rows = 96
    current = np.zeros((rows, 1), dtype=np.float64)
    current[17:81] = 0.1
    realized = current.copy()
    realized[16:80] = 0.1
    requested = realized.copy()
    gross = np.zeros(rows, dtype=np.float64)
    net = np.zeros(rows, dtype=np.float64)
    gross[16:48] = 0.001
    gross[48:80] = -0.0015
    net[16:48] = 0.0009
    net[48:80] = -0.0016
    costs = np.zeros(rows, dtype=np.float64)
    costs[[16, 80]] = 0.0001
    turnover = np.zeros(rows, dtype=np.float64)
    turnover[[16, 80]] = 0.1
    step = ActionPathStepTrace(
        decision_indices=np.arange(rows),
        current_weights=current,
        requested_targets=requested,
        projected_targets=requested,
        realized_weights=realized,
        active_risk_caps=np.full((rows, 1), 0.1),
        active_liquidity_caps=np.full((rows, 1), 0.1),
        fast_means=np.zeros(rows),
        fast_stds=np.zeros(rows),
        fast_qualified_directions=np.zeros(rows),
        fast_edge_margins=np.zeros(rows),
        after_cost_entry_objectives=np.zeros(rows),
        slow_means=np.zeros(rows),
        slow_stds=np.zeros(rows),
        slow_directions=np.zeros(rows),
        position_origins=tuple("unavailable" for _ in range(rows)),
        hierarchy_reasons=tuple("unavailable" for _ in range(rows)),
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        turnovers=turnover,
        submitted=turnover > 0.0,
        suppressed=np.zeros(rows, dtype=np.bool_),
        executed=turnover > 0.0,
    )
    transitions = ["flat"] * rows
    transitions[16] = "entry"
    transitions[17:80] = ["hold"] * 63
    transitions[80] = "exit"
    lifecycle = ActionPathLifecycleTrace(
        submitted_targets=requested,
        execution_intent_targets=requested,
        final_risk_targets=requested,
        applied_risk_scales=np.ones(rows),
        hard_risk_evidence_available=np.ones(rows, dtype=np.bool_),
        hard_risk_violations=np.zeros(rows, dtype=np.bool_),
        risk_reasons=tuple(() for _ in range(rows)),
        transition_classes=tuple(transitions),
        flatten_initiators=tuple("none" for _ in range(rows)),
    )
    return step, lifecycle


def _diagnostics():
    step, lifecycle = _trace()
    qualified = np.zeros(96, dtype=np.int8)
    qualified[[0, 16, 32]] = 1
    labels = np.zeros(96, dtype=np.float64)
    labels[16] = 0.02
    return build_causal_alpha_v11_diagnostics(
        symbol="BTCUSDT",
        episode_id="fold-0",
        step_trace=step,
        lifecycle_trace=lifecycle,
        qualified_directions=qualified,
        actionable_mask=np.ones(96, dtype=np.bool_),
        labels_4h=labels,
        one_way_cost_rates=np.full(96, 0.0007),
        expected_target_digest="a" * 64,
        regenerated_target_digest="a" * 64,
        policy_input_digest="b" * 64,
    )


def test_d1_splits_trade_at_first_neutral_cadence() -> None:
    evidence = _diagnostics()
    trade = evidence.trades[0]

    assert trade.entry_index == 16
    assert trade.first_neutral_index == 48
    assert trade.exit_index == 80
    assert trade.entry_to_neutral_net_log_return == pytest.approx(32 * np.log1p(0.0009))
    assert trade.neutral_to_exit_net_log_return == pytest.approx(32 * np.log1p(-0.0016))
    assert evidence.reconciliation_error < 1e-12


def test_entry_edge_uses_directional_4h_label_minus_round_trip_cost() -> None:
    evidence = _diagnostics()

    assert evidence.entries[0].entry_edge == pytest.approx(0.02 - 2 * 0.0007)
    assert evidence.long_summary.trade_count == 1
    assert evidence.short_summary.trade_count == 0


def test_d1_rejects_regenerated_target_digest_mismatch() -> None:
    step, lifecycle = _trace()

    with pytest.raises(ValueError, match="target digest mismatch"):
        build_causal_alpha_v11_diagnostics(
            symbol="BTCUSDT",
            episode_id="fold-0",
            step_trace=step,
            lifecycle_trace=lifecycle,
            qualified_directions=np.zeros(96),
            actionable_mask=np.ones(96, dtype=np.bool_),
            labels_4h=np.zeros(96),
            one_way_cost_rates=np.full(96, 0.0007),
            expected_target_digest="a" * 64,
            regenerated_target_digest="c" * 64,
            policy_input_digest="b" * 64,
        )
