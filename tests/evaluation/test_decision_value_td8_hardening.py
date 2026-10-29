"""Result-blind falsifiers for Issue #1025's corrected *future* study design."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.evaluation.test_decision_value_dev_study import _compact_market
from tests.strategies.test_decision_value import (
    _conservative_model,
    _context,
    _model,
    _snapshot,
)
from trade_rl.evaluation import decision_value_dev_study as study
from trade_rl.strategies.decision_value import (
    ConservativeLinearDecisionValuePolicy,
    LinearDecisionValuePolicy,
    preserve_posture_aware_proposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
)


def _held_snapshot():
    s = _snapshot(signal=0.0)
    return replace(
        s,
        account=replace(
            s.account,
            cash=600.0,
            exact_quantities=("4", "0"),
            quantities=(4.0, 0.0),
            weights=(0.4, 0.0),
        ),
    )


def test_all_value_heads_keep_hold_as_the_exact_zero_reference() -> None:
    snap = _held_snapshot()
    lm = _model()
    cm = _conservative_model(cov_diag_val=0.0, uncertainty_penalty_weight=0.0)
    linear = LinearDecisionValuePolicy(lm, runtime_context=_context(lm))
    c = ConservativeLinearDecisionValuePolicy(cm, runtime_context=_context(lm))
    hold = (ACTION_HOLD, ACTION_HOLD)
    assert linear.value_for_action(snap, hold) == 0.0
    assert c.value_for_action(snap, hold) == 0.0
    assert c.predictive_std_for_action(snap, hold) == 0.0


def test_nonpositive_scores_do_not_liquidate_the_held_position_without_flat() -> None:
    snap = _held_snapshot()
    lm = replace(
        _model(),
        coefficients=((-1.0, 0.0, 0.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )
    p = LinearDecisionValuePolicy(lm, runtime_context=_context(lm))
    assert p.decide_action(snap) == (ACTION_HOLD, ACTION_HOLD)
    actual = p.decide_portfolio(snap)
    assert actual.target_weights == pytest.approx(snap.account.weights)
    assert 0 in actual.hold_current_quantity_symbols


def test_flat_is_an_explicit_learned_candidate_not_structurally_zero() -> None:
    lm = _model()
    flat = (ACTION_FLAT, ACTION_FLAT)
    lm = replace(
        lm,
        candidate_actions=(lm.candidate_actions[0], flat, *lm.candidate_actions[1:]),
        coefficients=((-0.03, 0.0, 0.0, 0.0, 0.0, 0.0), *lm.coefficients),
    )
    p = LinearDecisionValuePolicy(lm, runtime_context=_context(lm))
    assert p.value_for_action(_held_snapshot(), flat) < 0.0
    assert p.value_for_action(_snapshot(), flat) <= 0.0


def test_baseline_hold_cancels_pending_order_instead_of_renaming_hold_to_entry() -> (
    None
):
    snap = _snapshot()
    from tests.strategies.test_decision_value import _pending_order

    snap = replace(snap, active_orders=(_pending_order(),))
    proposal = preserve_posture_aware_proposal(
        (ACTION_HOLD, ACTION_HOLD), snap, (0.4, 0.4)
    )
    assert proposal.target_weights == (0.0, 0.0)
    assert proposal.hold_current_quantity_symbols == (0, 1)


def test_new_study_version_requires_authenticated_artifact_before_fitting() -> None:
    old = study.default_2024q1_decision_value_dev_protocol()
    new = study.upgrade_posture_aware_dev_protocol_v2(old)
    assert new.training_reference_mode == "causal_posture_v2"
    assert new.protocol_digest != old.protocol_digest
    market = _compact_market(2200)
    market = replace(market, dataset_id=old.dataset_id)
    with patch.object(
        study, "fit_ridge_forecast", side_effect=AssertionError("FITTED")
    ):
        with pytest.raises(ValueError, match="verified|artifact"):
            study.run_decision_value_dev_study(market, new)


def test_posture_aware_pairing_covers_filled_pending_and_flat_predecision_states() -> (
    None
):
    dataset = _compact_market(90)
    proto = study.upgrade_posture_aware_dev_protocol_v2(
        study.default_2024q1_decision_value_dev_protocol()
    )
    code = (ACTION_LONG, ACTION_FLAT)
    observations = {}
    for t in (24, 30, 36, 42):
        pair = study._training_pair(dataset, proto, t, code)
        assert pair.eligible_for_label
        observations[t] = pair.baseline_snapshot
        assert pair.baseline_snapshot == pair.candidate_snapshot
    assert observations[24].account.exact_quantities == ("0", "0")
    assert any(Fraction(q) != 0 for q in observations[30].account.exact_quantities)
    assert any(Fraction(q) != 0 for q in observations[36].account.exact_quantities)
    assert observations[42].active_orders


def test_ledger_verification_checks_interval_flows_and_does_not_trust_totals() -> None:
    book = SimpleNamespace(
        portfolio_value=1000.0,
        cash=1000.0,
        total_cost=0.0,
        funding_pnl=0.0,
        borrow_cost=0.0,
        turnover_total=0.0,
        max_drawdown=0.0,
        exact_quantities=(0, 0),
    )
    interval = SimpleNamespace(
        start_index=1,
        next_index=2,
        interval_cost=999999.0,
        interval_funding=0.0,
        interval_borrow_cost=0.0,
        interval_dividend=0.0,
        interval_cash_interest=0.0,
        cash_before=1000.0,
        cash_after=1000.0,
        portfolio_value_before=1000.0,
        portfolio_value_after=1000.0,
        exact_quantities_before=("0", "0"),
        exact_quantities_after=("0", "0"),
        order_events=(),
        total_cost_before=0.0,
        total_cost_after=0.0,
        funding_pnl_before=0.0,
        funding_pnl_after=0.0,
        borrow_cost_before=0.0,
        borrow_cost_after=0.0,
        termination_reason=None,
    )
    ledger = SimpleNamespace(
        intervals=[interval],
        final_portfolio_value=1000.0,
        final_cash=1000.0,
        final_total_cost=0.0,
        final_funding_pnl=0.0,
        final_borrow_cost=0.0,
        final_turnover_total=0.0,
        final_max_drawdown=0.0,
        terminal_exact_quantities=("0", "0"),
        termination_reason=None,
    )
    result = SimpleNamespace(
        book=book, ledger_evidence=SimpleNamespace(intervals=ledger)
    )
    assert (
        study._verify_ledger_reconciliation(
            result,
            expected_intervals=1,
            dataset=_compact_market(5),
            initial_capital=1000.0,
        )
        is False
    )


def test_v2_fits_all_three_models_on_shared_posture_and_flat_labels() -> None:
    from trade_rl.evaluation.decision_value_dev_study import (
        _training_execution_cost,
    )
    from trade_rl.evaluation.decision_value_fit import (
        fit_conservative_linear_decision_value,
        fit_synthetic_linear_decision_value,
        fit_synthetic_shallow_decision_value,
    )
    from trade_rl.evaluation.paired_portfolio_label import (
        build_synthetic_paired_label,
        build_synthetic_paired_label_batch,
    )

    dataset = _compact_market(120)
    proto = study.upgrade_posture_aware_dev_protocol_v2(
        study.default_2024q1_decision_value_dev_protocol()
    )
    ref_digest = "d" * 64
    rows = []
    for t in (24, 30, 36, 42, 48, 54):
        for candidate in proto.study_plan.candidate_actions[1:]:
            pair = study._training_pair(dataset, proto, t, candidate)
            assert pair.eligible_for_label
            rows.append(
                build_synthetic_paired_label(
                    dataset,
                    pair,
                    baseline_action=proto.study_plan.candidate_actions[0],
                    candidate_action=candidate,
                    target_abs_weights=proto.target_abs_weights,
                    reference_policy_digest=ref_digest,
                    expected_execution_policy_digest=(
                        _training_execution_cost(proto).execution_policy_digest
                    ),
                    observation_record_available_at=dataset.timestamps[t],
                    settlement_record_available_at=dataset.timestamps[t + 24],
                    fit_cutoff=dataset.timestamps[80],
                    source_scope="real_development",
                )
            )
    batch = build_synthetic_paired_label_batch(
        tuple(rows),
        fit_cutoff=dataset.timestamps[80],
        evaluation_start=dataset.timestamps[81],
        expected_reference_policy_digest=ref_digest,
    )
    flat = (ACTION_FLAT, ACTION_FLAT)
    assert sum(r.candidate_action == flat for r in batch.samples) == 6
    assert any(r.candidate_action == flat and r.label_delta != 0 for r in batch.samples)
    models = [
        fit_synthetic_linear_decision_value(
            batch,
            candidate_actions=proto.study_plan.candidate_actions,
            posture_aware=True,
        ),
        fit_conservative_linear_decision_value(
            batch,
            candidate_actions=proto.study_plan.candidate_actions,
            posture_aware=True,
        ),
        fit_synthetic_shallow_decision_value(
            batch,
            candidate_actions=proto.study_plan.candidate_actions,
            posture_aware=True,
        ),
    ]
    assert [m.schema_version for m in models] == [
        "synthetic_linear_decision_value_v2",
        "conservative_linear_decision_value_v3",
        "synthetic_shallow_decision_value_v2",
    ]
    assert all(m.training_rows == len(rows) for m in models)
    assert all(any(c != 0 for c in m.coefficients[0]) for m in models)
    assert len(models[0].coefficients[0]) == len(models[1].coefficients[0])
    from trade_rl.strategies.decision_value import (
        ConservativeLinearDecisionValuePolicy,
        LinearDecisionValuePolicy,
        LinearDecisionValueRuntimeContext,
    )
    from trade_rl.strategies.decision_value_shallow import (
        ShallowDecisionValuePolicy,
    )

    policies = [
        LinearDecisionValuePolicy,
        ConservativeLinearDecisionValuePolicy,
        ShallowDecisionValuePolicy,
    ]
    snapshot = next(
        s.observation
        for s in batch.samples
        if s.candidate_action == flat and s.observation.active_orders
    )
    snapshot = replace(
        snapshot,
        index=81,
        timestamp_ns=int(dataset.timestamps[81].astype("int64")),
        agent_stop_index=100,
        remaining_decisions=19,
    )
    for model, policy_class in zip(models, policies, strict=True):
        context = LinearDecisionValueRuntimeContext(
            source_scope=model.source_scope,
            feature_config_digest=model.feature_config_digest,
            normalization_digest=model.normalization_digest,
            execution_policy_digest=model.execution_policy_digest,
            reference_policy_digest=model.reference_policy_digest,
        )
        policy = policy_class(model, runtime_context=context)
        assert policy.value_for_action(snapshot, (ACTION_HOLD, ACTION_HOLD)) == 0.0
        assert np.isfinite(policy.value_for_action(snapshot, flat))


def test_posture_v2_end_to_end_uses_only_verified_artifact_and_new_evidence(
    tmp_path: Path,
) -> None:
    """Causal synthetic full study, not a historical / G4 real-data execution."""
    from trade_rl.data.artifacts.publication import (
        publish_market_dataset_artifact,
    )

    dataset = _compact_market(120).with_content_identity(
        {"source": "closed synthetic posture fixture"}
    )
    artifact = publish_market_dataset_artifact(tmp_path / "dataset", dataset)
    old = study.default_2024q1_decision_value_dev_protocol()
    block = study.DecisionValueBlockSpec(
        block_id="synthetic_posture_test",
        train_start_index=24,
        train_stop_index=54,
        train_step=6,
        fit_cutoff_index=80,
        eval_start_index=81,
        eval_stop_index=114,
    )
    digest = study.compute_real_scope_authorization_digest(
        dataset_id=dataset.dataset_id,
        expected_artifact_digest=artifact.artifact_digest,
        symbols=dataset.symbols,
        initial_capital=old.study_plan.initial_capital,
        target_abs_weights=old.target_abs_weights,
        horizon_bars=old.study_plan.horizon_bars,
        terminal_reserve_bars=old.study_plan.terminal_reserve_bars,
        candidate_actions=old.study_plan.candidate_actions,
        fee_rate=old.study_plan.fee_rate,
        spread_rate=old.study_plan.spread_rate,
        impact_rate=old.base_execution_cost.impact_rate,
        max_participation_rate=old.base_execution_cost.max_participation_rate,
        risk_config=old.risk_config,
        blocks=(block,),
        ridge_feature_indices=old.ridge_feature_indices,
        ridge_horizon_hours=old.ridge_horizon_hours,
        ridge_alpha=old.ridge_alpha,
        ridge_entry_threshold=old.ridge_entry_threshold,
        ridge_exit_threshold=old.ridge_exit_threshold,
        prior_usage_disclosure="synthetic-only; no G4",
        learner_roster=old.study_plan.learner_roster,
        value_ridge_penalty=old.value_ridge_penalty,
        uncertainty_penalty_weight=old.uncertainty_penalty_weight,
        scenarios=old.scenarios,
    )
    old_synth = replace(
        old,
        dataset_id=dataset.dataset_id,
        expected_artifact_digest=artifact.artifact_digest,
        blocks=(block,),
        prior_usage_disclosure="synthetic-only; no G4",
        study_plan=replace(
            old.study_plan,
            real_scope_authorization_digest=digest,
        ),
    )
    proto = study.upgrade_posture_aware_dev_protocol_v2(old_synth)
    result = study.run_decision_value_dev_study(
        dataset,
        proto,
        artifact_root=artifact.root,
        output_dir=tmp_path / "output",
    )
    assert result.schema_version == "decision_value_dev_study_result_v2"
    assert result.training_reference_mode == "causal_posture_v2"
    block_result = result.blocks[0]
    assert block_result.training_rows == 6 * 9
    assert block_result.training_posture_counts is not None
    assert all(block_result.training_posture_counts.values())
    assert block_result.causality_probe_passed
    assert block_result.ledger_evidence_reconciled
    import json

    summary = json.loads((tmp_path / "output" / "result.json").read_text())
    assert summary["validity_status"] == "software_only_no_G4_authorization"
    assert summary["blocks"][0]["training_posture_counts"]["pending"] > 0
    assert not (tmp_path / "output" / "unstated_real_returns.json").exists()
