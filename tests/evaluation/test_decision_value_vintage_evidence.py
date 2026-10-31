"""TD8 prefix-specific teacher and real paired native outcome provenance."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.evaluation.decision_value_td8_support import (
    COST,
    ONE_WAY,
    RISK_CONFIG,
    TARGETS,
    signal_market,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.decision_value_dev_study import causal_b0_reference_provider
from trade_rl.evaluation.decision_value_feature_transform import (
    apply_feature_transform,
    fit_train_feature_transform,
)
from trade_rl.evaluation.decision_value_reachable_labels import (
    ReferencePolicyFactory,
    collect_reachable_state_labels,
)
from trade_rl.evaluation.paired_portfolio_label import (
    build_synthetic_paired_label_batch,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
)


def _context():
    raw = signal_market(420, seed=17)
    transform = fit_train_feature_transform(
        raw, train_start_index=24, train_stop_index=300, clip=3.0, as_of_index=330
    )
    view = apply_feature_transform(raw, transform)
    provider = causal_b0_reference_provider(
        view,
        raw_dataset=raw,
        causal_feature_clip=3.0,
        feature_indices=(0, 1, 2),
        horizon_hours=24,
        alpha=1.0,
        entry_threshold=0.002,
        exit_threshold=0.0005,
        target_abs_weights=TARGETS,
        one_way_switch_cost=ONE_WAY,
        min_fit_bars=96,
        first_fit_index=24,
    )
    return raw, view, provider


def _labels(view, provider):
    return collect_reachable_state_labels(
        view,
        reference_policy_provider=provider,
        reference_policy_digest="d" * 64,
        branch_indices=(248, 280),
        train_start_index=24,
        prefix_bars=48,
        horizon_bars=6,
        candidate_actions=(
            (ACTION_HOLD, ACTION_HOLD),
            (ACTION_LONG, ACTION_FLAT),
            (ACTION_FLAT, ACTION_LONG),
        ),
        target_abs_weights=TARGETS,
        initial_capital=100_000.0,
        execution_cost=COST,
        risk_config=RISK_CONFIG,
        fit_cutoff=view.timestamps[330],
        source_scope="synthetic_only",
        require_reference_vintage=True,
    )


def test_distinct_prefix_teachers_have_distinct_auditable_vintages():
    raw, _, provider = _context()
    left, right = provider(248, 200), provider(280, 232)
    assert isinstance(left, ReferencePolicyFactory)
    assert isinstance(right, ReferencePolicyFactory)
    left_vintage, right_vintage = left.vintage, right.vintage
    assert (
        left_vintage.source_dataset_id
        == right_vintage.source_dataset_id
        == raw.dataset_id
    )
    assert (left_vintage.prefix_start_index, right_vintage.prefix_start_index) == (
        200,
        232,
    )
    assert left_vintage.fit_cutoff_ns == int(raw.timestamps[200].astype("int64"))
    assert right_vintage.fit_cutoff_ns == int(raw.timestamps[232].astype("int64"))
    assert left_vintage.normalization_digest != right_vintage.normalization_digest
    assert left_vintage.model_digest != right_vintage.model_digest
    assert left_vintage.vintage_digest != right_vintage.vintage_digest
    assert left_vintage.source_runtime_digest == right_vintage.source_runtime_digest


def test_each_eligible_label_binds_teacher_prefix_and_both_native_outcomes():
    _, view, provider = _context()
    collection = _labels(view, provider)
    assert collection.samples
    assert len(collection.eligible_pair_outcomes) == len(collection.samples)
    vintages = {}
    for sample, outcome in zip(
        collection.samples, collection.eligible_pair_outcomes, strict=True
    ):
        vintage = sample.reference_vintage
        assert vintage is not None
        assert vintage.vintage_digest == outcome["reference_vintage"]["vintage_digest"]
        assert sample.pair_evidence_digest == outcome["evidence_digest"]
        assert (
            content_digest({k: v for k, v in outcome.items() if k != "evidence_digest"})
            == outcome["evidence_digest"]
        )
        assert outcome["dataset_id"] == view.dataset_id
        assert outcome["decision_time_ns"] == sample.decision_time_ns
        assert outcome["label_delta"] == sample.label_delta
        assert outcome["baseline"]["ledger_digest"]
        assert outcome["candidate"]["ledger_digest"]
        assert outcome["baseline"]["settled_label_eligible"] is True
        assert outcome["candidate"]["settled_label_eligible"] is True
        assert outcome["prefix_identity_digest"]
        vintages[vintage.prefix_start_index] = vintage.vintage_digest
    assert set(vintages) == {200, 232}
    assert vintages[200] != vintages[232]
    batch = build_synthetic_paired_label_batch(
        collection.samples,
        fit_cutoff=view.timestamps[330],
        evaluation_start=view.timestamps[331],
        expected_reference_policy_digest="d" * 64,
    )
    assert len(batch.samples) == len(collection.samples)
    swapped = replace(
        collection.samples[0],
        reference_vintage=collection.samples[-1].reference_vintage,
    )
    with pytest.raises(ValueError, match="vintage|evidence|identity"):
        build_synthetic_paired_label_batch(
            (swapped, *collection.samples[1:]),
            fit_cutoff=view.timestamps[330],
            evaluation_start=view.timestamps[331],
            expected_reference_policy_digest="d" * 64,
        )


def test_required_td8_vintage_fails_closed_before_any_native_pair():
    _, view, _ = _context()
    with pytest.raises(ValueError, match="vintage|reference"):
        _labels(view, lambda _t, _start: lambda: object())


def test_block_result_serializes_every_eligible_pair_receipt():
    from trade_rl.evaluation.decision_value_dev_study import DecisionValueBlockResult

    _, view, provider = _context()
    collection = _labels(view, provider)
    result = DecisionValueBlockResult(
        block_id="synthetic-vintage-audit",
        paired_replay_runs=collection.paired_replay_runs,
        paired_transitions=1,
        training_rows=len(collection.samples),
        overlap_groups=1,
        eval_replay_runs=0,
        wall_clock_seconds=0.0,
        b0_model_intercept=0.0,
        b1_model_coefficients_sha256="0" * 64,
        b1c_model_coefficients_sha256="0" * 64,
        b2_model_coefficients_sha256="0" * 64,
        causality_probe_passed=True,
        ledger_evidence_reconciled=True,
        arms=(),
        chosen_margin_multipliers={},
        margin_selections={},
        b1c_residual_dof=(),
        distinct_decision_times=2,
        ineligible_pairs=0,
        nonflat_branch_states=0,
        skipped_branches_without_reference=0,
        feature_transform_digest="a" * 64,
        covariate_shift=(),
        eligible_pair_outcomes=collection.eligible_pair_outcomes,
    )
    serialized = result.to_mapping()
    assert len(serialized["eligible_pair_outcomes"]) == len(collection.samples)
    assert (
        serialized["eligible_pair_outcomes"][0]["evidence_digest"]
        == collection.samples[0].pair_evidence_digest
    )
    assert serialized["training_rows"] == len(collection.samples)
