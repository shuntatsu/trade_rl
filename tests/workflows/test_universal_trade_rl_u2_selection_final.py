from __future__ import annotations

from dataclasses import replace

import pytest

from tests.workflows.test_universal_trade_rl_u2_development_authority import (
    _development_bundle,
)
from trade_rl.artifacts.hashing import content_digest

_PRIMARY_CELLS = ("B", "C1", "C2", "D1", "D2")
_ROBUSTNESS_SCOPES = ("D1", "D2", "D1+D2")


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _leaf(
    *,
    cell: str,
    training_seed: int,
    concrete_symbol: str,
    net_log_growth: float,
    gross_log_growth: float,
):
    module = _module()
    return module.UniversalTradeRLU2SelectionLeafMetrics(
        training_seed=training_seed,
        cell=cell,
        concrete_symbol=concrete_symbol,
        tile_identity=content_digest(
            {
                "fixture": "u2-final-selection-tile",
                "cell": cell,
                "symbol": concrete_symbol,
            }
        ),
        replay_evidence_digest=content_digest(
            {
                "fixture": "u2-final-selection-replay",
                "cell": cell,
                "seed": training_seed,
                "symbol": concrete_symbol,
            }
        ),
        leaf_net_log_growth=net_log_growth,
        leaf_gross_log_growth=gross_log_growth,
        turnover_per_day=0.5,
        meaningful_execution=True,
        hard_risk_violation_count=0,
        unexplained_execution_rejection_count=0,
    )


def _cell_leaves(cell: str):
    logs = {
        0: (0.03, 0.045),
        1: (0.02, 0.03),
        2: (0.0, 0.01),
    }
    return tuple(
        _leaf(
            cell=cell,
            training_seed=seed,
            concrete_symbol=symbol,
            net_log_growth=logs[seed][0],
            gross_log_growth=logs[seed][1],
        )
        for seed in (0, 1, 2)
        for symbol in ("DEV_A", "DEV_B")
    )


def _segments():
    module = _module()
    return (
        module.UniversalTradeRLU2ReducedBootstrapSegment(
            source_window="development_future_1",
            decision_timestamps_ns=(100, 200, 300, 400),
            net_log_excess=(0.02, 0.02, 0.02, 0.02),
        ),
        module.UniversalTradeRLU2ReducedBootstrapSegment(
            source_window="development_future_2",
            decision_timestamps_ns=(500, 600, 700, 800),
            net_log_excess=(0.02, 0.02, 0.02, 0.02),
        ),
    )


def _paired_segments(*, u2_contract, checkpoint_closure):
    module = _module()
    checkpoints = dict(checkpoint_closure.checkpoint_digests)
    symbols = ("DEV_A", "DEV_B")
    pairs = tuple(
        module.UniversalTradeRLU2PairedReplayScopeEvidence(
            training_seed=seed,
            source_window=window,
            cell=cell,
            concrete_symbol=symbol,
            scope_digest=content_digest(
                {
                    "fixture": "u2-final-selection-tile",
                    "cell": cell,
                    "symbol": symbol,
                }
            ),
            evaluation_dataset_digest=content_digest(
                {"fixture": "u2-final-pairing-dataset", "symbol": symbol}
            ),
            u2_contract_digest=u2_contract.digest,
            time_partition_digest=u2_contract.time_partition_digest,
            checkpoint_closure_digest=checkpoint_closure.digest,
            paired_candidate_checkpoint_digest=checkpoints[seed],
            candidate_replay_evidence_digest=content_digest(
                {
                    "fixture": "u2-final-pairing-candidate",
                    "seed": seed,
                    "window": window,
                    "symbol": symbol,
                }
            ),
            cash_replay_evidence_digest=content_digest(
                {
                    "fixture": "u2-final-pairing-cash",
                    "seed": seed,
                    "window": window,
                    "symbol": symbol,
                }
            ),
            decision_timestamps_ns=timestamps,
            candidate_minus_cash_net_log_excess=(0.02, 0.02, 0.02, 0.02),
        )
        for seed in (0, 1, 2)
        for window, cell, timestamps in (
            ("development_future_1", "D1", (100, 200, 300, 400)),
            ("development_future_2", "D2", (500, 600, 700, 800)),
        )
        for symbol in symbols
    )
    return module.reduce_universal_trade_rl_u2_paired_replay_evidence(
        pairs=pairs,
        expected_symbols=symbols,
    )


def _summary_for_seed(*, leaves, training_seed: int):
    module = _module()
    return module.summarize_universal_trade_rl_u2_selection_metrics(
        leaves=tuple(leaf for leaf in leaves if leaf.training_seed == training_seed)
    )


def _passing_selection_children(*, u2_contract, checkpoint_closure):
    module = _module()
    d1_leaves = _cell_leaves("D1")
    d2_leaves = _cell_leaves("D2")
    segments = _paired_segments(
        u2_contract=u2_contract,
        checkpoint_closure=checkpoint_closure,
    )

    d1 = module.evaluate_universal_trade_rl_u2_seed_robustness(
        scope="D1",
        leaves=d1_leaves,
        segments=segments,
    )
    d2 = module.evaluate_universal_trade_rl_u2_seed_robustness(
        scope="D2",
        leaves=d2_leaves,
        segments=segments,
    )
    d12 = module.evaluate_universal_trade_rl_u2_seed_robustness(
        scope="D1+D2",
        leaves=d1_leaves + d2_leaves,
        segments=segments,
    )

    by_d_cell = {
        summary.cell: summary
        for evidence in (d1, d2)
        for summary in evidence.summaries
        if summary.training_seed == 0
    }
    primary = []
    for cell in _PRIMARY_CELLS:
        if cell in by_d_cell:
            summary = by_d_cell[cell]
        else:
            summary = _summary_for_seed(
                leaves=_cell_leaves(cell),
                training_seed=0,
            )
        gate = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)
        assert gate.passed is True
        primary.append(gate)

    assert d1.passed is True
    assert d2.passed is True
    assert d12.passed is True
    return tuple(primary), (d1, d2, d12)


def _final_fixture():
    (
        closure,
        manifest,
        _partition,
        u2_contract,
        _predevelopment,
        _scope_closure,
        checkpoint_closure,
        exposure,
        base_lock,
    ) = _development_bundle()
    development_lock = (
        closure.build_authoritative_universal_trade_rl_u2_development_lock(
            base_lock=base_lock,
            checkpoint_closure=checkpoint_closure,
            training_exposure_evidence=exposure,
            manifest=manifest,
            u2_contract=u2_contract,
        )
    )
    primary, robustness = _passing_selection_children(
        u2_contract=u2_contract,
        checkpoint_closure=checkpoint_closure,
    )
    return (
        u2_contract,
        base_lock,
        development_lock,
        checkpoint_closure,
        primary,
        robustness,
    )


def _build_final(*, primary=None, robustness=None, development_lock=None):
    module = _module()
    (
        u2_contract,
        base_lock,
        canonical_development_lock,
        checkpoint_closure,
        canonical_primary,
        canonical_robustness,
    ) = _final_fixture()
    return module.build_universal_trade_rl_u2_development_selection_evidence(
        u2_contract=u2_contract,
        base_lock=base_lock,
        development_lock=(
            canonical_development_lock if development_lock is None else development_lock
        ),
        checkpoint_closure=checkpoint_closure,
        primary_cell_gates=(canonical_primary if primary is None else primary),
        seed_robustness_gates=(
            canonical_robustness if robustness is None else robustness
        ),
    )


def test_u2_final_selection_requires_all_cells_and_selects_only_seed0_exact_final() -> (
    None
):
    result = _build_final()
    (
        _u2_contract,
        _base_lock,
        _development_lock,
        checkpoint_closure,
        _primary,
        _robustness,
    ) = _final_fixture()

    assert result.passed is True
    assert result.admission_eligible is True
    assert result.production_eligible is False
    assert (
        result.selected_checkpoint_digest
        == dict(checkpoint_closure.checkpoint_digests)[0]
    )
    assert result.primary_cells == _PRIMARY_CELLS
    assert result.robustness_scopes == _ROBUSTNESS_SCOPES


def test_u2_final_selection_child_failure_clears_checkpoint_and_admission() -> None:
    module = _module()
    (
        _u2_contract,
        _base_lock,
        _development_lock,
        _checkpoint_closure,
        primary,
        robustness,
    ) = _final_fixture()
    failing_summary = module.summarize_universal_trade_rl_u2_selection_metrics(
        leaves=(
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_A",
                net_log_growth=0.0,
                gross_log_growth=0.02,
            ),
            _leaf(
                cell="B",
                training_seed=0,
                concrete_symbol="DEV_B",
                net_log_growth=0.0,
                gross_log_growth=0.02,
            ),
        )
    )
    failing_gate = module.evaluate_universal_trade_rl_u2_primary_cell_gate(
        summary=failing_summary
    )
    assert failing_gate.passed is False

    result = _build_final(
        primary=(failing_gate, *primary[1:]),
        robustness=robustness,
    )

    assert result.passed is False
    assert result.selected_checkpoint_digest is None
    assert result.admission_eligible is False
    assert result.production_eligible is False


def test_u2_final_selection_rejects_primary_d_summary_substitution() -> None:
    module = _module()
    (
        _u2_contract,
        _base_lock,
        _development_lock,
        _checkpoint_closure,
        primary,
        robustness,
    ) = _final_fixture()
    substituted = module.summarize_universal_trade_rl_u2_selection_metrics(
        leaves=tuple(
            _leaf(
                cell="D1",
                training_seed=0,
                concrete_symbol=symbol,
                net_log_growth=0.04,
                gross_log_growth=0.06,
            )
            for symbol in ("DEV_A", "DEV_B")
        )
    )
    substituted_gate = module.evaluate_universal_trade_rl_u2_primary_cell_gate(
        summary=substituted
    )
    assert substituted_gate.passed is True
    replaced_primary = tuple(
        substituted_gate if gate.cell == "D1" else gate for gate in primary
    )

    with pytest.raises(ValueError, match="D1|summary|identity|robustness"):
        _build_final(primary=replaced_primary, robustness=robustness)


def test_u2_final_selection_rejects_aggregate_summary_substitution() -> None:
    module = _module()
    (
        u2_contract,
        _base_lock,
        _development_lock,
        checkpoint_closure,
        primary,
        robustness,
    ) = _final_fixture()
    d1, d2, _d12 = robustness
    altered_d2_leaves = tuple(
        _leaf(
            cell="D2",
            training_seed=leaf.training_seed,
            concrete_symbol=leaf.concrete_symbol,
            net_log_growth=(
                0.025 if leaf.training_seed == 1 else leaf.leaf_net_log_growth
            ),
            gross_log_growth=(
                0.0375 if leaf.training_seed == 1 else leaf.leaf_gross_log_growth
            ),
        )
        for leaf in _cell_leaves("D2")
    )
    altered_d12 = module.evaluate_universal_trade_rl_u2_seed_robustness(
        scope="D1+D2",
        leaves=_cell_leaves("D1") + altered_d2_leaves,
        segments=_paired_segments(
            u2_contract=u2_contract,
            checkpoint_closure=checkpoint_closure,
        ),
    )
    assert altered_d12.passed is True

    with pytest.raises(ValueError, match="aggregate|summary|identity|D1|D2"):
        _build_final(
            primary=primary,
            robustness=(d1, d2, altered_d12),
        )


def test_u2_final_selection_rejects_development_lock_checkpoint_substitution() -> None:
    (
        _u2_contract,
        _base_lock,
        development_lock,
        _checkpoint_closure,
        primary,
        robustness,
    ) = _final_fixture()
    substituted_lock = replace(
        development_lock,
        final_checkpoint_closure_digest=content_digest(
            {"fixture": "substituted-final-checkpoint-closure"}
        ),
        digest="",
    )

    with pytest.raises(ValueError, match="checkpoint|closure|lock|identity"):
        _build_final(
            primary=primary,
            robustness=robustness,
            development_lock=substituted_lock,
        )


def test_u2_final_selection_rejects_robustness_without_cash_pairing_provenance() -> (
    None
):
    module = _module()
    (
        _u2_contract,
        _base_lock,
        _development_lock,
        _checkpoint_closure,
        primary,
        _robustness,
    ) = _final_fixture()
    d1_leaves = _cell_leaves("D1")
    d2_leaves = _cell_leaves("D2")
    legacy_segments = _segments()
    legacy_robustness = (
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D1",
            leaves=d1_leaves,
            segments=legacy_segments,
        ),
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D2",
            leaves=d2_leaves,
            segments=legacy_segments,
        ),
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D1+D2",
            leaves=d1_leaves + d2_leaves,
            segments=legacy_segments,
        ),
    )
    assert all(gate.passed for gate in legacy_robustness)
    assert all(
        not gate.bootstrap_result.paired_scope_evidence_digests
        for gate in legacy_robustness
    )

    with pytest.raises(ValueError, match="cash|pair|provenance|bootstrap"):
        _build_final(primary=primary, robustness=legacy_robustness)
