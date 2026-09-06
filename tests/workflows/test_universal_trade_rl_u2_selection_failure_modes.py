from __future__ import annotations

from dataclasses import replace

import pytest

from tests.workflows.test_universal_trade_rl_u2_selection_final import (
    _final_fixture,
    _leaf,
    _module,
)
from trade_rl.artifacts.hashing import content_digest


@pytest.fixture(scope="module")
def final_bundle():
    return _final_fixture()


@pytest.fixture(scope="module")
def canonical_objects(final_bundle):
    module = _module()
    (
        u2_contract,
        base_lock,
        development_lock,
        checkpoint_closure,
        primary,
        robustness,
    ) = final_bundle
    final = module.build_universal_trade_rl_u2_development_selection_evidence(
        u2_contract=u2_contract,
        base_lock=base_lock,
        development_lock=development_lock,
        checkpoint_closure=checkpoint_closure,
        primary_cell_gates=primary,
        seed_robustness_gates=robustness,
    )
    leaf = _leaf(
        cell="B",
        training_seed=0,
        concrete_symbol="DEV_A",
        net_log_growth=0.03,
        gross_log_growth=0.04,
    )
    return {
        "leaf": leaf,
        "symbol": primary[0].summary.symbol_metrics[0],
        "pair": robustness[0].paired_scope_evidence[0],
        "bootstrap": robustness[0].bootstrap_result,
        "robustness": robustness[0],
        "final": final,
    }


def _replace_invalid(artifact, **changes):
    changes.setdefault("digest", "")
    return replace(artifact, **changes)


@pytest.mark.parametrize(
    ("case", "changes"),
    (
        ("schema", {"schema_version": "invalid"}),
        ("bool-seed", {"training_seed": True}),
        ("unknown-seed", {"training_seed": 99}),
        ("empty-cell", {"cell": ""}),
        ("empty-symbol", {"concrete_symbol": ""}),
        ("tile-digest", {"tile_identity": "not-a-digest"}),
        ("replay-digest", {"replay_evidence_digest": "not-a-digest"}),
        ("net-nan", {"leaf_net_log_growth": float("nan")}),
        ("gross-nan", {"leaf_gross_log_growth": float("nan")}),
        ("negative-turnover", {"turnover_per_day": -0.1}),
        ("execution-type", {"meaningful_execution": 1}),
        ("negative-risk-count", {"hard_risk_violation_count": -1}),
        (
            "negative-rejection-count",
            {"unexplained_execution_rejection_count": -1},
        ),
    ),
)
def test_selection_leaf_contract_fails_closed(canonical_objects, case, changes) -> None:
    del case
    with pytest.raises((TypeError, ValueError)):
        _replace_invalid(canonical_objects["leaf"], **changes)


def test_selection_leaf_rejects_stale_digest(canonical_objects) -> None:
    with pytest.raises(ValueError, match="digest"):
        replace(
            canonical_objects["leaf"],
            digest=content_digest({"stale": "leaf"}),
        )


@pytest.mark.parametrize(
    ("case", "changes"),
    (
        ("schema", {"schema_version": "invalid"}),
        ("empty-symbol", {"concrete_symbol": ""}),
        ("empty-leaves", {"leaf_digests": ()}),
        ("bad-leaf-digest", {"leaf_digests": ("not-a-digest",)}),
        (
            "duplicate-leaf-digest",
            lambda symbol: {"leaf_digests": (symbol.leaf_digests[0],) * 2},
        ),
        ("net-nan", {"symbol_net_log_growth": float("nan")}),
        ("gross-nan", {"symbol_gross_log_growth": float("nan")}),
        ("execution-type", {"meaningful_execution": 1}),
        ("negative-risk-count", {"hard_risk_violation_count": -1}),
        (
            "negative-rejection-count",
            {"unexplained_execution_rejection_count": -1},
        ),
    ),
)
def test_selection_symbol_contract_fails_closed(
    canonical_objects, case, changes
) -> None:
    del case
    symbol = canonical_objects["symbol"]
    resolved = changes(symbol) if callable(changes) else changes
    with pytest.raises((TypeError, ValueError)):
        _replace_invalid(symbol, **resolved)


def test_selection_symbol_rejects_stale_digest(canonical_objects) -> None:
    with pytest.raises(ValueError, match="digest"):
        replace(
            canonical_objects["symbol"],
            digest=content_digest({"stale": "symbol"}),
        )


@pytest.mark.parametrize(
    ("case", "changes"),
    (
        ("schema", {"schema_version": "invalid"}),
        ("bool-seed", {"training_seed": True}),
        ("unknown-seed", {"training_seed": 99}),
        ("empty-window", {"source_window": ""}),
        ("empty-cell", {"cell": ""}),
        ("empty-symbol", {"concrete_symbol": ""}),
        ("scope-digest", {"scope_digest": "not-a-digest"}),
        ("dataset-digest", {"evaluation_dataset_digest": "not-a-digest"}),
        ("u2-digest", {"u2_contract_digest": "not-a-digest"}),
        ("time-digest", {"time_partition_digest": "not-a-digest"}),
        ("closure-digest", {"checkpoint_closure_digest": "not-a-digest"}),
        (
            "checkpoint-digest",
            {"paired_candidate_checkpoint_digest": "not-a-digest"},
        ),
        ("candidate-digest", {"candidate_replay_evidence_digest": "not-a-digest"}),
        ("cash-digest", {"cash_replay_evidence_digest": "not-a-digest"}),
        (
            "empty-series",
            {"decision_timestamps_ns": (), "candidate_minus_cash_net_log_excess": ()},
        ),
        (
            "misaligned-series",
            lambda pair: {
                "candidate_minus_cash_net_log_excess": pair.candidate_minus_cash_net_log_excess[
                    :-1
                ]
            },
        ),
        (
            "negative-timestamp",
            lambda pair: {
                "decision_timestamps_ns": (-1,) + pair.decision_timestamps_ns[1:]
            },
        ),
        (
            "duplicate-timestamp",
            lambda pair: {
                "decision_timestamps_ns": (pair.decision_timestamps_ns[0],) * 2
                + pair.decision_timestamps_ns[2:]
            },
        ),
        (
            "off-grid",
            lambda pair: {
                "decision_timestamps_ns": (
                    pair.decision_timestamps_ns[0],
                    pair.decision_timestamps_ns[0] + 1,
                )
                + pair.decision_timestamps_ns[2:]
            },
        ),
        (
            "nan-excess",
            lambda pair: {
                "candidate_minus_cash_net_log_excess": (float("nan"),)
                + pair.candidate_minus_cash_net_log_excess[1:]
            },
        ),
    ),
)
def test_paired_replay_artifact_fails_closed(canonical_objects, case, changes) -> None:
    del case
    pair = canonical_objects["pair"]
    resolved = changes(pair) if callable(changes) else changes
    with pytest.raises((TypeError, ValueError)):
        _replace_invalid(pair, **resolved)


def test_paired_replay_artifact_rejects_stale_digest(canonical_objects) -> None:
    with pytest.raises(ValueError, match="digest"):
        replace(
            canonical_objects["pair"],
            digest=content_digest({"stale": "pair"}),
        )


@pytest.mark.parametrize(
    ("case", "changes"),
    (
        ("schema", {"schema_version": "invalid"}),
        ("empty-windows", {"source_windows": ()}),
        (
            "duplicate-windows",
            lambda b: {
                "source_windows": (b.source_windows[0],) * len(b.source_windows)
            },
        ),
        (
            "invalid-window",
            lambda b: {"source_windows": ("invalid",) + b.source_windows[1:]},
        ),
        ("segment-digest", {"segment_digests": ("not-a-digest",)}),
        ("zero-block", {"block_lengths": (0,)}),
        (
            "paired-digest",
            {"paired_scope_evidence_digests": ("not-a-digest",)},
        ),
        (
            "duplicate-paired-digest",
            lambda b: {
                "paired_scope_evidence_digests": (b.paired_scope_evidence_digests[0],)
                * 2
            },
        ),
        ("mean-nan", {"observed_mean": float("nan")}),
        ("reversed-interval", lambda b: {"lower_ci": b.upper_ci + 1.0}),
        ("zero-resamples", {"resamples": 0}),
        ("negative-seed", {"bootstrap_seed": -1}),
        ("zero-confidence", {"confidence_level": 0.0}),
        ("quantile", {"quantile_method": "nearest"}),
        ("cross-boundary", {"blocks_may_cross_segment_boundary": True}),
        ("pass-state", lambda b: {"passed": not b.passed}),
    ),
)
def test_seed_robustness_bootstrap_contract_fails_closed(
    canonical_objects, case, changes
) -> None:
    del case
    bootstrap = canonical_objects["bootstrap"]
    resolved = changes(bootstrap) if callable(changes) else changes
    with pytest.raises((TypeError, ValueError)):
        _replace_invalid(bootstrap, **resolved)


def test_seed_robustness_bootstrap_rejects_stale_digest(canonical_objects) -> None:
    with pytest.raises(ValueError, match="digest"):
        replace(
            canonical_objects["bootstrap"],
            digest=content_digest({"stale": "bootstrap"}),
        )


@pytest.mark.parametrize(
    ("case", "changes"),
    (
        ("schema", {"schema_version": "invalid"}),
        ("scope", {"scope": "invalid"}),
        ("summary-type", {"summaries": (object(),)}),
        ("empty-summaries", {"summaries": ()}),
        ("empty-closure", {"scope_closure": ()}),
        (
            "duplicate-closure",
            lambda r: {"scope_closure": (r.scope_closure[0],) * 2},
        ),
        (
            "bad-closure-symbol",
            lambda r: {
                "scope_closure": ((r.scope_closure[0][0], "", r.scope_closure[0][2]),)
            },
        ),
        (
            "bad-closure-digest",
            lambda r: {
                "scope_closure": (
                    (r.scope_closure[0][0], r.scope_closure[0][1], "bad"),
                )
            },
        ),
        ("bootstrap-type", {"bootstrap_result": object()}),
        ("paired-type", {"paired_scope_evidence": (object(),)}),
        ("unsupported-reason", {"rejection_reasons": ("unsupported",)}),
        (
            "duplicate-reason",
            lambda r: {
                "rejection_reasons": (
                    (r.rejection_reasons[0], r.rejection_reasons[0])
                    if r.rejection_reasons
                    else (
                        "median_seed_symbol_balanced_net_wealth_not_above_cash",
                        "median_seed_symbol_balanced_net_wealth_not_above_cash",
                    )
                )
            },
        ),
        ("pass-state", lambda r: {"passed": not r.passed}),
    ),
)
def test_seed_robustness_evidence_contract_fails_closed(
    canonical_objects, case, changes
) -> None:
    del case
    robustness = canonical_objects["robustness"]
    resolved = changes(robustness) if callable(changes) else changes
    with pytest.raises((TypeError, ValueError)):
        _replace_invalid(robustness, **resolved)


def test_seed_robustness_evidence_rejects_stale_digest(canonical_objects) -> None:
    with pytest.raises(ValueError, match="digest"):
        replace(
            canonical_objects["robustness"],
            digest=content_digest({"stale": "robustness"}),
        )


@pytest.mark.parametrize(
    ("case", "changes"),
    (
        ("schema", {"schema_version": "invalid"}),
        ("u2-type", {"u2_contract": object()}),
        ("base-type", {"base_lock": object()}),
        ("development-type", {"development_lock": object()}),
        ("checkpoint-type", {"checkpoint_closure": object()}),
        ("primary-type", {"primary_cell_gates": (object(),)}),
        (
            "primary-order",
            lambda f: {
                "primary_cell_gates": (f.primary_cell_gates[1], f.primary_cell_gates[0])
                + f.primary_cell_gates[2:]
            },
        ),
        ("robustness-type", {"seed_robustness_gates": (object(),)}),
        (
            "robustness-order",
            lambda f: {
                "seed_robustness_gates": (
                    f.seed_robustness_gates[1],
                    f.seed_robustness_gates[0],
                )
                + f.seed_robustness_gates[2:]
            },
        ),
    ),
)
def test_final_selection_artifact_fails_closed(
    canonical_objects, case, changes
) -> None:
    del case
    final = canonical_objects["final"]
    resolved = changes(final) if callable(changes) else changes
    with pytest.raises((TypeError, ValueError)):
        _replace_invalid(final, **resolved)


def test_final_selection_artifact_rejects_stale_digest(canonical_objects) -> None:
    with pytest.raises(ValueError, match="digest"):
        replace(
            canonical_objects["final"],
            digest=content_digest({"stale": "final"}),
        )
