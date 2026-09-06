from __future__ import annotations

from dataclasses import replace

import pytest

from tests.workflows.test_universal_trade_rl_u2_selection_final import (
    _final_fixture,
)
from tests.workflows.test_universal_trade_rl_u2_selection_robustness import (
    _evaluate,
    _scope_leaves,
)
from trade_rl.artifacts.hashing import content_digest


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _build_final(bundle):
    module = _module()
    (
        u2_contract,
        base_lock,
        development_lock,
        checkpoint_closure,
        primary,
        robustness,
    ) = bundle
    return module.build_universal_trade_rl_u2_development_selection_evidence(
        u2_contract=u2_contract,
        base_lock=base_lock,
        development_lock=development_lock,
        checkpoint_closure=checkpoint_closure,
        primary_cell_gates=primary,
        seed_robustness_gates=robustness,
    )


def _refresh_digest(artifact):
    object.__setattr__(
        artifact,
        "digest",
        content_digest(artifact.to_payload(include_digest=False)),
    )
    return artifact


def _base_bootstrap():
    return _evaluate(scope="D1", leaves=_scope_leaves("D1")).bootstrap_result


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("schema", "schema"),
        ("empty_segments", "segment|incomplete"),
        ("duplicate_windows", "window|unique"),
        ("invalid_window", "window|invalid"),
        ("invalid_segment_digest", "digest"),
        ("invalid_block_length", "block length"),
        ("duplicate_pair_digest", "paired replay|unique"),
        ("reversed_interval", "interval|reversed"),
        ("zero_resamples", "resamples|positive"),
        ("negative_bootstrap_seed", "bootstrap seed|non-negative"),
        ("invalid_confidence", "confidence"),
        ("quantile_drift", "quantile"),
        ("cross_segment_blocks", "cross|D1|D2"),
        ("non_boolean_passed", "passed|boolean"),
        ("inconsistent_passed", "pass state|inconsistent"),
        ("digest_mismatch", "digest|mismatch"),
    ),
)
def test_u2_seed_robustness_bootstrap_result_rejects_contract_drift(
    case: str,
    match: str,
) -> None:
    base = _base_bootstrap()
    digest_a = content_digest({"fixture": "u2-bootstrap-a"})
    digest_b = content_digest({"fixture": "u2-bootstrap-b"})
    changes: dict[str, object]
    if case == "schema":
        changes = {"schema_version": "drifted-bootstrap-schema", "digest": ""}
    elif case == "empty_segments":
        changes = {
            "source_windows": (),
            "segment_digests": (),
            "block_lengths": (),
            "digest": "",
        }
    elif case == "duplicate_windows":
        changes = {
            "source_windows": ("development_future_1", "development_future_1"),
            "segment_digests": (digest_a, digest_b),
            "block_lengths": (1, 1),
            "digest": "",
        }
    elif case == "invalid_window":
        changes = {
            "source_windows": ("admission_future",),
            "segment_digests": (digest_a,),
            "block_lengths": (1,),
            "digest": "",
        }
    elif case == "invalid_segment_digest":
        changes = {"segment_digests": ("not-a-sha256",), "digest": ""}
    elif case == "invalid_block_length":
        changes = {"block_lengths": (0,), "digest": ""}
    elif case == "duplicate_pair_digest":
        changes = {
            "paired_scope_evidence_digests": (digest_a, digest_a),
            "digest": "",
        }
    elif case == "reversed_interval":
        changes = {"lower_ci": base.upper_ci + 1.0, "digest": ""}
    elif case == "zero_resamples":
        changes = {"resamples": 0, "digest": ""}
    elif case == "negative_bootstrap_seed":
        changes = {"bootstrap_seed": -1, "digest": ""}
    elif case == "invalid_confidence":
        changes = {"confidence_level": 1.0, "digest": ""}
    elif case == "quantile_drift":
        changes = {"quantile_method": "nearest", "digest": ""}
    elif case == "cross_segment_blocks":
        changes = {"blocks_may_cross_segment_boundary": True, "digest": ""}
    elif case == "non_boolean_passed":
        changes = {"passed": 1, "digest": ""}
    elif case == "inconsistent_passed":
        changes = {"passed": not base.passed, "digest": ""}
    elif case == "digest_mismatch":
        changes = {"digest": content_digest({"drift": "bootstrap-result"})}
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(case)

    with pytest.raises((TypeError, ValueError), match=match):
        replace(base, **changes)


def _paired_gate():
    return _final_fixture()[-1][0]


def _bootstrap_with_pair_digests(gate, pairs):
    return replace(
        gate.bootstrap_result,
        paired_scope_evidence_digests=tuple(pair.digest for pair in pairs),
        digest="",
    )


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("schema", "schema"),
        ("scope", "scope"),
        ("summary_type", "summary|invalid"),
        ("summary_closure", "summary|closure|fixed"),
        ("empty_scope_closure", "scope closure|sorted|unique"),
        ("wrong_scope_cell", "scope closure|cells"),
        ("empty_scope_symbol", "symbol|invalid"),
        ("invalid_scope_digest", "tile|digest"),
        ("bootstrap_type", "bootstrap|invalid"),
        ("bootstrap_scope", "bootstrap|scope"),
        ("paired_type", "paired replay|invalid"),
        ("bootstrap_settings", "bootstrap settings|drift"),
        ("unsupported_reason", "reason|unsupported"),
        ("duplicate_reason", "reason|unique"),
        ("non_boolean_passed", "passed|boolean"),
        ("digest_mismatch", "digest|mismatch"),
    ),
)
def test_u2_seed_robustness_evidence_rejects_internal_contract_drift(
    case: str,
    match: str,
) -> None:
    base = _evaluate(scope="D1", leaves=_scope_leaves("D1"))
    changes: dict[str, object]
    if case == "schema":
        changes = {"schema_version": "drifted-robustness-schema", "digest": ""}
    elif case == "scope":
        changes = {"scope": "D3", "digest": ""}
    elif case == "summary_type":
        changes = {"summaries": (object(),), "digest": ""}
    elif case == "summary_closure":
        changes = {"summaries": base.summaries[:-1], "digest": ""}
    elif case == "empty_scope_closure":
        changes = {"scope_closure": (), "digest": ""}
    elif case == "wrong_scope_cell":
        changes = {
            "scope_closure": tuple(
                ("D2", symbol, tile) for _cell, symbol, tile in base.scope_closure
            ),
            "digest": "",
        }
    elif case == "empty_scope_symbol":
        first = base.scope_closure[0]
        changes = {
            "scope_closure": tuple(
                sorted(((first[0], "", first[2]), *base.scope_closure[1:]))
            ),
            "digest": "",
        }
    elif case == "invalid_scope_digest":
        first = base.scope_closure[0]
        changes = {
            "scope_closure": tuple(
                sorted(((first[0], first[1], "not-a-sha256"), *base.scope_closure[1:]))
            ),
            "digest": "",
        }
    elif case == "bootstrap_type":
        changes = {"bootstrap_result": object(), "digest": ""}
    elif case == "bootstrap_scope":
        d2 = _evaluate(scope="D2", leaves=_scope_leaves("D2"))
        changes = {"bootstrap_result": d2.bootstrap_result, "digest": ""}
    elif case == "paired_type":
        changes = {"paired_scope_evidence": (object(),), "digest": ""}
    elif case == "bootstrap_settings":
        modified = replace(base.bootstrap_result, resamples=31, digest="")
        changes = {"bootstrap_result": modified, "digest": ""}
    elif case == "unsupported_reason":
        changes = {
            "rejection_reasons": ("not-a-preregistered-reason",),
            "passed": False,
            "digest": "",
        }
    elif case == "duplicate_reason":
        changes = {
            "rejection_reasons": (
                "median_seed_symbol_balanced_net_wealth_not_above_cash",
                "median_seed_symbol_balanced_net_wealth_not_above_cash",
            ),
            "passed": False,
            "digest": "",
        }
    elif case == "non_boolean_passed":
        changes = {"passed": 1, "digest": ""}
    elif case == "digest_mismatch":
        changes = {"digest": content_digest({"drift": "robustness-evidence"})}
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(case)

    with pytest.raises((TypeError, ValueError), match=match):
        replace(base, **changes)


def test_u2_seed_robustness_rejects_bootstrap_pairing_digest_substitution() -> None:
    gate = _paired_gate()
    modified = replace(
        gate.bootstrap_result,
        paired_scope_evidence_digests=tuple(
            reversed(gate.bootstrap_result.paired_scope_evidence_digests)
        ),
        digest="",
    )

    with pytest.raises(ValueError, match="pairing|provenance|bootstrap"):
        replace(gate, bootstrap_result=modified, digest="")


def test_u2_seed_robustness_rejects_paired_scope_closure_substitution() -> None:
    gate = _paired_gate()
    pairs = list(gate.paired_scope_evidence)
    pairs[0] = replace(
        pairs[0],
        scope_digest=content_digest({"drift": "paired-scope"}),
        digest="",
    )
    bootstrap = _bootstrap_with_pair_digests(gate, pairs)

    with pytest.raises(ValueError, match="scope closure|paired replay"):
        replace(
            gate,
            paired_scope_evidence=tuple(pairs),
            bootstrap_result=bootstrap,
            digest="",
        )


def test_u2_seed_robustness_rejects_incomplete_paired_seed_closure() -> None:
    gate = _paired_gate()
    pairs = tuple(pair for pair in gate.paired_scope_evidence if pair.training_seed != 2)
    bootstrap = _bootstrap_with_pair_digests(gate, pairs)

    with pytest.raises(ValueError, match="seed closure|incomplete"):
        replace(
            gate,
            paired_scope_evidence=pairs,
            bootstrap_result=bootstrap,
            digest="",
        )


def test_u2_seed_robustness_rejects_paired_window_drift() -> None:
    gate = _paired_gate()
    pairs = list(gate.paired_scope_evidence)
    pairs[0] = replace(
        pairs[0],
        source_window="development_future_2",
        digest="",
    )
    bootstrap = _bootstrap_with_pair_digests(gate, pairs)

    with pytest.raises(ValueError, match="window closure|incomplete"):
        replace(
            gate,
            paired_scope_evidence=tuple(pairs),
            bootstrap_result=bootstrap,
            digest="",
        )


def test_u2_seed_robustness_requires_pair_objects_when_bootstrap_binds_pairs() -> None:
    gate = _paired_gate()

    with pytest.raises(ValueError, match="paired replay evidence|missing"):
        replace(gate, paired_scope_evidence=(), digest="")


def _final_kwargs(bundle):
    (
        u2_contract,
        base_lock,
        development_lock,
        checkpoint_closure,
        primary,
        robustness,
    ) = bundle
    return {
        "u2_contract": u2_contract,
        "base_lock": base_lock,
        "development_lock": development_lock,
        "checkpoint_closure": checkpoint_closure,
        "primary_cell_gates": primary,
        "seed_robustness_gates": robustness,
    }


@pytest.mark.parametrize(
    ("field", "match"),
    (
        ("u2_contract", "U2 contract"),
        ("base_lock", "base Development lock"),
        ("development_lock", "authoritative Development lock"),
        ("checkpoint_closure", "checkpoint closure"),
    ),
)
def test_u2_final_selection_rejects_invalid_top_level_artifact_type(
    field: str,
    match: str,
) -> None:
    module = _module()
    kwargs = _final_kwargs(_final_fixture())
    kwargs[field] = object()

    with pytest.raises(TypeError, match=match):
        module.UniversalTradeRLU2DevelopmentSelectionEvidence(**kwargs)


def test_u2_final_selection_rejects_schema_drift() -> None:
    module = _module()
    kwargs = _final_kwargs(_final_fixture())

    with pytest.raises(ValueError, match="schema"):
        module.UniversalTradeRLU2DevelopmentSelectionEvidence(
            **kwargs,
            schema_version="drifted-final-selection-schema",
        )


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("base_u2", "base-lock U2"),
        ("development_base", "base-lock"),
        ("development_u2", "Development-lock U2"),
        ("checkpoint_u2", "checkpoint U2"),
        ("checkpoint_mapping", "checkpoint mapping"),
        ("predevelopment", "pre-development"),
        ("u1", "U1 identity"),
        ("normalizer", "normalizer"),
    ),
)
def test_u2_final_selection_rejects_cross_artifact_identity_drift(
    case: str,
    match: str,
) -> None:
    bundle = list(_final_fixture())
    drift = content_digest({"drift": case})
    if case == "base_u2":
        bundle[1] = replace(bundle[1], u2_contract_digest=drift, digest="")
    elif case == "development_base":
        bundle[2] = replace(bundle[2], base_lock_digest=drift, digest="")
    elif case == "development_u2":
        bundle[2] = replace(bundle[2], u2_contract_digest=drift, digest="")
    elif case == "checkpoint_u2":
        bundle[3] = replace(bundle[3], u2_contract_digest=drift, digest="")
    elif case == "checkpoint_mapping":
        rows = list(bundle[1].checkpoint_digests)
        rows[0] = (rows[0][0], drift)
        bundle[1] = replace(bundle[1], checkpoint_digests=tuple(rows), digest="")
    elif case == "predevelopment":
        bundle[2] = replace(bundle[2], predevelopment_contract_digest=drift, digest="")
    elif case == "u1":
        bundle[2] = replace(bundle[2], u1_contract_digest=drift, digest="")
    elif case == "normalizer":
        bundle[2] = replace(bundle[2], u1_normalizer_digest=drift, digest="")
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(case)

    with pytest.raises(ValueError, match=match):
        _build_final(tuple(bundle))


def test_u2_final_selection_rejects_primary_gate_type_and_order_drift() -> None:
    bundle = list(_final_fixture())
    primary = bundle[4]
    bundle[4] = (object(), *primary[1:])
    with pytest.raises(TypeError, match="primary gate|invalid"):
        _build_final(tuple(bundle))

    bundle = list(_final_fixture())
    primary = bundle[4]
    bundle[4] = (primary[1], primary[0], *primary[2:])
    with pytest.raises(ValueError, match="B/C1/C2|primary closure"):
        _build_final(tuple(bundle))


def test_u2_final_selection_rejects_robustness_gate_type_and_order_drift() -> None:
    bundle = list(_final_fixture())
    robustness = bundle[5]
    bundle[5] = (object(), *robustness[1:])
    with pytest.raises(TypeError, match="robustness gate|invalid"):
        _build_final(tuple(bundle))

    bundle = list(_final_fixture())
    robustness = bundle[5]
    bundle[5] = (robustness[1], robustness[0], robustness[2])
    with pytest.raises(ValueError, match="D1/D2/D1\+D2|robustness closure"):
        _build_final(tuple(bundle))


def test_u2_final_selection_rejects_primary_threshold_identity_drift() -> None:
    bundle = list(_final_fixture())
    gate = bundle[4][0]
    object.__setattr__(
        gate,
        "selection_thresholds_digest",
        content_digest({"drift": "primary-threshold"}),
    )
    _refresh_digest(gate)

    with pytest.raises(ValueError, match="primary threshold|drift"):
        _build_final(tuple(bundle))


def test_u2_final_selection_rejects_robustness_threshold_identity_drift() -> None:
    bundle = list(_final_fixture())
    gate = bundle[5][0]
    object.__setattr__(
        gate,
        "robustness_thresholds_digest",
        content_digest({"drift": "robustness-threshold"}),
    )
    _refresh_digest(gate)

    with pytest.raises(ValueError, match="robustness threshold|drift"):
        _build_final(tuple(bundle))


def _mutate_final_pair(field: str, value: str):
    bundle = list(_final_fixture())
    gate = bundle[5][0]
    pairs = list(gate.paired_scope_evidence)
    pairs[0] = replace(pairs[0], **{field: value, "digest": ""})
    bootstrap = _bootstrap_with_pair_digests(gate, pairs)
    object.__setattr__(gate, "paired_scope_evidence", tuple(pairs))
    object.__setattr__(gate, "bootstrap_result", bootstrap)
    _refresh_digest(gate)
    return tuple(bundle)


@pytest.mark.parametrize(
    ("field", "match"),
    (
        ("u2_contract_digest", "paired replay U2"),
        ("time_partition_digest", "paired replay time"),
        ("checkpoint_closure_digest", "checkpoint closure"),
        ("paired_candidate_checkpoint_digest", "checkpoint identity"),
    ),
)
def test_u2_final_selection_rejects_nested_pair_identity_drift(
    field: str,
    match: str,
) -> None:
    bundle = _mutate_final_pair(field, content_digest({"drift": field}))

    with pytest.raises(ValueError, match=match):
        _build_final(bundle)


def test_u2_final_selection_rejects_aggregate_bootstrap_identity_drift() -> None:
    bundle = list(_final_fixture())
    aggregate = bundle[5][2]
    digests = list(aggregate.bootstrap_result.segment_digests)
    digests[0] = content_digest({"drift": "aggregate-bootstrap-segment"})
    bootstrap = replace(
        aggregate.bootstrap_result,
        segment_digests=tuple(digests),
        digest="",
    )
    object.__setattr__(aggregate, "bootstrap_result", bootstrap)
    _refresh_digest(aggregate)

    with pytest.raises(ValueError, match="aggregate bootstrap|identity"):
        _build_final(tuple(bundle))


def test_u2_final_selection_rejects_aggregate_scope_union_drift() -> None:
    bundle = list(_final_fixture())
    aggregate = bundle[5][2]
    object.__setattr__(
        aggregate,
        "scope_closure",
        (*aggregate.scope_closure, ("D2", "EXTRA", content_digest({"extra": "tile"}))),
    )
    _refresh_digest(aggregate)

    with pytest.raises(ValueError, match="aggregate scope closure|mismatch"):
        _build_final(tuple(bundle))


def test_u2_final_selection_rejects_aggregate_pairing_order_drift() -> None:
    bundle = list(_final_fixture())
    aggregate = bundle[5][2]
    pairs = list(aggregate.paired_scope_evidence)
    pairs[0], pairs[1] = pairs[1], pairs[0]
    object.__setattr__(aggregate, "paired_scope_evidence", tuple(pairs))
    _refresh_digest(aggregate)

    with pytest.raises(ValueError, match="cash-pairing provenance|mismatch"):
        _build_final(tuple(bundle))
