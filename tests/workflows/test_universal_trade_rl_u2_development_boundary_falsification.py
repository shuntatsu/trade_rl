from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.workflows.test_universal_trade_rl_u2_development_authority import (
    _development_bundle,
)
from tests.workflows.test_universal_trade_rl_u2_development_closure import (
    _checkpoint_members,
    _exposure_rows,
    _predevelopment_bundle,
    _real_manifest,
    _role_symbols,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole


def _sha(label: str) -> str:
    return content_digest({"u2-development-boundary-falsification": label})


def _mutated(artifact, field: str, value):
    clone = replace(artifact)
    object.__setattr__(clone, field, value)
    return clone


def _authority_case():
    from trade_rl.workflows import universal_trade_rl_u2_development_authority as authority

    (
        closure,
        manifest,
        _partition,
        u2_contract,
        predevelopment,
        scope_closure,
        checkpoint_closure,
        exposure,
        base_lock,
    ) = _development_bundle()
    development_lock = closure.build_authoritative_universal_trade_rl_u2_development_lock(
        base_lock=base_lock,
        checkpoint_closure=checkpoint_closure,
        training_exposure_evidence=exposure,
        manifest=manifest,
        u2_contract=u2_contract,
    )
    return authority, {
        "predevelopment_contract": predevelopment,
        "base_lock": base_lock,
        "development_lock": development_lock,
        "manifest": manifest,
        "u2_contract": u2_contract,
        "supplied_scope_closure": scope_closure,
        "source_tree_digest": base_lock.source_tree_digest,
        "lockfile_digest": base_lock.lockfile_digest,
        "evaluation_runtime_identity_digest": base_lock.evaluation_runtime_identity_digest,
    }


def _call_authority(authority, args: dict[str, object]) -> None:
    authority.require_universal_trade_rl_u2_development_lock_for_replay(**args)


def _with_base(args: dict[str, object], **changes: object) -> dict[str, object]:
    base_lock = replace(args["base_lock"], **changes, digest="")
    development_lock = replace(
        args["development_lock"],
        base_lock_digest=base_lock.digest,
        digest="",
    )
    return {
        **args,
        "base_lock": base_lock,
        "development_lock": development_lock,
    }


def test_u2_replay_authority_rejects_wrong_boundary_types() -> None:
    authority, canonical = _authority_case()
    cases = (
        ("predevelopment_contract", "pre-development"),
        ("base_lock", "base|lock"),
        ("development_lock", "authoritative|lock"),
        ("manifest", "manifest|universe"),
        ("u2_contract", "U2|contract"),
        ("supplied_scope_closure", "scope|closure"),
    )
    for field, pattern in cases:
        args = {**canonical, field: object()}
        with pytest.raises(TypeError, match=pattern):
            _call_authority(authority, args)


def test_u2_replay_authority_rejects_predevelopment_and_base_identity_drift() -> None:
    authority, canonical = _authority_case()

    for field, pattern in (
        ("universe_manifest_digest", "pre-development.*universe|universe.*identity"),
        ("u2_contract_digest", "pre-development.*U2|U2.*identity"),
    ):
        args = {
            **canonical,
            "predevelopment_contract": _mutated(
                canonical["predevelopment_contract"], field, _sha(f"predev-{field}")
            ),
        }
        with pytest.raises(ValueError, match=pattern):
            _call_authority(authority, args)

    for field, pattern in (
        ("predevelopment_contract_digest", "base-lock.*pre-development"),
        ("universe_manifest_digest", "base-lock.*universe"),
        ("u2_contract_digest", "base-lock.*U2"),
        ("u1_contract_digest", "base-lock.*U1"),
        ("u1_normalizer_digest", "base-lock.*normalizer"),
    ):
        args = _with_base(canonical, **{field: _sha(f"base-{field}")})
        with pytest.raises(ValueError, match=pattern):
            _call_authority(authority, args)


def test_u2_replay_authority_rejects_authoritative_and_scope_identity_drift() -> None:
    authority, canonical = _authority_case()
    for field, pattern in (
        ("base_lock_digest", "authoritative.*base-lock"),
        ("predevelopment_contract_digest", "authoritative.*pre-development"),
        ("universe_manifest_digest", "authoritative.*universe"),
        ("u2_contract_digest", "authoritative.*U2"),
        ("u1_contract_digest", "authoritative.*U1"),
        ("u1_normalizer_digest", "authoritative.*normalizer"),
        ("replay_authority_schema", "authority.*schema"),
    ):
        value = "drifted-authority-schema" if field == "replay_authority_schema" else _sha(
            f"authority-{field}"
        )
        args = {
            **canonical,
            "development_lock": _mutated(
                canonical["development_lock"], field, value
            ),
        }
        with pytest.raises(ValueError, match=pattern):
            _call_authority(authority, args)

    for field, pattern in (
        ("universe_manifest_digest", "scope.*universe"),
        ("u2_contract_digest", "scope.*U2"),
    ):
        args = {
            **canonical,
            "supplied_scope_closure": _mutated(
                canonical["supplied_scope_closure"], field, _sha(f"scope-{field}")
            ),
        }
        with pytest.raises(ValueError, match=pattern):
            _call_authority(authority, args)


def test_u2_replay_authority_rejects_runtime_lock_and_open_state_drift() -> None:
    authority, canonical = _authority_case()
    for field, pattern in (
        ("source_tree_digest", "source-tree"),
        ("lockfile_digest", "lockfile"),
        ("evaluation_runtime_identity_digest", "runtime"),
    ):
        args = {**canonical, field: _sha(f"supplied-{field}")}
        with pytest.raises(ValueError, match=pattern):
            _call_authority(authority, args)

    for field, pattern in (
        ("development_numeric_open_count", "zero prior Development|Development opens"),
        ("admission_numeric_open_count", "zero Admission|Admission opens"),
    ):
        args = _with_base(canonical, **{field: 1})
        with pytest.raises(ValueError, match=pattern):
            _call_authority(authority, args)

    with pytest.raises(TypeError, match="replay session"):
        authority.replay_universal_trade_rl_u2_development_scope(
            session=object(),
            request=object(),
        )


def test_u2_closure_helpers_reject_noncanonical_digest_and_symbol_inputs() -> None:
    module = __import__(
        "trade_rl.workflows.universal_trade_rl_u2_development_closure",
        fromlist=["dummy"],
    )
    digests = tuple((seed, _sha(f"seed-{seed}")) for seed in (0, 1, 2))

    with pytest.raises(TypeError, match="immutable tuple"):
        module._canonical_digest_pairs(list(digests), field="test mapping")
    with pytest.raises(ValueError, match="seed/digest pairs"):
        module._canonical_digest_pairs(((0,), (1, _sha("1")), (2, _sha("2"))), field="test mapping")
    with pytest.raises(ValueError, match="seed must be an integer"):
        module._canonical_digest_pairs(((True, _sha("0")), (1, _sha("1")), (2, _sha("2"))), field="test mapping")
    with pytest.raises(ValueError, match="canonical seeds"):
        module._canonical_digest_pairs((digests[1], digests[0], digests[2]), field="test mapping")

    for symbols in ((), ("DEV_B", "DEV_A"), ("DEV_A", "DEV_A"), ("",)):
        with pytest.raises(ValueError, match="Train symbols"):
            module._canonical_train_symbols(symbols)


def test_u2_closure_rejects_predevelopment_identity_and_constructor_drift() -> None:
    module = __import__(
        "trade_rl.workflows.universal_trade_rl_u2_development_closure",
        fromlist=["dummy"],
    )
    manifest, u2_contract, predevelopment = _predevelopment_bundle()

    with pytest.raises(TypeError, match="pre-development contract"):
        module._require_predevelopment_u2_identity(
            predevelopment_contract=object(), u2_contract=u2_contract
        )
    with pytest.raises(TypeError, match="UniversalTradeRLU2Contract"):
        module._require_predevelopment_u2_identity(
            predevelopment_contract=predevelopment, u2_contract=object()
        )
    with pytest.raises(ValueError, match="U2 contract identity"):
        module._require_predevelopment_u2_identity(
            predevelopment_contract=_mutated(
                predevelopment, "u2_contract_digest", _sha("predev-u2")
            ),
            u2_contract=u2_contract,
        )
    with pytest.raises(ValueError, match="universe manifest identity"):
        module._require_predevelopment_u2_identity(
            predevelopment_contract=_mutated(
                predevelopment, "universe_manifest_digest", _sha("predev-universe")
            ),
            u2_contract=u2_contract,
        )

    with pytest.raises(TypeError, match="requires a manifest"):
        module.build_authoritative_universal_trade_rl_u2_predevelopment_contract(
            manifest=object(), u2_contract=u2_contract
        )
    with pytest.raises(TypeError, match="requires U2 contract"):
        module.build_authoritative_universal_trade_rl_u2_predevelopment_contract(
            manifest=manifest, u2_contract=object()
        )


def test_u2_final_checkpoint_closure_rejects_malformed_members_and_identity_drift() -> None:
    module = __import__(
        "trade_rl.workflows.universal_trade_rl_u2_development_closure",
        fromlist=["dummy"],
    )
    _manifest, u2_contract, predevelopment = _predevelopment_bundle()
    members = _checkpoint_members(u2_contract=u2_contract)
    valid = module.build_universal_trade_rl_u2_final_checkpoint_closure(
        predevelopment_contract=predevelopment,
        u2_contract=u2_contract,
        members=members,
    )

    with pytest.raises(ValueError, match="schema"):
        replace(valid, schema_version="drift", digest="")
    with pytest.raises(ValueError, match="Production NO-GO"):
        replace(valid, production_status="GO", digest="")
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, digest=_sha("wrong-final-closure-digest"))

    with pytest.raises(ValueError, match="canonical seed order"):
        module.build_universal_trade_rl_u2_final_checkpoint_closure(
            predevelopment_contract=predevelopment,
            u2_contract=u2_contract,
            members=(members[1], members[0], members[2]),
        )

    fake_plan = SimpleNamespace(seed=0)
    with pytest.raises(TypeError, match="member plan"):
        module.build_universal_trade_rl_u2_final_checkpoint_closure(
            predevelopment_contract=predevelopment,
            u2_contract=u2_contract,
            members=((fake_plan, members[0][1], members[0][2]), members[1], members[2]),
        )
    with pytest.raises(TypeError, match="member checkpoint"):
        module.build_universal_trade_rl_u2_final_checkpoint_closure(
            predevelopment_contract=predevelopment,
            u2_contract=u2_contract,
            members=((members[0][0], object(), members[0][2]), members[1], members[2]),
        )
    with pytest.raises(ValueError, match="digest|sha|SHA"):
        module.build_universal_trade_rl_u2_final_checkpoint_closure(
            predevelopment_contract=predevelopment,
            u2_contract=u2_contract,
            members=((members[0][0], members[0][1], "not-a-sha"), members[1], members[2]),
        )

    plan_fields = (
        ("u2_contract_digest", "U2 identity"),
        ("u1_contract_digest", "U1 identity"),
        ("normalizer_digest", "normalizer"),
        ("time_partition_digest", "time partition"),
        ("training_config_digest", "training config"),
    )
    for field, pattern in plan_fields:
        drifted_plan = _mutated(members[0][0], field, _sha(f"plan-{field}"))
        bad_members = ((drifted_plan, members[0][1], members[0][2]), members[1], members[2])
        with pytest.raises(ValueError, match=pattern):
            module.build_universal_trade_rl_u2_final_checkpoint_closure(
                predevelopment_contract=predevelopment,
                u2_contract=u2_contract,
                members=bad_members,
            )

    drifted_source_plan = _mutated(
        members[1][0], "source_closure_digest", _sha("other-source-closure")
    )
    with pytest.raises(ValueError, match="share one source closure"):
        module.build_universal_trade_rl_u2_final_checkpoint_closure(
            predevelopment_contract=predevelopment,
            u2_contract=u2_contract,
            members=(members[0], (drifted_source_plan, members[1][1], members[1][2]), members[2]),
        )


def test_u2_training_exposure_artifacts_reject_invalid_rows_and_state() -> None:
    module = __import__(
        "trade_rl.workflows.universal_trade_rl_u2_development_closure",
        fromlist=["dummy"],
    )
    manifest, u2_contract, predevelopment = _predevelopment_bundle()
    train_symbols = _role_symbols(manifest, UniversalTradeRLSymbolRole.TRAIN)
    rows = _exposure_rows(train_symbols=train_symbols)
    row = rows[0]

    for changes, pattern in (
        ({"schema_version": "drift"}, "row schema"),
        ({"training_seed": 99}, "seed"),
        ({"worker_index": 99}, "worker"),
        ({"concrete_symbol": ""}, "symbol"),
        ({"completed_episode_count": -1}, "non-negative"),
        ({"partial_final_episode_step_count": row.decision_step_count + 1}, "partial steps"),
    ):
        with pytest.raises(ValueError, match=pattern):
            replace(row, **changes, digest="")
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(row, digest=_sha("wrong-exposure-row-digest"))

    evidence = module.build_universal_trade_rl_u2_training_exposure_evidence(
        predevelopment_contract=predevelopment,
        manifest=manifest,
        u2_contract=u2_contract,
        rows=rows,
    )
    with pytest.raises(ValueError, match="evidence schema"):
        replace(evidence, schema_version="drift", digest="")
    with pytest.raises(TypeError, match="invalid row"):
        replace(evidence, rows=(object(), *evidence.rows[1:]), digest="")
    swapped = (evidence.rows[1], evidence.rows[0], *evidence.rows[2:])
    with pytest.raises(ValueError, match="canonical identity order"):
        replace(evidence, rows=swapped, digest="")

    multiple_partial = list(evidence.rows)
    multiple_partial[1] = replace(
        multiple_partial[1], partial_final_episode_step_count=1, digest=""
    )
    with pytest.raises(ValueError, match="multiple partial final episodes"):
        replace(evidence, rows=tuple(multiple_partial), digest="")
    with pytest.raises(ValueError, match="post-hoc reweighting"):
        replace(evidence, posthoc_reweighting_allowed=True, digest="")
    with pytest.raises(ValueError, match="required before Development open"):
        replace(evidence, required_before_development_open=False, digest="")
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(evidence, digest=_sha("wrong-exposure-evidence-digest"))

    with pytest.raises(TypeError, match="universe manifest"):
        module.build_universal_trade_rl_u2_training_exposure_evidence(
            predevelopment_contract=predevelopment,
            manifest=object(),
            u2_contract=u2_contract,
            rows=rows,
        )
    with pytest.raises(ValueError, match="universe manifest identity"):
        module.build_universal_trade_rl_u2_training_exposure_evidence(
            predevelopment_contract=predevelopment,
            manifest=_real_manifest(salt="coverage-drift"),
            u2_contract=u2_contract,
            rows=rows,
        )


def test_u2_authoritative_development_lock_rejects_invalid_state_and_identity() -> None:
    module = __import__(
        "trade_rl.workflows.universal_trade_rl_u2_development_closure",
        fromlist=["dummy"],
    )
    (
        _closure,
        manifest,
        _partition,
        u2_contract,
        _predevelopment,
        _scope_closure,
        checkpoint_closure,
        exposure,
        base_lock,
    ) = _development_bundle()
    lock = module.build_authoritative_universal_trade_rl_u2_development_lock(
        base_lock=base_lock,
        checkpoint_closure=checkpoint_closure,
        training_exposure_evidence=exposure,
        manifest=manifest,
        u2_contract=u2_contract,
    )

    for changes, pattern in (
        ({"schema_version": "drift"}, "lock schema"),
        ({"replay_authority_schema": "drift"}, "authority schema"),
        ({"admission_status": "OPEN"}, "Admission SEALED"),
        ({"production_status": "GO"}, "Production NO-GO"),
    ):
        with pytest.raises(ValueError, match=pattern):
            replace(lock, **changes, digest="")
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(lock, digest=_sha("wrong-authoritative-lock-digest"))

    type_cases = (
        ("base_lock", "base lock"),
        ("checkpoint_closure", "checkpoint closure"),
        ("training_exposure_evidence", "exposure evidence"),
        ("manifest", "manifest"),
        ("u2_contract", "U2 contract"),
    )
    canonical = {
        "base_lock": base_lock,
        "checkpoint_closure": checkpoint_closure,
        "training_exposure_evidence": exposure,
        "manifest": manifest,
        "u2_contract": u2_contract,
    }
    for field, pattern in type_cases:
        with pytest.raises(TypeError, match=pattern):
            module.build_authoritative_universal_trade_rl_u2_development_lock(
                **{**canonical, field: object()}
            )

    with pytest.raises(ValueError, match="universe manifest identity"):
        module.build_authoritative_universal_trade_rl_u2_development_lock(
            **{**canonical, "manifest": _real_manifest(salt="lock-manifest-drift")}
        )

    for field, pattern in (
        ("universe_manifest_digest", "base-lock universe"),
        ("u2_contract_digest", "base-lock U2"),
        ("u1_contract_digest", "base-lock U1"),
        ("u1_normalizer_digest", "base-lock normalizer"),
        ("predevelopment_contract_digest", "pre-development identity"),
    ):
        drifted_base = replace(base_lock, **{field: _sha(f"lock-base-{field}")}, digest="")
        with pytest.raises(ValueError, match=pattern):
            module.build_authoritative_universal_trade_rl_u2_development_lock(
                **{**canonical, "base_lock": drifted_base}
            )

    for artifact_field, target_field, pattern in (
        ("checkpoint_closure", "u2_contract_digest", "checkpoint U2"),
        ("training_exposure_evidence", "u2_contract_digest", "exposure U2"),
        ("training_exposure_evidence", "universe_manifest_digest", "exposure universe"),
    ):
        artifact = replace(canonical[artifact_field])
        object.__setattr__(artifact, target_field, _sha(f"{artifact_field}-{target_field}"))
        with pytest.raises(ValueError, match=pattern):
            module.build_authoritative_universal_trade_rl_u2_development_lock(
                **{**canonical, artifact_field: artifact}
            )

    for field, pattern in (
        ("development_numeric_open_count", "zero Development opens"),
        ("admission_numeric_open_count", "zero Admission opens"),
    ):
        drifted_base = replace(base_lock, **{field: 1}, digest="")
        with pytest.raises(ValueError, match=pattern):
            module.build_authoritative_universal_trade_rl_u2_development_lock(
                **{**canonical, "base_lock": drifted_base}
            )
