from __future__ import annotations

from dataclasses import replace

import pytest

from tests.workflows.test_universal_trade_rl_u2_contract import (
    _fixture as _u2_contract_fixture,
)
from tests.workflows.test_universal_trade_rl_u2_development_closure import (
    _checkpoint_members,
    _exposure_rows,
    _predevelopment_bundle,
    _role_symbols,
    _u1_contract,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    build_universal_trade_rl_u2_development_scope_closure,
)
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    build_universal_trade_rl_u2_development_lock,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    build_universal_trade_rl_u2_time_partition,
)

_SOURCE_TREE_DIGEST = content_digest({"fixture": "predev-source-tree"})
_LOCKFILE_DIGEST = content_digest({"fixture": "predev-lockfile"})
_RUNTIME_DIGEST = content_digest({"fixture": "predev-runtime"})


def test_u2_episode_sampling_machine_contract_disables_exact_resume() -> None:
    fixture = _u2_contract_fixture()
    contract = build_universal_trade_rl_u2_contract(
        manifest=fixture.manifest,
        u1_contract=fixture.u1_contract,
        time_partition=fixture.time_partition,
        rl_training_provenance=fixture.rl_training_provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )

    expected = content_digest(
        {
            "schema_version": "universal_trade_rl_u2_episode_sampling_contract_v1",
            "time_partition_digest": fixture.time_partition.digest,
            "fit_end_ns": fixture.time_partition.fit_end_ns,
            "fit_scope_only": True,
            "episode_hours": 720,
            "eligible_start_sampling": "uniform",
            "outcome_must_not_exceed_fit_end": True,
            "exact_mid_episode_resume_supported": False,
            "restart_from_timestep_zero_required": True,
            "regime_oversampling": False,
            "stress_oversampling": False,
        }
    )

    assert contract.episode_sampling_contract_digest == expected


def _development_lock_bundle():
    from trade_rl.workflows import universal_trade_rl_u2_development_closure as closure

    manifest, u2_contract, predevelopment = _predevelopment_bundle()
    partition = build_universal_trade_rl_u2_time_partition(manifest=manifest)
    scope_closure = build_universal_trade_rl_u2_development_scope_closure(
        manifest=manifest,
        time_partition=partition,
        u2_contract=u2_contract,
    )
    checkpoint_closure = closure.build_universal_trade_rl_u2_final_checkpoint_closure(
        predevelopment_contract=predevelopment,
        u2_contract=u2_contract,
        members=_checkpoint_members(u2_contract=u2_contract),
    )
    train_symbols = _role_symbols(manifest, UniversalTradeRLSymbolRole.TRAIN)
    exposure = closure.build_universal_trade_rl_u2_training_exposure_evidence(
        predevelopment_contract=predevelopment,
        manifest=manifest,
        u2_contract=u2_contract,
        rows=_exposure_rows(train_symbols=train_symbols),
    )
    evaluation_dataset_digests = tuple(
        sorted(
            {
                (scope.concrete_symbol, scope.evaluation_dataset_digest)
                for scope in scope_closure.scopes
            }
        )
    )
    base_lock = build_universal_trade_rl_u2_development_lock(
        predevelopment_contract=predevelopment,
        u1_contract_digest=u2_contract.u1_contract_digest,
        u1_normalizer_digest=u2_contract.u1_normalizer_digest,
        checkpoint_digests=checkpoint_closure.checkpoint_digests,
        development_scope_closure_digest=scope_closure.digest,
        evaluation_dataset_digests=evaluation_dataset_digests,
        source_tree_digest=_SOURCE_TREE_DIGEST,
        lockfile_digest=_LOCKFILE_DIGEST,
        evaluation_runtime_identity_digest=_RUNTIME_DIGEST,
        development_numeric_open_count=0,
        admission_numeric_open_count=0,
    )
    authoritative_lock = (
        closure.build_authoritative_universal_trade_rl_u2_development_lock(
            base_lock=base_lock,
            checkpoint_closure=checkpoint_closure,
            training_exposure_evidence=exposure,
            manifest=manifest,
            u2_contract=u2_contract,
        )
    )
    u1_contract = _u1_contract(
        manifest=manifest,
        fit_end_ns=partition.fit_end_ns,
    )
    return (
        manifest,
        partition,
        u2_contract,
        u1_contract,
        predevelopment,
        scope_closure,
        base_lock,
        authoritative_lock,
    )


def test_u2_authoritative_replay_lock_validation_rejects_drift() -> None:
    from trade_rl.workflows import (
        universal_trade_rl_u2_development_authority as authority,
    )

    (
        manifest,
        _partition,
        u2_contract,
        _u1_contract_value,
        predevelopment,
        scope_closure,
        base_lock,
        authoritative_lock,
    ) = _development_lock_bundle()

    authority.require_universal_trade_rl_u2_development_lock_for_replay(
        predevelopment_contract=predevelopment,
        base_lock=base_lock,
        development_lock=authoritative_lock,
        manifest=manifest,
        u2_contract=u2_contract,
        supplied_scope_closure=scope_closure,
        source_tree_digest=_SOURCE_TREE_DIGEST,
        lockfile_digest=_LOCKFILE_DIGEST,
        evaluation_runtime_identity_digest=_RUNTIME_DIGEST,
    )

    drifted_base = replace(
        base_lock,
        development_scope_closure_digest=content_digest({"drift": "scope"}),
        digest="",
    )
    with pytest.raises(ValueError, match="lock|scope|identity|digest"):
        authority.require_universal_trade_rl_u2_development_lock_for_replay(
            predevelopment_contract=predevelopment,
            base_lock=drifted_base,
            development_lock=authoritative_lock,
            manifest=manifest,
            u2_contract=u2_contract,
            supplied_scope_closure=scope_closure,
            source_tree_digest=_SOURCE_TREE_DIGEST,
            lockfile_digest=_LOCKFILE_DIGEST,
            evaluation_runtime_identity_digest=_RUNTIME_DIGEST,
        )


def test_u2_authoritative_session_checks_lock_before_numeric_builder(
    monkeypatch,
) -> None:
    from trade_rl.workflows import (
        universal_trade_rl_u2_development_authority as authority,
    )

    (
        manifest,
        partition,
        u2_contract,
        u1_contract,
        predevelopment,
        scope_closure,
        base_lock,
        authoritative_lock,
    ) = _development_lock_bundle()

    calls: list[dict[str, object]] = []

    class NumericBoundaryReached(RuntimeError):
        pass

    def fake_numeric_builder(**kwargs):
        calls.append(kwargs)
        raise NumericBoundaryReached

    monkeypatch.setattr(
        authority,
        "_build_universal_trade_rl_u2_development_replay_session_unlocked",
        fake_numeric_builder,
    )

    with pytest.raises(NumericBoundaryReached):
        authority.build_authoritative_universal_trade_rl_u2_development_replay_session(
            predevelopment_contract=predevelopment,
            base_lock=base_lock,
            development_lock=authoritative_lock,
            manifest=manifest,
            time_partition=partition,
            u2_contract=u2_contract,
            u1_contract=u1_contract,
            policy_contract=object(),
            normalizer=object(),
            supplied_scope_closure=scope_closure,
            artifact_locators={},
            source_loader=lambda _locator: None,
            environment_factory=lambda _dataset: None,
            source_tree_digest=_SOURCE_TREE_DIGEST,
            lockfile_digest=_LOCKFILE_DIGEST,
            evaluation_runtime_identity_digest=_RUNTIME_DIGEST,
        )
    assert len(calls) == 1

    bad_base = replace(
        base_lock,
        evaluation_runtime_identity_digest=content_digest({"drift": "runtime"}),
        digest="",
    )
    with pytest.raises(ValueError, match="lock|runtime|identity|digest"):
        authority.build_authoritative_universal_trade_rl_u2_development_replay_session(
            predevelopment_contract=predevelopment,
            base_lock=bad_base,
            development_lock=authoritative_lock,
            manifest=manifest,
            time_partition=partition,
            u2_contract=u2_contract,
            u1_contract=u1_contract,
            policy_contract=object(),
            normalizer=object(),
            supplied_scope_closure=scope_closure,
            artifact_locators={},
            source_loader=lambda _locator: None,
            environment_factory=lambda _dataset: None,
            source_tree_digest=_SOURCE_TREE_DIGEST,
            lockfile_digest=_LOCKFILE_DIGEST,
            evaluation_runtime_identity_digest=_RUNTIME_DIGEST,
        )
    assert len(calls) == 1
