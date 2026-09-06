from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.workflows.test_universal_trade_rl_u2_contract import _fixture
from tests.workflows.test_universal_trade_rl_u2_predevelopment import _manifest
from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.checkpointing import CHECKPOINT_POLICY_NAME, CheckpointManifest
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_FINAL_TIMESTEPS,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    build_universal_trade_rl_u2_development_lock,
)
from trade_rl.workflows.universal_trade_rl_u2_training import (
    UniversalTradeRLU2SeedTrainingPlan,
)

_ENVIRONMENT_DIGEST = content_digest({"fixture": "u2-final-environment"})
_SOURCE_CLOSURE_DIGEST = content_digest({"fixture": "u2-source-closure"})
_U1_DIGEST = content_digest({"fixture": "u2-u1"})
_NORMALIZER_DIGEST = content_digest({"fixture": "u2-normalizer"})
_TIME_DIGEST = content_digest({"fixture": "u2-time"})
_TRAINING_CONFIG_DIGEST = content_digest(
    build_universal_trade_rl_u2_training_config().digest_payload()
)


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_development_closure

    return universal_trade_rl_u2_development_closure


def _real_small_u2_contract():
    base = _fixture()
    return build_universal_trade_rl_u2_contract(
        manifest=base.manifest,
        u1_contract=base.u1_contract,
        time_partition=base.time_partition,
        rl_training_provenance=base.rl_training_provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )


def _predevelopment_contract(*, u2_contract_digest: str):
    module = _module()
    manifest = _manifest()
    return module.build_authoritative_universal_trade_rl_u2_predevelopment_contract(
        manifest=manifest,
        u2_contract=module.U2ContractIdentityProbe(
            digest=u2_contract_digest,
            universe_manifest_digest=manifest.digest,
        ),
    )


def _plan(*, u2_contract_digest: str, seed: int) -> UniversalTradeRLU2SeedTrainingPlan:
    return UniversalTradeRLU2SeedTrainingPlan(
        u2_contract_digest=u2_contract_digest,
        source_closure_digest=_SOURCE_CLOSURE_DIGEST,
        u1_contract_digest=_U1_DIGEST,
        normalizer_digest=_NORMALIZER_DIGEST,
        time_partition_digest=_TIME_DIGEST,
        training_config_digest=_TRAINING_CONFIG_DIGEST,
        seed=seed,
        final_timesteps=U2_FINAL_TIMESTEPS,
        primary_candidate=seed == 0,
        production_status="NO-GO",
    )


def _checkpoint(
    *,
    plan: UniversalTradeRLU2SeedTrainingPlan,
    timestep: int = U2_FINAL_TIMESTEPS,
    environment_digest: str = _ENVIRONMENT_DIGEST,
) -> CheckpointManifest:
    policy_digest = content_digest(
        {"fixture": "u2-final-policy", "seed": plan.seed, "timestep": timestep}
    )
    payload = {
        "algorithm": "ppo",
        "environment_digest": environment_digest,
        "observed_timestep": timestep,
        "policy_digest": policy_digest,
        "policy_file": CHECKPOINT_POLICY_NAME,
        "requested_timestep": timestep,
        "schema_version": "policy_checkpoint_v1",
        "seed": plan.seed,
        "training_config_digest": plan.training_config_digest,
    }
    return CheckpointManifest(
        digest=content_digest(payload),
        algorithm="ppo",
        seed=plan.seed,
        requested_timestep=timestep,
        observed_timestep=timestep,
        environment_digest=environment_digest,
        training_config_digest=plan.training_config_digest,
        policy_digest=policy_digest,
        policy_path=Path(f"seed-{plan.seed}/policy.zip"),
    )


def _checkpoint_members(*, u2_contract_digest: str):
    return tuple(
        (
            plan := _plan(u2_contract_digest=u2_contract_digest, seed=seed),
            _checkpoint(plan=plan),
            _ENVIRONMENT_DIGEST,
        )
        for seed in (0, 1, 2)
    )


def _exposure_rows(*, train_symbols: tuple[str, ...]):
    module = _module()
    return tuple(
        module.UniversalTradeRLU2TrainingExposureRow(
            training_seed=seed,
            worker_index=worker,
            concrete_symbol=symbol,
            completed_episode_count=2,
            decision_step_count=5_760,
            partial_final_episode_step_count=(17 if symbol == train_symbols[0] else 0),
            routing_cycle_count=2,
        )
        for seed in (0, 1, 2)
        for worker in range(8)
        for symbol in train_symbols
    )


def test_u2_authoritative_predevelopment_rejects_manifest_identity_mismatch() -> None:
    module = _module()
    manifest = _manifest()
    small_contract = _real_small_u2_contract()

    with pytest.raises(ValueError, match="universe|manifest|identity"):
        module.build_authoritative_universal_trade_rl_u2_predevelopment_contract(
            manifest=manifest,
            u2_contract=small_contract,
        )


def test_u2_final_checkpoint_closure_requires_three_exact_final_members() -> None:
    module = _module()
    predevelopment = _predevelopment_contract(u2_contract_digest="a" * 64)
    members = _checkpoint_members(u2_contract_digest=predevelopment.u2_contract_digest)

    closure = module.build_universal_trade_rl_u2_final_checkpoint_closure(
        predevelopment_contract=predevelopment,
        members=members,
    )

    assert tuple(seed for seed, _ in closure.checkpoint_digests) == (0, 1, 2)
    assert tuple(seed for seed, _ in closure.training_plan_digests) == (0, 1, 2)
    assert tuple(seed for seed, _ in closure.environment_digests) == (0, 1, 2)

    with pytest.raises(ValueError, match="seed|exact|closure|three"):
        module.build_universal_trade_rl_u2_final_checkpoint_closure(
            predevelopment_contract=predevelopment,
            members=members[:2],
        )

    wrong_plan = members[1][0]
    wrong_checkpoint = _checkpoint(plan=wrong_plan, timestep=262_144)
    bad_members = (
        members[0],
        (wrong_plan, wrong_checkpoint, _ENVIRONMENT_DIGEST),
        members[2],
    )
    with pytest.raises(ValueError, match="final|timestep|checkpoint"):
        module.build_universal_trade_rl_u2_final_checkpoint_closure(
            predevelopment_contract=predevelopment,
            members=bad_members,
        )


def test_u2_training_exposure_evidence_requires_complete_seed_worker_symbol_grid() -> (
    None
):
    module = _module()
    predevelopment = _predevelopment_contract(u2_contract_digest="a" * 64)
    train_symbols = _manifest().config.train_symbols
    rows = _exposure_rows(train_symbols=train_symbols)

    evidence = module.build_universal_trade_rl_u2_training_exposure_evidence(
        predevelopment_contract=predevelopment,
        expected_train_symbols=train_symbols,
        rows=rows,
    )

    assert evidence.expected_train_symbols == train_symbols
    assert len(evidence.rows) == 3 * 8 * len(train_symbols)
    assert evidence.posthoc_reweighting_allowed is False

    with pytest.raises(ValueError, match="complete|grid|worker|symbol|row"):
        module.build_universal_trade_rl_u2_training_exposure_evidence(
            predevelopment_contract=predevelopment,
            expected_train_symbols=train_symbols,
            rows=rows[:-1],
        )


def test_u2_authoritative_development_lock_requires_validated_checkpoint_and_exposure() -> (
    None
):
    module = _module()
    predevelopment = _predevelopment_contract(u2_contract_digest="a" * 64)
    train_symbols = _manifest().config.train_symbols
    checkpoint_closure = module.build_universal_trade_rl_u2_final_checkpoint_closure(
        predevelopment_contract=predevelopment,
        members=_checkpoint_members(
            u2_contract_digest=predevelopment.u2_contract_digest,
        ),
    )
    exposure = module.build_universal_trade_rl_u2_training_exposure_evidence(
        predevelopment_contract=predevelopment,
        expected_train_symbols=train_symbols,
        rows=_exposure_rows(train_symbols=train_symbols),
    )
    base_lock = build_universal_trade_rl_u2_development_lock(
        predevelopment_contract=predevelopment,
        u1_contract_digest=_U1_DIGEST,
        u1_normalizer_digest=_NORMALIZER_DIGEST,
        checkpoint_digests=checkpoint_closure.checkpoint_digests,
        development_scope_closure_digest=content_digest({"fixture": "scope-closure"}),
        evaluation_dataset_digests=tuple(
            (symbol, content_digest({"fixture": "eval-view", "symbol": symbol}))
            for symbol in sorted(("DEV01", "DEV02", "DEV03"))
        ),
        source_tree_digest=content_digest({"fixture": "source-tree"}),
        lockfile_digest=content_digest({"fixture": "uv-lock"}),
        evaluation_runtime_identity_digest=content_digest({"fixture": "runtime"}),
        development_numeric_open_count=0,
        admission_numeric_open_count=0,
    )

    lock = module.build_authoritative_universal_trade_rl_u2_development_lock(
        base_lock=base_lock,
        checkpoint_closure=checkpoint_closure,
        training_exposure_evidence=exposure,
    )

    assert lock.base_lock_digest == base_lock.digest
    assert lock.final_checkpoint_closure_digest == checkpoint_closure.digest
    assert lock.training_exposure_evidence_digest == exposure.digest
    assert lock.admission_status == "SEALED"
    assert lock.production_status == "NO-GO"

    altered_base = replace(
        base_lock,
        checkpoint_digests=tuple(reversed(base_lock.checkpoint_digests)),
        digest="",
    )
    with pytest.raises(ValueError, match="checkpoint|closure|mapping|identity"):
        module.build_authoritative_universal_trade_rl_u2_development_lock(
            base_lock=altered_base,
            checkpoint_closure=checkpoint_closure,
            training_exposure_evidence=exposure,
        )
