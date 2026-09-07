from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from tests.workflows.test_universal_trade_rl_u2_contract import _u1_contract
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolRole,
    UniversalTradeRLUniverseConfig,
)
from trade_rl.rl.checkpointing import CHECKPOINT_POLICY_NAME, CheckpointManifest
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitPurpose,
    build_universal_trade_rl_fit_provenance,
)
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_FINAL_TIMESTEPS,
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    UniversalTradeRLU2PreDevelopmentContract,
    build_universal_trade_rl_u2_development_lock,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    build_universal_trade_rl_u2_time_partition,
)
from trade_rl.workflows.universal_trade_rl_u2_training import (
    UniversalTradeRLU2SeedTrainingPlan,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)

_STEP_NS = 15 * 60 * 1_000_000_000
_BARS_PER_DAY = 96
_START_NS = _STEP_NS * 3_000_000
_TOTAL_BARS = 620 * _BARS_PER_DAY
_ENVIRONMENT_DIGEST = content_digest({"fixture": "u2-final-environment"})
_SOURCE_CLOSURE_DIGEST = content_digest({"fixture": "u2-source-closure"})


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_development_closure

    return universal_trade_rl_u2_development_closure


def _symbols(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{index:02d}" for index in range(1, count + 1))


def _role_symbols(
    manifest: UniversalTradeRLUniverseManifest,
    role: UniversalTradeRLSymbolRole,
) -> tuple[str, ...]:
    return tuple(entry.symbol for entry in manifest.entries if entry.role is role)


def _real_manifest(*, salt: str = "real") -> UniversalTradeRLUniverseManifest:
    train = tuple(sorted(("BTCUSDT", *_symbols("TRN", 8))))
    development = _symbols("DEV", 3)
    admission = _symbols("ADM", 3)
    config = UniversalTradeRLUniverseConfig(
        train_symbols=train,
        development_symbols=development,
        admission_symbols=admission,
    )
    sources = tuple(
        UniversalTradeRLSymbolSource(
            symbol=symbol,
            dataset_digest=sha256(f"{salt}:{symbol}".encode()).hexdigest(),
            first_timestamp_ns=_START_NS,
            last_timestamp_ns=_START_NS + (_TOTAL_BARS - 1) * _STEP_NS,
            row_count=_TOTAL_BARS,
        )
        for symbol in sorted((*train, *development, *admission))
    )
    return build_universal_trade_rl_universe_manifest(config=config, sources=sources)


def _real_u2_contract(
    *,
    manifest: UniversalTradeRLUniverseManifest | None = None,
) -> tuple[UniversalTradeRLUniverseManifest, UniversalTradeRLU2Contract]:
    resolved_manifest = manifest or _real_manifest()
    partition = build_universal_trade_rl_u2_time_partition(manifest=resolved_manifest)
    u1_contract = _u1_contract(
        manifest=resolved_manifest,
        fit_end_ns=partition.fit_end_ns,
    )
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=resolved_manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )
    train_symbols = _role_symbols(
        resolved_manifest,
        UniversalTradeRLSymbolRole.TRAIN,
    )
    provenance = build_universal_trade_rl_fit_provenance(
        manifest=resolved_manifest,
        access=access,
        purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
        source_symbols=train_symbols,
        knowledge_cutoff=partition.fit_end_ns,
    )
    u2_contract = build_universal_trade_rl_u2_contract(
        manifest=resolved_manifest,
        u1_contract=u1_contract,
        time_partition=partition,
        rl_training_provenance=provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )
    return resolved_manifest, u2_contract


def _predevelopment_bundle() -> tuple[
    UniversalTradeRLUniverseManifest,
    UniversalTradeRLU2Contract,
    UniversalTradeRLU2PreDevelopmentContract,
]:
    module = _module()
    manifest, u2_contract = _real_u2_contract()
    predevelopment = (
        module.build_authoritative_universal_trade_rl_u2_predevelopment_contract(
            manifest=manifest,
            u2_contract=u2_contract,
        )
    )
    return manifest, u2_contract, predevelopment


def _plan(
    *,
    u2_contract: UniversalTradeRLU2Contract,
    seed: int,
) -> UniversalTradeRLU2SeedTrainingPlan:
    return UniversalTradeRLU2SeedTrainingPlan(
        u2_contract_digest=u2_contract.digest,
        source_closure_digest=_SOURCE_CLOSURE_DIGEST,
        u1_contract_digest=u2_contract.u1_contract_digest,
        normalizer_digest=u2_contract.u1_normalizer_digest,
        time_partition_digest=u2_contract.time_partition_digest,
        training_config_digest=u2_contract.training_config_digest,
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


def _checkpoint_members(*, u2_contract: UniversalTradeRLU2Contract):
    return tuple(
        (
            plan := _plan(u2_contract=u2_contract, seed=seed),
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
    _manifest, u2_contract = _real_u2_contract()
    mismatched_manifest = _real_manifest(salt="mismatch")

    with pytest.raises(ValueError, match="universe|manifest|identity"):
        module.build_authoritative_universal_trade_rl_u2_predevelopment_contract(
            manifest=mismatched_manifest,
            u2_contract=u2_contract,
        )


def test_u2_final_checkpoint_closure_requires_three_exact_final_members() -> None:
    module = _module()
    _manifest, u2_contract, predevelopment = _predevelopment_bundle()
    members = _checkpoint_members(u2_contract=u2_contract)

    closure = module.build_universal_trade_rl_u2_final_checkpoint_closure(
        predevelopment_contract=predevelopment,
        u2_contract=u2_contract,
        members=members,
    )

    assert tuple(seed for seed, _ in closure.checkpoint_digests) == (0, 1, 2)
    assert tuple(seed for seed, _ in closure.training_plan_digests) == (0, 1, 2)
    assert tuple(seed for seed, _ in closure.environment_digests) == (0, 1, 2)

    with pytest.raises(ValueError, match="seed|exact|closure|three"):
        module.build_universal_trade_rl_u2_final_checkpoint_closure(
            predevelopment_contract=predevelopment,
            u2_contract=u2_contract,
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
            u2_contract=u2_contract,
            members=bad_members,
        )


def test_u2_training_exposure_evidence_requires_complete_seed_worker_symbol_grid() -> (
    None
):
    module = _module()
    manifest, u2_contract, predevelopment = _predevelopment_bundle()
    train_symbols = _role_symbols(manifest, UniversalTradeRLSymbolRole.TRAIN)
    rows = _exposure_rows(train_symbols=train_symbols)

    evidence = module.build_universal_trade_rl_u2_training_exposure_evidence(
        predevelopment_contract=predevelopment,
        manifest=manifest,
        u2_contract=u2_contract,
        rows=rows,
    )

    assert evidence.expected_train_symbols == train_symbols
    assert len(evidence.rows) == 3 * 8 * len(train_symbols)
    assert evidence.posthoc_reweighting_allowed is False

    with pytest.raises(ValueError, match="complete|grid|worker|symbol|row"):
        module.build_universal_trade_rl_u2_training_exposure_evidence(
            predevelopment_contract=predevelopment,
            manifest=manifest,
            u2_contract=u2_contract,
            rows=rows[:-1],
        )


def test_u2_authoritative_development_lock_requires_validated_checkpoint_and_exposure() -> (
    None
):
    module = _module()
    manifest, u2_contract, predevelopment = _predevelopment_bundle()
    checkpoint_closure = module.build_universal_trade_rl_u2_final_checkpoint_closure(
        predevelopment_contract=predevelopment,
        u2_contract=u2_contract,
        members=_checkpoint_members(u2_contract=u2_contract),
    )
    train_symbols = _role_symbols(manifest, UniversalTradeRLSymbolRole.TRAIN)
    development_symbols = _role_symbols(
        manifest,
        UniversalTradeRLSymbolRole.DEVELOPMENT,
    )
    exposure = module.build_universal_trade_rl_u2_training_exposure_evidence(
        predevelopment_contract=predevelopment,
        manifest=manifest,
        u2_contract=u2_contract,
        rows=_exposure_rows(train_symbols=train_symbols),
    )
    evaluation_symbols = tuple(sorted((*train_symbols, *development_symbols)))
    base_lock = build_universal_trade_rl_u2_development_lock(
        predevelopment_contract=predevelopment,
        u1_contract_digest=u2_contract.u1_contract_digest,
        u1_normalizer_digest=u2_contract.u1_normalizer_digest,
        checkpoint_digests=checkpoint_closure.checkpoint_digests,
        development_scope_closure_digest=content_digest({"fixture": "scope-closure"}),
        evaluation_dataset_digests=tuple(
            (symbol, content_digest({"fixture": "eval-view", "symbol": symbol}))
            for symbol in evaluation_symbols
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
        manifest=manifest,
        u2_contract=u2_contract,
    )

    assert lock.base_lock_digest == base_lock.digest
    assert lock.final_checkpoint_closure_digest == checkpoint_closure.digest
    assert lock.training_exposure_evidence_digest == exposure.digest
    assert lock.admission_status == "SEALED"
    assert lock.production_status == "NO-GO"

    altered_base = replace(
        base_lock,
        checkpoint_digests=tuple(
            (
                seed,
                content_digest({"fixture": "altered-checkpoint", "seed": seed}),
            )
            for seed, _digest in base_lock.checkpoint_digests
        ),
        digest="",
    )
    with pytest.raises(ValueError, match="checkpoint|closure|mapping|identity"):
        module.build_authoritative_universal_trade_rl_u2_development_lock(
            base_lock=altered_base,
            checkpoint_closure=checkpoint_closure,
            training_exposure_evidence=exposure,
            manifest=manifest,
            u2_contract=u2_contract,
        )

    incomplete_datasets = replace(
        base_lock,
        evaluation_dataset_digests=base_lock.evaluation_dataset_digests[:-1],
        digest="",
    )
    with pytest.raises(ValueError, match="evaluation|dataset|complete|mapping|symbol"):
        module.build_authoritative_universal_trade_rl_u2_development_lock(
            base_lock=incomplete_datasets,
            checkpoint_closure=checkpoint_closure,
            training_exposure_evidence=exposure,
            manifest=manifest,
            u2_contract=u2_contract,
        )
