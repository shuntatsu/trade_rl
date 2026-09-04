from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.workflows.test_universal_trade_rl_u2_contract import _fixture
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.rl.checkpointing import (
    CHECKPOINT_POLICY_NAME,
    CheckpointManifest,
)
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
)

_ENVIRONMENT_DIGEST = content_digest({"fixture": "u2-training-environment"})


@dataclass(frozen=True, slots=True)
class U2TrainingRuntimeFixture:
    contract: UniversalTradeRLU2Contract
    closure: U2TrainingSourceClosure


def _module():
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u2_training"
        )
    except ModuleNotFoundError:
        pytest.fail("Universal Trade RL U2 training orchestration is not implemented")


def _runtime_fixture() -> U2TrainingRuntimeFixture:
    base = _fixture()
    contract = build_universal_trade_rl_u2_contract(
        manifest=base.manifest,
        u1_contract=base.u1_contract,
        time_partition=base.time_partition,
        rl_training_provenance=base.rl_training_provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )
    fit = base.time_partition.window("fit")
    entries = tuple(
        sorted(
            (
                entry
                for entry in base.manifest.entries
                if entry.role is UniversalTradeRLSymbolRole.TRAIN
            ),
            key=lambda entry: entry.symbol,
        )
    )
    closure = U2TrainingSourceClosure(
        u2_contract_digest=contract.digest,
        universe_manifest_digest=contract.universe_manifest_digest,
        u1_contract_digest=contract.u1_contract_digest,
        normalizer_digest=contract.u1_normalizer_digest,
        normalizer_provenance_digest=base.u1_contract.normalizer_provenance_digest,
        time_partition_digest=contract.time_partition_digest,
        fit_first_timestamp_ns=fit.first_timestamp_ns,
        fit_last_timestamp_ns=fit.last_timestamp_ns,
        fit_stop_timestamp_ns_exclusive=(
            fit.last_timestamp_ns + base.time_partition.decision_step_ns
        ),
        fit_bar_count=fit.bar_count,
        sources=tuple(
            U2TrainingSource(
                symbol=entry.symbol,
                dataset_digest=entry.dataset_digest,
                source_first_timestamp_ns=entry.first_timestamp_ns,
                source_last_timestamp_ns=entry.last_timestamp_ns,
                source_row_count=entry.row_count,
                fit_first_timestamp_ns=fit.first_timestamp_ns,
                fit_last_timestamp_ns=fit.last_timestamp_ns,
                fit_stop_timestamp_ns_exclusive=(
                    fit.last_timestamp_ns + base.time_partition.decision_step_ns
                ),
                fit_bar_count=fit.bar_count,
            )
            for entry in entries
        ),
    )
    return U2TrainingRuntimeFixture(contract=contract, closure=closure)


def _plan(fixture: U2TrainingRuntimeFixture):
    return _module().build_universal_trade_rl_u2_seed_training_plan(
        contract=fixture.contract,
        source_closure=fixture.closure,
        seed=0,
    )


def _checkpoint(*, plan, timestep: int) -> CheckpointManifest:
    policy_digest = content_digest({"fixture": "u2-policy", "timestep": timestep})
    payload = {
        "algorithm": "ppo",
        "environment_digest": _ENVIRONMENT_DIGEST,
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
        environment_digest=_ENVIRONMENT_DIGEST,
        training_config_digest=plan.training_config_digest,
        policy_digest=policy_digest,
        policy_path=Path("policy.zip"),
    )


def test_u2_seed_training_plan_binds_contract_source_closure_and_fixed_budget() -> None:
    fixture = _runtime_fixture()
    plan = _plan(fixture)

    assert plan.u2_contract_digest == fixture.contract.digest
    assert plan.source_closure_digest == fixture.closure.digest
    assert plan.u1_contract_digest == fixture.contract.u1_contract_digest
    assert plan.normalizer_digest == fixture.contract.u1_normalizer_digest
    assert plan.time_partition_digest == fixture.contract.time_partition_digest
    assert plan.training_config_digest == fixture.contract.training_config_digest
    assert plan.seed == 0
    assert plan.final_timesteps == 524_288
    assert plan.primary_candidate is True
    assert plan.production_status == "NO-GO"
    assert len(plan.digest) == 64


def test_u2_selection_checkpoint_requires_exact_final_fixed_budget() -> None:
    fixture = _runtime_fixture()
    module = _module()
    plan = _plan(fixture)
    validator = getattr(
        module,
        "require_universal_trade_rl_u2_selection_checkpoint",
        None,
    )
    assert callable(validator), "U2 final-only checkpoint validator is not implemented"

    final = _checkpoint(plan=plan, timestep=524_288)
    assert (
        validator(
            plan=plan,
            checkpoint=final,
            expected_environment_digest=_ENVIRONMENT_DIGEST,
        )
        is final
    )

    for timestep in (32_768, 262_144):
        with pytest.raises(ValueError, match="final|timestep|checkpoint"):
            validator(
                plan=plan,
                checkpoint=_checkpoint(plan=plan, timestep=timestep),
                expected_environment_digest=_ENVIRONMENT_DIGEST,
            )
