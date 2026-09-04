from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest

from tests.workflows.test_universal_trade_rl_u2_contract import _fixture
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
)


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


def test_u2_seed_training_plan_binds_contract_source_closure_and_fixed_budget() -> None:
    fixture = _runtime_fixture()
    module = _module()

    plan = module.build_universal_trade_rl_u2_seed_training_plan(
        contract=fixture.contract,
        source_closure=fixture.closure,
        seed=0,
    )

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
