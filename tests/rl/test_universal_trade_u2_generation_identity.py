from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Callable, cast

import pytest

from tests.workflows.test_universal_trade_rl_u2_contract import _fixture
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
)

_U2_ENVIRONMENT_MODULE = "trade_rl.workflows.universal_trade_rl_u2_environment"


def _generation_builder() -> Callable[..., str]:
    module = importlib.import_module(_U2_ENVIRONMENT_MODULE)
    builder = getattr(
        module,
        "build_universal_trade_rl_u2_environment_generation_digest",
        None,
    )
    if not callable(builder):
        pytest.fail("U2 run-level environment generation digest is not implemented")
    return cast(Callable[..., str], builder)


def _generation_fixture() -> tuple[
    UniversalTradeRLU2Contract,
    U2TrainingSourceClosure,
    tuple[InstrumentDatasetBinding, ...],
]:
    fixture = _fixture()
    contract = build_universal_trade_rl_u2_contract(
        manifest=fixture.manifest,
        u1_contract=fixture.u1_contract,
        time_partition=fixture.time_partition,
        rl_training_provenance=fixture.rl_training_provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )
    fit = fixture.time_partition.window("fit")
    train_entries = tuple(
        entry
        for entry in fixture.manifest.entries
        if entry.role is UniversalTradeRLSymbolRole.TRAIN
    )
    sources = tuple(
        U2TrainingSource(
            symbol=entry.symbol,
            dataset_digest=entry.dataset_digest,
            source_first_timestamp_ns=entry.first_timestamp_ns,
            source_last_timestamp_ns=entry.last_timestamp_ns,
            source_row_count=entry.row_count,
            fit_first_timestamp_ns=fit.first_timestamp_ns,
            fit_last_timestamp_ns=fit.last_timestamp_ns,
            fit_stop_timestamp_ns_exclusive=(
                fit.last_timestamp_ns + fixture.time_partition.decision_step_ns
            ),
            fit_bar_count=fit.bar_count,
        )
        for entry in train_entries
    )
    closure = U2TrainingSourceClosure(
        u2_contract_digest=contract.digest,
        universe_manifest_digest=fixture.manifest.digest,
        u1_contract_digest=fixture.u1_contract.digest,
        normalizer_digest=fixture.u1_contract.normalizer_digest,
        normalizer_provenance_digest=fixture.u1_contract.normalizer_provenance_digest,
        time_partition_digest=fixture.time_partition.digest,
        fit_first_timestamp_ns=fit.first_timestamp_ns,
        fit_last_timestamp_ns=fit.last_timestamp_ns,
        fit_stop_timestamp_ns_exclusive=(
            fit.last_timestamp_ns + fixture.time_partition.decision_step_ns
        ),
        fit_bar_count=fit.bar_count,
        sources=sources,
    )
    bindings = tuple(
        InstrumentDatasetBinding(
            concrete_symbol=source.symbol,
            source_dataset_id=content_digest(
                {
                    "fixture": "u2-generation-fit-view",
                    "symbol": source.symbol,
                }
            ),
            symbol_dataset_digest=source.dataset_digest,
            execution_metadata_digest=content_digest(
                {
                    "fixture": "u2-generation-execution",
                    "symbol": source.symbol,
                }
            ),
            instrument_descriptor_digest=content_digest(
                {
                    "fixture": "u2-generation-descriptor",
                    "symbol": source.symbol,
                }
            ),
            split="train",
        )
        for source in sources
    )
    return contract, closure, bindings


def test_u2_environment_generation_digest_matches_exact_preregistered_payload() -> None:
    contract, closure, bindings = _generation_fixture()

    digest = _generation_builder()(
        u2_contract=contract,
        source_closure=closure,
        bindings=bindings,
        run_seed=0,
    )

    expected_payload = {
        "schema_version": "universal_trade_rl_u2_environment_generation_v1",
        "u2_contract_digest": contract.digest,
        "source_closure_digest": closure.digest,
        "training_config_digest": contract.training_config_digest,
        "run_seed": 0,
        "n_envs": 8,
        "vector_environment_mode": "in_process",
        "environment_indices": tuple(range(8)),
        "binding_digests": tuple(
            (source.symbol, binding.digest)
            for source, binding in zip(closure.sources, bindings, strict=True)
        ),
        "router_contract_digest": contract.router_contract_digest,
        "episode_sampling_contract_digest": contract.episode_sampling_contract_digest,
        "episode_seed_schema": "universal_trade_rl_u2_episode_seed_v1",
    }
    assert digest == content_digest(expected_payload)


def test_u2_environment_generation_digest_canonicalizes_binding_order() -> None:
    contract, closure, bindings = _generation_fixture()
    builder = _generation_builder()

    canonical = builder(
        u2_contract=contract,
        source_closure=closure,
        bindings=bindings,
        run_seed=0,
    )
    reordered = builder(
        u2_contract=contract,
        source_closure=closure,
        bindings=tuple(reversed(bindings)),
        run_seed=0,
    )

    assert reordered == canonical


def test_u2_environment_generation_digest_changes_with_seed_source_or_binding() -> None:
    contract, closure, bindings = _generation_fixture()
    builder = _generation_builder()
    baseline = builder(
        u2_contract=contract,
        source_closure=closure,
        bindings=bindings,
        run_seed=0,
    )

    assert (
        builder(
            u2_contract=contract,
            source_closure=closure,
            bindings=bindings,
            run_seed=1,
        )
        != baseline
    )

    changed_sources = list(closure.sources)
    changed_sources[0] = replace(changed_sources[0], dataset_digest="e" * 64)
    changed_closure = replace(
        closure,
        sources=tuple(changed_sources),
        digest="",
    )
    changed_bindings = list(bindings)
    changed_bindings[0] = replace(
        changed_bindings[0],
        symbol_dataset_digest="e" * 64,
    )
    assert (
        builder(
            u2_contract=contract,
            source_closure=changed_closure,
            bindings=tuple(changed_bindings),
            run_seed=0,
        )
        != baseline
    )

    one_binding_changed = list(bindings)
    one_binding_changed[0] = replace(
        one_binding_changed[0],
        execution_metadata_digest="f" * 64,
    )
    assert (
        builder(
            u2_contract=contract,
            source_closure=closure,
            bindings=tuple(one_binding_changed),
            run_seed=0,
        )
        != baseline
    )


def test_u2_environment_generation_digest_rejects_unregistered_member_seed() -> None:
    contract, closure, bindings = _generation_fixture()

    with pytest.raises(ValueError, match="seed|training|member"):
        _generation_builder()(
            u2_contract=contract,
            source_closure=closure,
            bindings=bindings,
            run_seed=99,
        )
