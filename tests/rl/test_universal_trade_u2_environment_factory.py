from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from tests.rl.test_universal_trade_u2_environment import U2EnvironmentFixture
from tests.rl.universal_trade_test_support import make_u1_wrapper
from tests.workflows.test_universal_trade_rl_u2_contract import _fixture
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.rl.universal_episode_router import DETERMINISTIC_INSTRUMENT_ROUTER_SCHEMA
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_preflight import U2TrainingSourceClosure

pytest_plugins = ("tests.rl.test_universal_trade_u2_environment",)

_U2_ENVIRONMENT_MODULE = "trade_rl.workflows.universal_trade_rl_u2_environment"


def _factory_class() -> type[Any]:
    module = importlib.import_module(_U2_ENVIRONMENT_MODULE)
    factory_class = getattr(module, "UniversalTradeRLU2EnvironmentFactory", None)
    if not isinstance(factory_class, type):
        pytest.fail("U2 high-level indexed environment factory is not implemented")
    return factory_class


def _adapt_contract(
    fixture: U2EnvironmentFixture,
) -> tuple[UniversalTradeRLU2Contract, U2TrainingSourceClosure]:
    canonical_fixture = _fixture()
    canonical_contract = build_universal_trade_rl_u2_contract(
        manifest=canonical_fixture.manifest,
        u1_contract=canonical_fixture.u1_contract,
        time_partition=canonical_fixture.time_partition,
        rl_training_provenance=canonical_fixture.rl_training_provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )
    fit_end_ns = fixture.closure.fit_last_timestamp_ns
    router_contract_digest = content_digest(
        {
            "schema_version": "universal_trade_rl_u2_router_contract_v1",
            "router_schema": DETERMINISTIC_INSTRUMENT_ROUTER_SCHEMA,
            "symbol_source": "u0_manifest_train_role",
            "universe_manifest_digest": fixture.manifest.digest,
            "partition_digest": fixture.closure.time_partition_digest,
            "cycle_contract": "each_train_symbol_exactly_once_before_repeat",
            "routing_scope": "environment_local",
            "deterministic": True,
        }
    )
    episode_sampling_contract_digest = content_digest(
        {
            "schema_version": "universal_trade_rl_u2_episode_sampling_contract_v1",
            "time_partition_digest": fixture.closure.time_partition_digest,
            "fit_end_ns": fit_end_ns,
            "fit_scope_only": True,
            "episode_hours": 720,
            "eligible_start_sampling": "uniform",
            "outcome_must_not_exceed_fit_end": True,
            "deterministic_resumable": True,
            "regime_oversampling": False,
            "stress_oversampling": False,
        }
    )
    baseline_contract_digest = content_digest(
        {
            "schema_version": "universal_trade_rl_u2_baseline_contract_v1",
            "u1_contract_digest": fixture.u1_contract.digest,
            "primary_baseline": {"name": "cash", "constant_action": 0.0},
            "diagnostic_baselines": (
                {"name": "constant_long", "constant_action": 1.0},
                {"name": "constant_short", "constant_action": -1.0},
            ),
            "same_risk_execution_accounting_scope": True,
        }
    )
    contract = replace(
        canonical_contract,
        universe_manifest_digest=fixture.manifest.digest,
        u1_contract_digest=fixture.u1_contract.digest,
        u1_normalizer_digest=fixture.normalizer.digest,
        u1_normalizer_knowledge_cutoff_ns=fit_end_ns,
        time_partition_digest=fixture.closure.time_partition_digest,
        fit_end_ns=fit_end_ns,
        rl_training_provenance_digest=content_digest(
            {"fixture": "u2-high-level-factory-rl-training-provenance"}
        ),
        rl_training_knowledge_cutoff_ns=fit_end_ns,
        router_contract_digest=router_contract_digest,
        episode_sampling_contract_digest=episode_sampling_contract_digest,
        baseline_contract_digest=baseline_contract_digest,
        digest="",
    )
    closure = replace(
        fixture.closure,
        u2_contract_digest=contract.digest,
        digest="",
    )
    return contract, closure


def _build_factory_inputs(
    fixture: U2EnvironmentFixture,
) -> tuple[
    UniversalTradeRLU2Contract,
    U2TrainingSourceClosure,
    dict[str, str],
    Callable[[str | Path], MarketDataset],
    list[str],
    Callable[[MarketDataset], UniversalTradeEnvironment],
    list[tuple[str, int, UniversalTradeEnvironment]],
]:
    contract, closure = _adapt_contract(fixture)
    locators = {symbol: f"fixture://{symbol}" for symbol in fixture.train_symbols}
    by_locator = {
        locators[symbol]: fixture.datasets[symbol] for symbol in fixture.train_symbols
    }
    source_loads: list[str] = []
    child_builds: list[tuple[str, int, UniversalTradeEnvironment]] = []

    def source_loader(locator: str | Path) -> MarketDataset:
        token = str(locator)
        source_loads.append(token)
        return by_locator[token]

    def u1_environment_factory(dataset: MarketDataset) -> UniversalTradeEnvironment:
        environment = make_u1_wrapper(
            dataset=dataset,
            contract=fixture.policy_contract,
            normalizer=fixture.normalizer,
        )
        child_builds.append((dataset.symbols[0], id(dataset), environment))
        return environment

    return (
        contract,
        closure,
        locators,
        source_loader,
        source_loads,
        u1_environment_factory,
        child_builds,
    )


def _construct_factory(
    fixture: U2EnvironmentFixture,
    *,
    run_seed: int = 0,
):
    (
        contract,
        closure,
        locators,
        source_loader,
        source_loads,
        u1_environment_factory,
        child_builds,
    ) = _build_factory_inputs(fixture)
    factory = _factory_class()(
        u2_contract=contract,
        source_closure=closure,
        artifact_locators=locators,
        u1_contract=fixture.u1_contract,
        policy_contract=fixture.policy_contract,
        normalizer=fixture.normalizer,
        u1_environment_factory=u1_environment_factory,
        run_seed=run_seed,
        source_artifact_loader=source_loader,
    )
    return factory, closure, locators, source_loads, child_builds


def test_u2_high_level_factory_materializes_sources_once_and_reuses_fit_datasets(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    factory, closure, locators, source_loads, child_builds = _construct_factory(
        u2_environment_fixture
    )

    assert source_loads == [locators[source.symbol] for source in closure.sources]
    assert tuple(binding.concrete_symbol for binding in factory.bindings) == tuple(
        source.symbol for source in closure.sources
    )

    worker0 = cast(EpisodeRoutedSingleInstrumentEnv, factory())
    worker1 = cast(EpisodeRoutedSingleInstrumentEnv, factory.for_environment_index(1)())
    try:
        assert source_loads == [locators[source.symbol] for source in closure.sources]
        assert worker0.environment_index == 0
        assert worker1.environment_index == 1
        assert worker0.environment_digest == factory.environment_generation_digest
        assert worker1.environment_digest == factory.environment_generation_digest
        assert worker0.router_digest != worker1.router_digest

        assert len(child_builds) == 2 * len(closure.sources)
        environment_ids = tuple(id(record[2]) for record in child_builds)
        assert len(set(environment_ids)) == len(environment_ids)
        for symbol in (source.symbol for source in closure.sources):
            dataset_ids = {
                dataset_object_id
                for observed_symbol, dataset_object_id, _environment in child_builds
                if observed_symbol == symbol
            }
            assert len(dataset_ids) == 1
    finally:
        worker0.close()
        worker1.close()


def test_u2_high_level_factory_call_matches_fresh_worker_zero(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    factory, _closure, _locators, _source_loads, _child_builds = _construct_factory(
        u2_environment_fixture
    )
    direct = cast(EpisodeRoutedSingleInstrumentEnv, factory())
    indexed = cast(EpisodeRoutedSingleInstrumentEnv, factory.for_environment_index(0)())
    try:
        assert direct is not indexed
        assert direct.environment_index == 0
        assert indexed.environment_index == 0
        assert direct.environment_digest == indexed.environment_digest
        assert direct.router_digest == indexed.router_digest
    finally:
        direct.close()
        indexed.close()


@pytest.mark.parametrize("index", (-1, 8, True))
def test_u2_high_level_factory_rejects_worker_index_outside_fixed_vector_set(
    u2_environment_fixture: U2EnvironmentFixture,
    index: object,
) -> None:
    factory, _closure, _locators, _source_loads, _child_builds = _construct_factory(
        u2_environment_fixture
    )

    with pytest.raises(ValueError, match="index|environment|worker|0|7|8"):
        factory.for_environment_index(index)


def test_u2_high_level_factory_rejects_unregistered_seed_before_source_io(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    (
        contract,
        closure,
        locators,
        source_loader,
        source_loads,
        u1_environment_factory,
        _child_builds,
    ) = _build_factory_inputs(u2_environment_fixture)

    with pytest.raises(ValueError, match="seed|training|member"):
        _factory_class()(
            u2_contract=contract,
            source_closure=closure,
            artifact_locators=locators,
            u1_contract=u2_environment_fixture.u1_contract,
            policy_contract=u2_environment_fixture.policy_contract,
            normalizer=u2_environment_fixture.normalizer,
            u1_environment_factory=u1_environment_factory,
            run_seed=99,
            source_artifact_loader=source_loader,
        )

    assert source_loads == []
