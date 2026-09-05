from __future__ import annotations

import importlib
from dataclasses import dataclass, replace

import pytest

from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitPurpose,
    build_universal_trade_rl_fit_provenance,
)
from trade_rl.workflows.universal_trade_rl_u1_contract import UniversalTradeRLU1Contract
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    UniversalTradeRLU2DevelopmentScopeClosure,
    build_universal_trade_rl_u2_development_scope_closure,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
    build_universal_trade_rl_u2_time_partition,
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
_START_NS = _STEP_NS * 4_000_000
_TOTAL_BARS = 620 * _BARS_PER_DAY


@dataclass(frozen=True, slots=True)
class ReplayContractFixture:
    manifest: UniversalTradeRLUniverseManifest
    partition: UniversalTradeRLU2TimePartition
    contract: UniversalTradeRLU2Contract
    closure: UniversalTradeRLU2DevelopmentScopeClosure


def _module():
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u2_replay"
        )
    except ModuleNotFoundError:
        pytest.fail("Universal Trade RL U2 deterministic replay is not implemented")


def _fixture() -> ReplayContractFixture:
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("SOLUSDT",),
        admission_symbols=("XRPUSDT",),
    )
    definitions = (
        ("BTCUSDT", "a", 0, 0),
        ("ETHUSDT", "b", 10, 0),
        ("SOLUSDT", "c", 5, 20),
        ("XRPUSDT", "d", 0, 10),
    )
    sources = tuple(
        UniversalTradeRLSymbolSource(
            symbol=symbol,
            dataset_digest=digest_char * 64,
            first_timestamp_ns=_START_NS + start_offset * _STEP_NS,
            last_timestamp_ns=(
                _START_NS
                + start_offset * _STEP_NS
                + (_TOTAL_BARS - start_offset - end_trim - 1) * _STEP_NS
            ),
            row_count=_TOTAL_BARS - start_offset - end_trim,
        )
        for symbol, digest_char, start_offset, end_trim in definitions
    )
    manifest = build_universal_trade_rl_universe_manifest(
        config=config,
        sources=sources,
    )
    partition = build_universal_trade_rl_u2_time_partition(manifest=manifest)
    u1_contract = UniversalTradeRLU1Contract(
        universe_manifest_digest=manifest.digest,
        u0_identity_digest="8" * 64,
        policy_contract_digest="7" * 64,
        normalizer_digest="e" * 64,
        normalizer_provenance_digest="f" * 64,
        normalizer_knowledge_cutoff_ns=partition.fit_end_ns,
        normalizer_clip_value=10.0,
        observation_schema_digest="1" * 64,
        state_layout_digest="2" * 64,
        policy_state_fields=("current_weight",),
        runtime_config_digest="3" * 64,
        execution_policy_digest="4" * 64,
        pretrade_risk_digest="5" * 64,
        portfolio_risk_digest="6" * 64,
    )
    train_access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )
    provenance = build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=train_access,
        purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
        source_symbols=config.train_symbols,
        knowledge_cutoff=partition.fit_end_ns,
    )
    contract = build_universal_trade_rl_u2_contract(
        manifest=manifest,
        u1_contract=u1_contract,
        time_partition=partition,
        rl_training_provenance=provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )
    closure = build_universal_trade_rl_u2_development_scope_closure(
        manifest=manifest,
        time_partition=partition,
        u2_contract=contract,
    )
    return ReplayContractFixture(
        manifest=manifest,
        partition=partition,
        contract=contract,
        closure=closure,
    )


def test_u2_replay_rejects_incomplete_supplied_closure_before_numeric_loading() -> None:
    fixture = _fixture()
    incomplete = replace(
        fixture.closure,
        scopes=fixture.closure.scopes[:-1],
        digest="",
    )
    loader_calls: list[object] = []

    def loader(locator: object):
        loader_calls.append(locator)
        raise AssertionError("numeric loader must not be called")

    with pytest.raises(ValueError, match="canonical|closure|scope"):
        _module().build_universal_trade_rl_u2_development_replay_session(
            manifest=fixture.manifest,
            time_partition=fixture.partition,
            u2_contract=fixture.contract,
            u1_contract=None,
            policy_contract=None,
            normalizer=None,
            supplied_scope_closure=incomplete,
            artifact_locators={},
            source_loader=loader,
            environment_factory=lambda _dataset: None,
        )

    assert loader_calls == []


def test_u2_replay_request_accepts_only_preregistered_evaluation_seeds() -> None:
    module = _module()
    scope = _fixture().closure.scopes[0]

    for seed in (0, 1, 2):
        request = module.UniversalTradeRLU2ReplayRequest(
            scope_digest=scope.digest,
            policy_variant=module.UniversalTradeRLU2ReplayVariant.CASH,
            evaluation_seed=seed,
            paired_candidate_checkpoint_digest="a" * 64,
        )
        assert request.evaluation_seed == seed

    with pytest.raises(ValueError, match="seed"):
        module.UniversalTradeRLU2ReplayRequest(
            scope_digest=scope.digest,
            policy_variant=module.UniversalTradeRLU2ReplayVariant.CASH,
            evaluation_seed=3,
            paired_candidate_checkpoint_digest="a" * 64,
        )
