from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest

from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolRole,
    UniversalTradeRLUniverseConfig,
)
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
_START_NS = _STEP_NS * 3_000_000
_TOTAL_BARS = 620 * _BARS_PER_DAY


@dataclass(frozen=True, slots=True)
class U2EvaluationFixture:
    manifest: UniversalTradeRLUniverseManifest
    partition: UniversalTradeRLU2TimePartition
    contract: UniversalTradeRLU2Contract


def _module():
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u2_evaluation"
        )
    except ModuleNotFoundError:
        pytest.fail("Universal Trade RL U2 evaluation scope closure is not implemented")


def _fixture() -> U2EvaluationFixture:
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("SOLUSDT",),
        admission_symbols=("XRPUSDT",),
    )
    sources = tuple(
        UniversalTradeRLSymbolSource(
            symbol=symbol,
            dataset_digest=digest_char * 64,
            first_timestamp_ns=_START_NS,
            last_timestamp_ns=_START_NS + (_TOTAL_BARS - 1) * _STEP_NS,
            row_count=_TOTAL_BARS,
        )
        for symbol, digest_char in (
            ("BTCUSDT", "a"),
            ("ETHUSDT", "b"),
            ("SOLUSDT", "c"),
            ("XRPUSDT", "d"),
        )
    )
    manifest = build_universal_trade_rl_universe_manifest(
        config=config, sources=sources
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
    return U2EvaluationFixture(
        manifest=manifest,
        partition=partition,
        contract=contract,
    )


def test_u2_development_scope_closure_matches_preregistered_a_through_d_cells() -> None:
    fixture = _fixture()
    module = _module()

    closure = module.build_universal_trade_rl_u2_development_scope_closure(
        manifest=fixture.manifest,
        time_partition=fixture.partition,
        u2_contract=fixture.contract,
    )

    expected_cells = {
        "A": (UniversalTradeRLSymbolRole.TRAIN, "seen_time_probe", "diagnostic_only"),
        "B": (
            UniversalTradeRLSymbolRole.DEVELOPMENT,
            "seen_time_probe",
            "mandatory",
        ),
        "C1": (
            UniversalTradeRLSymbolRole.TRAIN,
            "development_future_1",
            "mandatory",
        ),
        "C2": (
            UniversalTradeRLSymbolRole.TRAIN,
            "development_future_2",
            "mandatory",
        ),
        "D1": (
            UniversalTradeRLSymbolRole.DEVELOPMENT,
            "development_future_1",
            "mandatory",
        ),
        "D2": (
            UniversalTradeRLSymbolRole.DEVELOPMENT,
            "development_future_2",
            "mandatory",
        ),
    }
    cell_order = ("A", "B", "C1", "C2", "D1", "D2")

    assert closure.universe_manifest_digest == fixture.manifest.digest
    assert closure.time_partition_digest == fixture.partition.digest
    assert closure.u2_contract_digest == fixture.contract.digest
    assert tuple(dict.fromkeys(scope.cell for scope in closure.scopes)) == cell_order
    assert all(scope.cell != "E" for scope in closure.scopes)
    assert all(scope.source_window != "admission_future" for scope in closure.scopes)

    for cell in cell_order:
        role, window_name, selection_use = expected_cells[cell]
        expected_symbols = tuple(
            entry.symbol for entry in fixture.manifest.entries if entry.role is role
        )
        expected_tiles = fixture.partition.tiles_for(window_name)
        observed = tuple(scope for scope in closure.scopes if scope.cell == cell)

        assert len(observed) == len(expected_symbols) * len(expected_tiles)
        assert tuple(dict.fromkeys(scope.concrete_symbol for scope in observed)) == (
            expected_symbols
        )
        for scope in observed:
            tile = expected_tiles[scope.tile_index]
            entry = fixture.manifest.entry_for(scope.concrete_symbol)
            assert scope.symbol_role is role
            assert scope.selection_use == selection_use
            assert scope.source_window == window_name
            assert scope.source_dataset_digest == entry.dataset_digest
            assert scope.outcome_start_bar_index == tile.start_bar_index
            assert (
                scope.outcome_stop_bar_index_exclusive == tile.stop_bar_index_exclusive
            )
            assert scope.evaluation_range == tile.evaluation_range
            assert scope.decision_count == tile.decision_count

    assert len(closure.digest) == 64
