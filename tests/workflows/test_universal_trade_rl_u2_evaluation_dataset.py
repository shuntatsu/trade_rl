from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import make_u1_market
from trade_rl.data import MarketDataset
from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolRole,
    UniversalTradeRLUniverseConfig,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    UniversalTradeRLU2DevelopmentScopeClosure,
    UniversalTradeRLU2EvaluationScope,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)

_COMMON_VIEW_START = 16
_COMMON_VIEW_STOP = 480


@dataclass(frozen=True, slots=True)
class U2EvaluationDatasetFixture:
    manifest: UniversalTradeRLUniverseManifest
    closure: UniversalTradeRLU2DevelopmentScopeClosure
    sources: dict[str, MarketDataset]
    locators: dict[str, Path]


def _module() -> Any:
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u2_evaluation_dataset"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Universal Trade RL U2 Development dataset loader is not implemented"
        )


def _timestamp_ns(value: np.datetime64) -> int:
    return int(value.astype("datetime64[ns]").astype(np.int64))


def _source(dataset: MarketDataset) -> UniversalTradeRLSymbolSource:
    return UniversalTradeRLSymbolSource(
        symbol=dataset.symbols[0],
        dataset_digest=dataset.dataset_id,
        first_timestamp_ns=_timestamp_ns(dataset.timestamps[0]),
        last_timestamp_ns=_timestamp_ns(dataset.timestamps[-1]),
        row_count=dataset.n_bars,
    )


def _scope(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    dataset: MarketDataset,
    cell: str,
    role: UniversalTradeRLSymbolRole,
    selection_use: str,
) -> UniversalTradeRLU2EvaluationScope:
    view = MarketDatasetView(dataset, _COMMON_VIEW_START, _COMMON_VIEW_STOP)
    return UniversalTradeRLU2EvaluationScope(
        u2_contract_digest="1" * 64,
        universe_manifest_digest=manifest.digest,
        time_partition_digest="2" * 64,
        u1_contract_digest="3" * 64,
        cell=cell,
        selection_use=selection_use,
        symbol_role=role,
        concrete_symbol=dataset.symbols[0],
        source_dataset_digest=dataset.dataset_id,
        evaluation_dataset_digest=view.identity,
        evaluation_source_start_bar_index=_COMMON_VIEW_START,
        evaluation_source_stop_bar_index_exclusive=_COMMON_VIEW_STOP,
        source_window="seen_time_probe",
        tile_index=0,
        outcome_start_bar_index=100,
        outcome_stop_bar_index_exclusive=200,
        evaluation_start_bar_index=99,
        evaluation_stop_bar_index=200,
    )


def _fixture() -> U2EvaluationDatasetFixture:
    sources = {
        symbol: make_u1_market(symbol=symbol, n_bars=512, price_scale=price_scale)
        for symbol, price_scale in (
            ("BTCUSDT", 1.0),
            ("SOLUSDT", 1.2),
            ("XRPUSDT", 0.8),
        )
    }
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT",),
        development_symbols=("SOLUSDT",),
        admission_symbols=("XRPUSDT",),
    )
    manifest = build_universal_trade_rl_universe_manifest(
        config=config,
        sources=tuple(_source(sources[symbol]) for symbol in sorted(sources)),
    )
    scopes = (
        _scope(
            manifest=manifest,
            dataset=sources["BTCUSDT"],
            cell="A",
            role=UniversalTradeRLSymbolRole.TRAIN,
            selection_use="diagnostic_only",
        ),
        _scope(
            manifest=manifest,
            dataset=sources["SOLUSDT"],
            cell="B",
            role=UniversalTradeRLSymbolRole.DEVELOPMENT,
            selection_use="mandatory",
        ),
    )
    closure = UniversalTradeRLU2DevelopmentScopeClosure(
        universe_manifest_digest=manifest.digest,
        time_partition_digest="2" * 64,
        u2_contract_digest="1" * 64,
        scopes=scopes,
    )
    return U2EvaluationDatasetFixture(
        manifest=manifest,
        closure=closure,
        sources=sources,
        locators={
            "BTCUSDT": Path("btc-source"),
            "SOLUSDT": Path("sol-source"),
        },
    )


def test_u2_development_dataset_loader_materializes_exact_common_views_once() -> None:
    fixture = _fixture()
    by_locator = {
        fixture.locators[symbol]: fixture.sources[symbol] for symbol in fixture.locators
    }
    calls: list[Path] = []

    def loader(locator: object) -> MarketDataset:
        assert isinstance(locator, Path)
        calls.append(locator)
        return by_locator[locator]

    loaded = _module().load_universal_trade_rl_u2_development_evaluation_datasets(
        manifest=fixture.manifest,
        scope_closure=fixture.closure,
        artifact_locators=fixture.locators,
        loader=loader,
    )

    assert tuple(loaded) == ("BTCUSDT", "SOLUSDT")
    assert calls == [Path("btc-source"), Path("sol-source")]
    for symbol, dataset in loaded.items():
        source = fixture.sources[symbol]
        expected_view = MarketDatasetView(
            source,
            _COMMON_VIEW_START,
            _COMMON_VIEW_STOP,
        )
        assert dataset.dataset_id == expected_view.identity
        assert dataset.n_bars == _COMMON_VIEW_STOP - _COMMON_VIEW_START
        np.testing.assert_array_equal(
            dataset.close,
            source.close[_COMMON_VIEW_START:_COMMON_VIEW_STOP],
        )


@pytest.mark.parametrize(
    "locators",
    (
        {"BTCUSDT": Path("btc-source")},
        {
            "BTCUSDT": Path("btc-source"),
            "SOLUSDT": Path("sol-source"),
            "XRPUSDT": Path("xrp-source"),
        },
    ),
)
def test_u2_development_dataset_loader_rejects_locator_drift_before_numeric_load(
    locators: dict[str, Path],
) -> None:
    fixture = _fixture()
    calls = 0

    def loader(_locator: object) -> MarketDataset:
        nonlocal calls
        calls += 1
        raise AssertionError("loader must not be called for invalid locator closure")

    with pytest.raises(ValueError, match="locator|closure|Development|symbol"):
        _module().load_universal_trade_rl_u2_development_evaluation_datasets(
            manifest=fixture.manifest,
            scope_closure=fixture.closure,
            artifact_locators=locators,
            loader=loader,
        )
    assert calls == 0


def test_u2_development_dataset_loader_rejects_admission_scope_before_numeric_load() -> (
    None
):
    fixture = _fixture()
    admission = fixture.sources["XRPUSDT"]
    original = fixture.closure.scopes[0]
    admission_view = MarketDatasetView(
        admission,
        _COMMON_VIEW_START,
        _COMMON_VIEW_STOP,
    )
    spoofed = replace(
        original,
        concrete_symbol="XRPUSDT",
        source_dataset_digest=admission.dataset_id,
        evaluation_dataset_digest=admission_view.identity,
        digest="",
    )
    closure = replace(
        fixture.closure,
        scopes=(spoofed, fixture.closure.scopes[1]),
        digest="",
    )
    calls = 0

    def loader(_locator: object) -> MarketDataset:
        nonlocal calls
        calls += 1
        raise AssertionError("Admission source must remain sealed")

    with pytest.raises((PermissionError, ValueError), match="Admission|role|scope"):
        _module().load_universal_trade_rl_u2_development_evaluation_datasets(
            manifest=fixture.manifest,
            scope_closure=closure,
            artifact_locators={
                "SOLUSDT": Path("sol-source"),
                "XRPUSDT": Path("xrp-source"),
            },
            loader=loader,
        )
    assert calls == 0


def test_u2_development_dataset_loader_rejects_wrong_canonical_source_identity() -> (
    None
):
    fixture = _fixture()
    wrong_btc = make_u1_market(symbol="BTCUSDT", n_bars=512, price_scale=2.0)
    by_locator = {
        Path("btc-source"): wrong_btc,
        Path("sol-source"): fixture.sources["SOLUSDT"],
    }

    with pytest.raises(ValueError, match="source|dataset|identity"):
        _module().load_universal_trade_rl_u2_development_evaluation_datasets(
            manifest=fixture.manifest,
            scope_closure=fixture.closure,
            artifact_locators=fixture.locators,
            loader=lambda locator: by_locator[locator],
        )


def test_u2_development_dataset_loader_rejects_unverified_source() -> None:
    fixture = _fixture()
    unverified_btc = replace(
        fixture.sources["BTCUSDT"],
        identity_payload_json=None,
    )
    by_locator = {
        Path("btc-source"): unverified_btc,
        Path("sol-source"): fixture.sources["SOLUSDT"],
    }

    with pytest.raises(ValueError, match="verified|canonical|identity"):
        _module().load_universal_trade_rl_u2_development_evaluation_datasets(
            manifest=fixture.manifest,
            scope_closure=fixture.closure,
            artifact_locators=fixture.locators,
            loader=lambda locator: by_locator[locator],
        )


def test_u2_development_dataset_loader_rejects_missing_manifest_evaluation_symbol_before_numeric_load() -> None:
    fixture = _fixture()
    incomplete_closure = replace(
        fixture.closure,
        scopes=(fixture.closure.scopes[0],),
        digest="",
    )
    calls = 0

    def loader(_locator: object) -> MarketDataset:
        nonlocal calls
        calls += 1
        raise AssertionError("loader must not be called for incomplete scope closure")

    with pytest.raises(ValueError, match="complete|closure|Development|symbol"):
        _module().load_universal_trade_rl_u2_development_evaluation_datasets(
            manifest=fixture.manifest,
            scope_closure=incomplete_closure,
            artifact_locators={"BTCUSDT": Path("btc-source")},
            loader=loader,
        )
    assert calls == 0
