from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import make_u1_market
from trade_rl.data import publish_market_dataset_artifact
from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import U2_DECISION_STEP_NS


def _module() -> Any:
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u2_fit_dataset"
        )
    except ModuleNotFoundError:
        pytest.fail("Universal Trade RL U2 FIT dataset loader is not implemented")


def _timestamp_ns(value: np.datetime64) -> int:
    return int(value.astype("datetime64[ns]").astype(np.int64))


def _single_source_closure(source: U2TrainingSource) -> U2TrainingSourceClosure:
    return U2TrainingSourceClosure(
        u2_contract_digest="1" * 64,
        universe_manifest_digest="2" * 64,
        u1_contract_digest="3" * 64,
        normalizer_digest="4" * 64,
        normalizer_provenance_digest="5" * 64,
        time_partition_digest="6" * 64,
        fit_first_timestamp_ns=source.fit_first_timestamp_ns,
        fit_last_timestamp_ns=source.fit_last_timestamp_ns,
        fit_stop_timestamp_ns_exclusive=source.fit_stop_timestamp_ns_exclusive,
        fit_bar_count=source.fit_bar_count,
        sources=(source,),
    )


def test_u2_fit_loader_materializes_exact_market_dataset_view(tmp_path: Path) -> None:
    source_dataset = make_u1_market(symbol="BTCUSDT", n_bars=10_000)
    artifact_root = tmp_path / "BTCUSDT"
    publish_market_dataset_artifact(artifact_root, source_dataset)

    fit_start = 1_000
    fit_stop = 9_000
    fit_first = _timestamp_ns(source_dataset.timestamps[fit_start])
    fit_last = _timestamp_ns(source_dataset.timestamps[fit_stop - 1])
    source = U2TrainingSource(
        symbol="BTCUSDT",
        dataset_digest=source_dataset.dataset_id,
        source_first_timestamp_ns=_timestamp_ns(source_dataset.timestamps[0]),
        source_last_timestamp_ns=_timestamp_ns(source_dataset.timestamps[-1]),
        source_row_count=source_dataset.n_bars,
        fit_first_timestamp_ns=fit_first,
        fit_last_timestamp_ns=fit_last,
        fit_stop_timestamp_ns_exclusive=fit_last + U2_DECISION_STEP_NS,
        fit_bar_count=fit_stop - fit_start,
    )

    loaded = _module().load_universal_trade_rl_u2_fit_datasets(
        closure=_single_source_closure(source),
        artifact_locators={"BTCUSDT": artifact_root},
    )

    assert tuple(loaded) == ("BTCUSDT",)
    fit_dataset = loaded["BTCUSDT"]
    expected_view = MarketDatasetView(source_dataset, fit_start, fit_stop)
    assert fit_dataset.dataset_id == expected_view.identity
    assert fit_dataset.n_bars == fit_stop - fit_start
    assert _timestamp_ns(fit_dataset.timestamps[0]) == fit_first
    assert _timestamp_ns(fit_dataset.timestamps[-1]) == fit_last
    np.testing.assert_array_equal(
        fit_dataset.close,
        source_dataset.close[fit_start:fit_stop],
    )
