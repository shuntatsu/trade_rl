from __future__ import annotations

import importlib
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import make_u1_market
from trade_rl.data import MarketDataset, publish_market_dataset_artifact
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


def _source_for_dataset(
    dataset: MarketDataset,
    *,
    fit_start: int,
    fit_stop: int,
) -> U2TrainingSource:
    fit_first = _timestamp_ns(dataset.timestamps[fit_start])
    fit_last = _timestamp_ns(dataset.timestamps[fit_stop - 1])
    return U2TrainingSource(
        symbol=dataset.symbols[0],
        dataset_digest=dataset.dataset_id,
        source_first_timestamp_ns=_timestamp_ns(dataset.timestamps[0]),
        source_last_timestamp_ns=_timestamp_ns(dataset.timestamps[-1]),
        source_row_count=dataset.n_bars,
        fit_first_timestamp_ns=fit_first,
        fit_last_timestamp_ns=fit_last,
        fit_stop_timestamp_ns_exclusive=fit_last + U2_DECISION_STEP_NS,
        fit_bar_count=fit_stop - fit_start,
    )


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
    source = _source_for_dataset(
        source_dataset,
        fit_start=fit_start,
        fit_stop=fit_stop,
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
    assert _timestamp_ns(fit_dataset.timestamps[0]) == source.fit_first_timestamp_ns
    assert _timestamp_ns(fit_dataset.timestamps[-1]) == source.fit_last_timestamp_ns
    np.testing.assert_array_equal(
        fit_dataset.close,
        source_dataset.close[fit_start:fit_stop],
    )


@pytest.mark.parametrize(
    "locators",
    (
        {},
        {"BTCUSDT": Path("btc"), "XRPUSDT": Path("xrp")},
    ),
)
def test_u2_fit_loader_requires_exact_train_locator_closure(
    locators: dict[str, Path],
) -> None:
    source_dataset = make_u1_market(symbol="BTCUSDT", n_bars=512)
    source = _source_for_dataset(source_dataset, fit_start=32, fit_stop=480)

    with pytest.raises(ValueError, match="locator|Train|closure|symbol"):
        _module().load_universal_trade_rl_u2_fit_datasets(
            closure=_single_source_closure(source),
            artifact_locators=locators,
        )


def test_u2_fit_loader_rejects_wrong_canonical_source_content(tmp_path: Path) -> None:
    expected = make_u1_market(symbol="BTCUSDT", n_bars=512, price_scale=1.0)
    wrong = make_u1_market(symbol="BTCUSDT", n_bars=512, price_scale=2.0)
    wrong_root = tmp_path / "wrong"
    publish_market_dataset_artifact(wrong_root, wrong)
    source = _source_for_dataset(expected, fit_start=32, fit_stop=480)

    with pytest.raises(ValueError, match="source|dataset|identity"):
        _module().load_universal_trade_rl_u2_fit_datasets(
            closure=_single_source_closure(source),
            artifact_locators={"BTCUSDT": wrong_root},
        )


def test_u2_fit_loader_identity_is_independent_of_artifact_locator(
    tmp_path: Path,
) -> None:
    source_dataset = make_u1_market(symbol="BTCUSDT", n_bars=512)
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    publish_market_dataset_artifact(root_a, source_dataset)
    shutil.copytree(root_a, root_b)
    source = _source_for_dataset(source_dataset, fit_start=32, fit_stop=480)
    closure = _single_source_closure(source)

    first = _module().load_universal_trade_rl_u2_fit_datasets(
        closure=closure,
        artifact_locators={"BTCUSDT": root_a},
    )
    second = _module().load_universal_trade_rl_u2_fit_datasets(
        closure=closure,
        artifact_locators={"BTCUSDT": root_b},
    )

    assert first["BTCUSDT"].dataset_id == second["BTCUSDT"].dataset_id


def test_u2_fit_loader_rejects_unverified_source_before_slicing() -> None:
    source_dataset = make_u1_market(symbol="BTCUSDT", n_bars=512)
    unverified = replace(source_dataset, identity_payload_json=None)
    source = _source_for_dataset(source_dataset, fit_start=32, fit_stop=480)

    with pytest.raises(ValueError, match="verified|canonical|identity"):
        _module().load_universal_trade_rl_u2_fit_datasets(
            closure=_single_source_closure(source),
            artifact_locators={"BTCUSDT": Path("ignored")},
            loader=lambda _locator: unverified,
        )
