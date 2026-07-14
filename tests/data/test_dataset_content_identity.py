from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.data.market import MarketDataset


def dataset(*, fee_rate: float = 0.0) -> MarketDataset:
    n_bars = 8
    close = np.full((n_bars, 1), 100.0)
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=("ASSET",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("global",),
        periods_per_year=8_760,
        fee_rate=np.full((n_bars, 1), fee_rate),
    )


def test_execution_semantics_are_part_of_dataset_identity() -> None:
    zero_fee = dataset(fee_rate=0.0).with_content_identity()
    charged = dataset(fee_rate=0.001).with_content_identity()
    assert zero_fee.dataset_id != charged.dataset_id


def test_formal_artifact_rejects_unidentified_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content identity"):
        write_market_dataset_files(tmp_path, dataset())


def test_content_identity_round_trip_recomputes_all_arrays(tmp_path: Path) -> None:
    original = dataset(fee_rate=0.001).with_content_identity({"source": "unit-test"})
    write_market_dataset_files(tmp_path, original)
    restored = load_market_dataset_artifact(tmp_path)
    assert restored.dataset_id == original.dataset_id
    assert restored.identity_payload_json == original.identity_payload_json

    with pytest.raises(ValueError, match="dataset_id"):
        replace(
            restored,
            dataset_id="f" * 64,
            fee_rate=np.full((restored.n_bars, restored.n_symbols), 0.002),
        )
