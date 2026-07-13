from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.artifacts import (
    MarketDatasetView,
    load_market_dataset_artifact,
    write_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset


def _dataset() -> MarketDataset:
    n_bars = 12
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.linspace(100.0, 111.0, n_bars, dtype=np.float64)[:, None]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.linspace(0.0, 1.0, n_bars, dtype=np.float32)[:, None, None],
        global_features=np.linspace(1.0, 2.0, n_bars, dtype=np.float32)[:, None],
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 10_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("momentum",),
        global_feature_names=("market",),
        periods_per_year=8_760,
        fee_rate=np.full((n_bars, 1), 0.0005),
        borrow_available=np.ones((n_bars, 1), dtype=np.bool_),
        cash_rate=np.linspace(0.0, 0.001, n_bars),
    )


def test_market_dataset_artifact_round_trip_preserves_resolved_arrays(
    tmp_path: Path,
) -> None:
    original = _dataset()

    manifest_path = write_market_dataset_artifact(tmp_path, original)
    restored = load_market_dataset_artifact(tmp_path)

    assert manifest_path == tmp_path / "manifest.json"
    assert restored.dataset_id == original.dataset_id
    assert restored.symbols == original.symbols
    assert restored.feature_names == original.feature_names
    assert restored.global_feature_names == original.global_feature_names
    for field_name in (
        "timestamps",
        "features",
        "global_features",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "tradable",
        "feature_available",
        "fee_rate",
        "borrow_available",
        "cash_rate",
    ):
        np.testing.assert_array_equal(
            getattr(restored, field_name),
            getattr(original, field_name),
        )


def test_market_dataset_artifact_rejects_tampered_npz(tmp_path: Path) -> None:
    write_market_dataset_artifact(tmp_path, _dataset())
    arrays_path = tmp_path / "arrays.npz"
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="digest"):
        load_market_dataset_artifact(tmp_path)


def test_market_dataset_view_rejects_escape_and_materializes_range() -> None:
    dataset = _dataset()
    view = MarketDatasetView(dataset=dataset, start=2, stop=10)

    child = view.subview(3, 8)
    materialized = child.materialize()

    assert child.start == 3
    assert child.stop == 8
    assert materialized.n_bars == 5
    np.testing.assert_array_equal(materialized.close[:, 0], dataset.close[3:8, 0])
    assert materialized.dataset_id != dataset.dataset_id

    with pytest.raises(ValueError, match="outside"):
        view.subview(1, 8)
    with pytest.raises(ValueError, match="outside"):
        view.subview(3, 11)
