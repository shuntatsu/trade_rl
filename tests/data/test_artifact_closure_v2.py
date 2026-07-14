from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import load_market_dataset_artifact, publish_market_dataset_artifact
from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract, MarketBuildConfig
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def _verified_dataset():
    n = 8
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(n) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(n, dtype=np.float64)
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=np.concatenate((close[:1], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(n, 100.0),
        funding_rate=np.zeros(n),
        tradable=np.ones(n, dtype=np.bool_),
    )
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
        )
    ).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (InstrumentContract(symbol="BTCUSDT", listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),),
    )


def _unverified_dataset() -> MarketDataset:
    verified = _verified_dataset()
    return MarketDataset(
        **{
            item: getattr(verified, item)
            for item in (
                "dataset_id", "symbols", "timestamps", "features", "global_features",
                "open", "high", "low", "close", "volume", "funding_rate", "tradable",
                "feature_available", "feature_names", "global_feature_names", "periods_per_year",
            )
        }
    )


def test_publish_rejects_identity_less_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical identity"):
        publish_market_dataset_artifact(tmp_path / "dataset", _unverified_dataset())


def test_loader_rejects_undeclared_files_and_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    publish_market_dataset_artifact(root, _verified_dataset())
    (root / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        load_market_dataset_artifact(root)

    (root / "extra.txt").unlink()
    (root / "link").symlink_to(root / "manifest.json")
    with pytest.raises(ValueError, match="symlink"):
        load_market_dataset_artifact(root)
