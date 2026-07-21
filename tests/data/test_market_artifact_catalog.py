from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trade_rl.catalog import ArtifactKind
from trade_rl.data import artifact as artifact_module
from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def _dataset():
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        24
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(24, dtype=np.float64)
    open_price = np.concatenate([close[:1], close[:-1]])
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full(24, 1_000.0),
        funding_rate=np.zeros(24),
        tradable=np.ones(24, dtype=np.bool_),
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
    )
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (contract,),
    )


def test_publish_registers_after_atomic_filesystem_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def capture(registration):
        observed["registration"] = registration
        observed["root_exists"] = (tmp_path / "artifact").is_dir()
        return None

    monkeypatch.setattr(artifact_module, "register_artifact_if_configured", capture)

    published = artifact_module.publish_market_dataset_artifact(
        tmp_path / "artifact", _dataset()
    )

    registration = observed["registration"]
    assert observed["root_exists"] is True
    assert registration.artifact_kind is ArtifactKind.MARKET_DATASET
    assert registration.artifact_digest == published.artifact_digest
    assert registration.dataset_id == _dataset().dataset_id


def test_catalog_failure_does_not_remove_valid_published_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(artifact_module, "register_artifact_if_configured", fail)
    root = tmp_path / "artifact"

    with pytest.raises(RuntimeError, match="catalog unavailable"):
        artifact_module.publish_market_dataset_artifact(root, _dataset())

    assert (root / "manifest.json").is_file()
    assert (root / "arrays.npz").is_file()
