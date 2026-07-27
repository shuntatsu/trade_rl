from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trade_rl.catalog.contracts import ArtifactRecord, ArtifactRegistration
from trade_rl.data import MarketDataset, load_market_dataset_artifact, publish_market_dataset_artifact
from trade_rl.workflows.dataset_catalog_reconciliation import (
    reconcile_market_dataset_catalog,
)


def _dataset() -> MarketDataset:
    n_bars = 8
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = (100.0 + np.arange(n_bars, dtype=np.float64))[:, None]
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.arange(n_bars, dtype=np.float32)[:, None, None],
        global_features=np.ones((n_bars, 1), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        tick_size=np.full((n_bars, 1), 0.1),
        lot_size=np.full((n_bars, 1), 0.001),
        minimum_notional=np.full((n_bars, 1), 5.0),
    ).with_content_identity()


class _RetryCatalog:
    def __init__(self) -> None:
        self.attempts: list[ArtifactRegistration] = []

    def register(self, registration: ArtifactRegistration) -> ArtifactRecord:
        self.attempts.append(registration)
        if len(self.attempts) == 1:
            raise ConnectionError("catalog unavailable")
        now = datetime(2026, 1, 2, tzinfo=UTC)
        return ArtifactRecord(registration=registration, created_at=now, last_seen_at=now)


def test_reconciliation_retries_without_republishing_dataset(tmp_path: Path) -> None:
    published = publish_market_dataset_artifact(tmp_path / "dataset", _dataset())
    catalog = _RetryCatalog()

    with pytest.raises(ConnectionError, match="catalog unavailable"):
        reconcile_market_dataset_catalog(published.root, catalog)

    loaded_after_failure = load_market_dataset_artifact(published.root)
    assert loaded_after_failure.dataset_id == _dataset().dataset_id

    record = reconcile_market_dataset_catalog(published.root, catalog)

    assert len(catalog.attempts) == 2
    first, second = catalog.attempts
    assert first == second
    assert record.registration.artifact_digest == published.artifact_digest
    assert record.registration.dataset_id == _dataset().dataset_id
    assert record.registration.location == str(published.root.resolve())
    assert record.registration.metadata["n_bars"] == 8
    assert record.registration.metadata["symbols"] == ("BTCUSDT",)
