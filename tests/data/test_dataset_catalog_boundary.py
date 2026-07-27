from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import (
    MarketDataset,
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
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


def test_dataset_publication_never_consults_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADE_RL_DATABASE_URL", "postgresql://must-not-be-used")
    monkeypatch.setattr(
        "trade_rl.catalog.service.catalog_factory",
        lambda _: pytest.fail("dataset publication consulted the artifact catalog"),
    )

    published = publish_market_dataset_artifact(tmp_path / "dataset", _dataset())

    loaded = load_market_dataset_artifact(published.root)
    assert loaded.dataset_id == _dataset().dataset_id
    assert published.root.is_dir()


def test_data_package_does_not_import_catalog() -> None:
    violations: list[str] = []
    for path in sorted(Path("trade_rl/data").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "trade_rl.catalog" in source:
            violations.append(path.as_posix())
    assert violations == []
