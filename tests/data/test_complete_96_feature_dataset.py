from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import InstrumentContract, MarketBuildConfig
from trade_rl.data.source import RawMarketSeries
from trade_rl.integrations.binance import binance_multitimeframe_feature_specs


def _series(timeframe_minutes: int, bars: int) -> RawMarketSeries:
    step = np.timedelta64(timeframe_minutes, "m")
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(bars) * step
    phase = np.arange(bars, dtype=np.float64)
    close = 100.0 * np.exp(0.0002 * phase + 0.01 * np.sin(phase / 17.0))
    open_price = np.concatenate((close[:1], close[:-1]))
    spread = 0.0025 * close
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=timestamps,
        open=open_price,
        high=np.maximum(open_price, close) + spread,
        low=np.minimum(open_price, close) - spread,
        close=close,
        volume=1_000_000.0 + 50_000.0 * (1.0 + np.sin(phase / 11.0)),
        funding_rate=0.0001 * np.sin(phase / 31.0),
        funding_available=np.ones(bars, dtype=np.bool_),
        tradable=np.ones(bars, dtype=np.bool_),
    )


class CompleteSource:
    def __init__(self) -> None:
        self.series = {
            "15m": _series(15, 60 * 24 * 4 + 1),
            "1h": _series(60, 60 * 24 + 1),
            "4h": _series(240, 60 * 6 + 1),
            "1d": _series(1_440, 61),
        }

    def load(self, symbol: str) -> RawMarketSeries:
        assert symbol == "BTCUSDT"
        return self.series["1h"]

    def load_timeframe(self, symbol: str, timeframe: str) -> RawMarketSeries:
        assert symbol == "BTCUSDT"
        return self.series[timeframe]


def test_complete_preset_builds_96_causal_features() -> None:
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="1h",
        feature_timeframes=("15m", "4h", "1d"),
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(base_timeframe="1h", features=specs)
    ).build(
        CompleteSource(),
        (
            InstrumentContract(
                symbol="BTCUSDT",
                listed_at=datetime(2025, 1, 1, tzinfo=UTC),
            ),
        ),
    )

    assert dataset.n_features == 96
    assert dataset.features.shape == (1_441, 1, 96)
    assert dataset.feature_available[-1, 0].all()
    assert np.isfinite(dataset.features).all()
    assert np.isfinite(dataset.feature_staleness).all()
    assert dataset.feature_names[20].endswith("ichimoku_tenkan_distance_9bar")
    assert dataset.feature_names[-1].endswith("ichimoku_cloud_thickness_9_26_52")
