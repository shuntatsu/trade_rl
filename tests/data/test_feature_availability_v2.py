from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract, MarketBuildConfig
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def test_builder_exports_real_feature_age_and_global_availability() -> None:
    n = 7
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(n) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(n, dtype=np.float64)
    funding_available = np.array([True, False, False, True, False, False, False])
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=np.concatenate((close[:1], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(n, 100.0),
        funding_rate=np.array([0.001, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0]),
        funding_available=funding_available,
        tradable=np.ones(n, dtype=np.bool_),
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(
                FeatureSpec(
                    name="funding",
                    kind=FeatureKind.FUNDING_BPS,
                    max_staleness_hours=2.0,
                ),
            ),
        )
    ).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (InstrumentContract(symbol="BTCUSDT", listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),),
    )

    assert dataset.feature_staleness_hours is not None
    assert dataset.feature_staleness is not None
    assert dataset.feature_staleness_hours[1, 0, 0] == 1.0
    assert dataset.feature_staleness_hours[2, 0, 0] == 2.0
    assert dataset.feature_staleness[1, 0, 0] == 0.5
    assert dataset.feature_available[2, 0, 0]
    assert not dataset.feature_available[6, 0, 0]

    assert dataset.global_feature_available is not None
    assert dataset.global_feature_staleness_hours is not None
    assert dataset.global_feature_available.shape == dataset.global_features.shape
