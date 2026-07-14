from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.market import MarketCalendarKind
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def test_builder_preserves_irregular_session_clock() -> None:
    timestamps = np.array(
        [
            "2026-01-02T15:00:00",
            "2026-01-02T16:00:00",
            "2026-01-05T09:00:00",
            "2026-01-05T10:00:00",
        ],
        dtype="datetime64[ns]",
    )
    close = np.array([100.0, 101.0, 102.0, 103.0])
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 101.0, 102.0]),
        high=close + 1.0,
        low=np.minimum(close, np.array([100.0, 100.0, 101.0, 102.0])) - 1.0,
        close=close,
        volume=np.full(4, 100.0),
        funding_rate=np.zeros(4),
        tradable=np.ones(4, dtype=np.bool_),
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
            calendar_kind=MarketCalendarKind.SESSION,
            session_periods_per_year=1_638,
        )
    ).build(
        InMemoryMarketDataSource({"AAPL": raw}),
        (
            InstrumentContract(
                symbol="AAPL", listed_at=datetime(2020, 1, 1, tzinfo=timezone.utc)
            ),
        ),
    )

    assert dataset.calendar_kind is MarketCalendarKind.SESSION
    np.testing.assert_array_equal(dataset.timestamps, timestamps)
    assert dataset.elapsed_hours(1, 2) == 65.0
    assert dataset.periods_per_year == 1_638
