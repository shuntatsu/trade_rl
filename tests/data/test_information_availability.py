from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import (
    CsvMarketDataSource,
    InMemoryMarketDataSource,
    RawMarketSeries,
)


def _timestamps(count: int = 24) -> np.ndarray:
    return np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        count
    ) * np.timedelta64(1, "h")


def _raw(*, delayed_index: int | None = None) -> RawMarketSeries:
    timestamps = _timestamps()
    close = np.exp(np.arange(len(timestamps), dtype=np.float64) * 0.001)
    open_price = np.concatenate([close[:1], close[:-1]])
    available_at = timestamps.copy()
    if delayed_index is not None:
        available_at[delayed_index] += np.timedelta64(1, "h")
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=available_at,
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full(len(timestamps), 1_000.0),
        funding_rate=np.zeros(len(timestamps)),
        tradable=np.ones(len(timestamps), dtype=np.bool_),
    )


def _build(raw: RawMarketSeries):
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


def test_raw_series_defaults_availability_to_event_timestamp() -> None:
    raw = _raw()

    assert raw.available_at is not None
    np.testing.assert_array_equal(raw.available_at, raw.timestamps)


def test_raw_series_rejects_availability_before_event_time() -> None:
    timestamps = _timestamps()
    close = np.ones(len(timestamps), dtype=np.float64)

    with pytest.raises(ValueError, match="available_at"):
        RawMarketSeries(
            timestamps=timestamps,
            available_at=timestamps - np.timedelta64(1, "h"),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=np.ones(len(timestamps)),
            funding_rate=np.zeros(len(timestamps)),
            tradable=np.ones(len(timestamps), dtype=np.bool_),
        )


def test_delayed_market_row_is_not_visible_to_same_time_features() -> None:
    market = _build(_raw(delayed_index=10))

    assert market.information_available is not None
    assert not market.information_available[10, 0]
    assert not market.feature_available[10, 0, 0]
    assert not market.feature_available[11, 0, 0]
    assert market.feature_available[12, 0, 0]


def test_information_availability_changes_dataset_identity() -> None:
    on_time = _build(_raw())
    delayed = _build(_raw(delayed_index=10))

    assert on_time.dataset_id != delayed.dataset_id


def test_csv_source_parses_available_at(tmp_path: Path) -> None:
    path = tmp_path / "BTCUSDT.csv"
    path.write_text(
        "timestamp,available_at,open,high,low,close,volume\n"
        "2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,1,1,1,1,10\n"
        "2026-01-01T01:00:00Z,2026-01-01T02:00:00Z,1,1,1,1,10\n",
        encoding="utf-8",
    )

    raw = CsvMarketDataSource(tmp_path).load("BTCUSDT")

    assert raw.available_at is not None
    assert raw.available_at[1] == np.datetime64("2026-01-01T02:00:00", "ns")
