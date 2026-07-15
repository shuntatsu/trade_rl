from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import MultiTimeframeMarketDataSource, RawMarketSeries


def _series(
    timestamps: list[str],
    closes: list[float],
    *,
    available_at: list[str] | None = None,
) -> RawMarketSeries:
    timestamp_array = np.asarray(timestamps, dtype="datetime64[ns]")
    close = np.asarray(closes, dtype=np.float64)
    open_price = np.concatenate((close[:1], close[:-1]))
    return RawMarketSeries(
        timestamps=timestamp_array,
        available_at=(
            timestamp_array
            if available_at is None
            else np.asarray(available_at, dtype="datetime64[ns]")
        ),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.arange(1, len(close) + 1, dtype=np.float64) * 100.0,
        funding_rate=np.zeros(len(close), dtype=np.float64),
        funding_available=np.zeros(len(close), dtype=np.bool_),
        tradable=np.ones(len(close), dtype=np.bool_),
    )


class MemoryMultiTimeframeSource:
    def __init__(self, values: dict[tuple[str, str], RawMarketSeries]) -> None:
        self.values = values

    def load(self, symbol: str) -> RawMarketSeries:
        return self.load_timeframe(symbol, "1h")

    def load_timeframe(self, symbol: str, timeframe: str) -> RawMarketSeries:
        return self.values[(symbol, timeframe)]


def _base() -> RawMarketSeries:
    timestamps = [f"2026-01-01T{hour:02d}:00:00" for hour in range(9)]
    return _series(timestamps, [100.0 + hour for hour in range(9)])


def _contract() -> tuple[InstrumentContract, ...]:
    return (
        InstrumentContract(
            symbol="BTCUSDT",
            listed_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )


def _config(*, max_staleness_hours: float = 24.0) -> MarketBuildConfig:
    return MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="4h__ret_1bar",
                kind=FeatureKind.LOG_RETURN,
                timeframe="4h",
                lookback=1,
                max_staleness_hours=max_staleness_hours,
            ),
        ),
    )


def test_feature_spec_binds_native_timeframe_to_identity() -> None:
    hourly = FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN)
    four_hour = FeatureSpec(
        name="ret",
        kind=FeatureKind.LOG_RETURN,
        timeframe="4h",
    )

    assert hourly.resolved_timeframe("1h") == "1h"
    assert four_hour.resolved_timeframe("1h") == "4h"
    assert four_hour.canonical_payload()["timeframe"] == "4h"
    assert hourly.canonical_payload() != four_hour.canonical_payload()

    with pytest.raises(ValueError, match="timeframe"):
        FeatureSpec(name="bad", kind=FeatureKind.LOG_RETURN, timeframe="3h")


def test_multitimeframe_protocol_is_runtime_checkable() -> None:
    source = MemoryMultiTimeframeSource({("BTCUSDT", "1h"): _base()})
    assert isinstance(source, MultiTimeframeMarketDataSource)


def test_four_hour_feature_is_not_visible_before_native_close() -> None:
    source = MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _base(),
            ("BTCUSDT", "4h"): _series(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T04:00:00",
                    "2026-01-01T08:00:00",
                ],
                [100.0, 104.0, 112.0],
            ),
        }
    )

    dataset = MarketDatasetBuilder(_config()).build(source, _contract())

    assert not dataset.feature_available[3, 0, 0]
    assert dataset.feature_available[4, 0, 0]
    np.testing.assert_allclose(dataset.features[4, 0, 0], np.log(104.0 / 100.0))
    np.testing.assert_allclose(dataset.features[7, 0, 0], np.log(104.0 / 100.0))
    np.testing.assert_allclose(dataset.features[8, 0, 0], np.log(112.0 / 104.0))


def test_native_availability_delay_prevents_same_close_leakage() -> None:
    source = MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _base(),
            ("BTCUSDT", "4h"): _series(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T04:00:00",
                    "2026-01-01T08:00:00",
                ],
                [100.0, 104.0, 112.0],
                available_at=[
                    "2026-01-01T00:00:00",
                    "2026-01-01T05:00:00",
                    "2026-01-01T08:00:00",
                ],
            ),
        }
    )

    dataset = MarketDatasetBuilder(_config()).build(source, _contract())

    assert not dataset.feature_available[4, 0, 0]
    assert dataset.feature_available[5, 0, 0]
    np.testing.assert_allclose(dataset.features[5, 0, 0], np.log(104.0 / 100.0))
    assert dataset.feature_staleness_hours[5, 0, 0] == pytest.approx(1.0)


def test_multitimeframe_staleness_expires_on_base_clock() -> None:
    source = MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _base(),
            ("BTCUSDT", "4h"): _series(
                ["2026-01-01T00:00:00", "2026-01-01T04:00:00"],
                [100.0, 104.0],
            ),
        }
    )

    dataset = MarketDatasetBuilder(_config(max_staleness_hours=1.0)).build(
        source,
        _contract(),
    )

    assert dataset.feature_available[4, 0, 0]
    assert dataset.feature_available[5, 0, 0]
    assert not dataset.feature_available[6, 0, 0]
    assert dataset.feature_staleness[6, 0, 0] == pytest.approx(1.0)


def test_latest_closed_fifteen_minute_feature_aligns_to_hourly_decision() -> None:
    source = MemoryMultiTimeframeSource(
        {
            ("BTCUSDT", "1h"): _series(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T01:00:00",
                    "2026-01-01T02:00:00",
                ],
                [100.0, 110.0, 120.0],
            ),
            ("BTCUSDT", "15m"): _series(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:15:00",
                    "2026-01-01T00:30:00",
                    "2026-01-01T00:45:00",
                    "2026-01-01T01:00:00",
                    "2026-01-01T01:15:00",
                    "2026-01-01T01:30:00",
                    "2026-01-01T01:45:00",
                    "2026-01-01T02:00:00",
                ],
                [100.0, 101.0, 103.0, 106.0, 110.0, 111.0, 113.0, 116.0, 120.0],
            ),
        }
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="15m__ret_1bar",
                kind=FeatureKind.LOG_RETURN,
                timeframe="15m",
                lookback=1,
            ),
        ),
    )

    dataset = MarketDatasetBuilder(config).build(source, _contract())

    np.testing.assert_allclose(dataset.features[1, 0, 0], np.log(110.0 / 106.0))
    np.testing.assert_allclose(dataset.features[2, 0, 0], np.log(120.0 / 116.0))
