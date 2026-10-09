"""Independent native event oracles: sub-hour cross asset windows must not subsample."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import RawMarketSeries


class _Source:
    def __init__(self, values: dict[tuple[str, str], RawMarketSeries]) -> None:
        self.values = values

    def load(self, symbol: str) -> RawMarketSeries:
        return self.load_timeframe(symbol, "1h")

    def load_timeframe(self, symbol: str, timeframe: str) -> RawMarketSeries:
        return self.values[symbol, timeframe]


def _bars(start: str, minutes: int, closes: np.ndarray) -> RawMarketSeries:
    times = np.datetime64(start, "ns") + np.arange(len(closes)) * np.timedelta64(
        minutes, "m"
    )
    previous = np.concatenate((closes[:1], closes[:-1]))
    return RawMarketSeries(
        timestamps=times,
        available_at=times,
        open=previous,
        high=np.maximum(previous, closes),
        low=np.minimum(previous, closes),
        close=closes,
        volume=np.ones(len(closes), dtype=np.float64) * 100_000.0,
        funding_rate=np.zeros(len(closes)),
        funding_available=np.zeros(len(closes), dtype=np.bool_),
        tradable=np.ones(len(closes), dtype=np.bool_),
    )


def _source() -> _Source:
    steps = np.arange(32, dtype=np.float64)
    btc_returns = 0.022 * np.sin(steps * 0.69) + 0.008 * np.cos(steps * 0.31)
    eth_returns = 0.017 * np.cos(steps * 0.53) - 0.007 * np.sin(steps * 0.41)
    btc_close = 100.0 * np.exp(np.r_[0.0, np.cumsum(btc_returns)])
    eth_close = 90.0 * np.exp(np.r_[0.0, np.cumsum(eth_returns)])
    return _Source(
        {
            ("BTCUSDT", "15m"): _bars("2026-01-01T00:00", 15, btc_close),
            ("ETHUSDT", "15m"): _bars("2026-01-01T00:00", 15, eth_close),
            ("BTCUSDT", "1h"): _bars("2026-01-01T00:00", 60, btc_close[::4]),
            ("ETHUSDT", "1h"): _bars("2026-01-01T00:00", 60, eth_close[::4]),
        }
    )


def _dataset(source: _Source) -> object:
    specs = (
        FeatureSpec(
            name="15m__log_return_1bar",
            kind=FeatureKind.LOG_RETURN,
            timeframe="15m",
            lookback=1,
        ),
        FeatureSpec(
            name="15m__rolling_correlation_to_btc_4bar",
            kind=FeatureKind.ROLLING_CORRELATION_TO_BTC,
            timeframe="15m",
            lookback=4,
            min_periods=4,
        ),
        FeatureSpec(
            name="15m__rolling_beta_to_btc_4bar",
            kind=FeatureKind.ROLLING_BETA_TO_BTC,
            timeframe="15m",
            lookback=4,
            min_periods=4,
        ),
    )
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=specs,
            cross_asset_reference_symbol="BTCUSDT",
        )
    ).build(
        source,
        (
            InstrumentContract("BTCUSDT", listed_at=datetime(2025, 1, 1, tzinfo=UTC)),
            InstrumentContract("ETHUSDT", listed_at=datetime(2025, 1, 1, tzinfo=UTC)),
        ),
    )


def test_15m_rolling_correlation_and_beta_use_four_consecutive_native_bars() -> None:
    dataset = _dataset(_source())
    btc = _source().values["BTCUSDT", "15m"].close
    eth = _source().values["ETHUSDT", "15m"].close
    br = np.diff(np.log(btc))[-4:]
    er = np.diff(np.log(eth))[-4:]
    expected_corr = np.corrcoef(br, er)[0, 1]
    expected_beta = np.cov(br, er, ddof=0)[0, 1] / np.var(br)
    actual_corr = dataset.features[8, 1, 1]
    actual_beta = dataset.features[8, 1, 2]
    assert dataset.feature_available[8, 1, 1:3].all()
    assert actual_corr == pytest.approx(expected_corr, abs=1e-9)
    assert actual_beta == pytest.approx(expected_beta, abs=1e-9)
    # An hourly sample of one 15m return per hour must NOT equal a native
    # four-consecutive-bar rolling statistic.
    hourly_sample = np.corrcoef(
        np.diff(np.log(btc))[3::4][-4:],
        np.diff(np.log(eth))[3::4][-4:],
    )[0, 1]
    assert abs(expected_corr - hourly_sample) > 0.01


def test_native_cross_asset_future_modification_preserves_earlier_decisions() -> None:
    original_source = _source()
    original = _dataset(original_source)
    raw = original_source.values["BTCUSDT", "15m"]
    altered_close = raw.close.copy()
    altered_close[22:] *= 1.21
    mutated = dict(original_source.values)
    mutated["BTCUSDT", "15m"] = replace(
        raw,
        close=altered_close,
        high=np.maximum(raw.high, altered_close),
        low=np.minimum(raw.low, altered_close),
    )
    changed = _dataset(_Source(mutated))
    np.testing.assert_array_equal(original.features[:5], changed.features[:5])
    np.testing.assert_array_equal(
        original.feature_available[:5], changed.feature_available[:5]
    )


def test_delayed_native_publication_cannot_backdate_cross_asset_features() -> None:
    source = _source()
    raw = source.values["BTCUSDT", "15m"]
    later = raw.available_at.copy()
    later[16] = np.datetime64("2026-01-01T05:00", "ns")
    modified = dict(source.values)
    modified["BTCUSDT", "15m"] = replace(raw, available_at=later)
    normal = _dataset(source)
    delayed = _dataset(_Source(modified))
    np.testing.assert_array_equal(normal.features[:4], delayed.features[:4])
    # At hour four the native event ending at hour four has NOT been observed.
    assert delayed.feature_staleness_hours[4, 1, 1] > 0.0
    assert normal.feature_staleness_hours[4, 1, 1] == pytest.approx(0.0)
