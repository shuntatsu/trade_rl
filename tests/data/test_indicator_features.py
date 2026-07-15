from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.source import RawMarketSeries


def _trend_series(n_bars: int = 96) -> RawMarketSeries:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = np.asarray(
        [
            np.datetime64((start + timedelta(hours=index)).replace(tzinfo=None), "ns")
            for index in range(n_bars)
        ]
    )
    close = np.linspace(100.0, 160.0, n_bars)
    open_price = close - 0.25
    high = close + 1.0
    low = close - 1.0
    volume = 1_000.0 + np.arange(n_bars, dtype=np.float64) * 10.0
    funding = np.zeros(n_bars, dtype=np.float64)
    funding[::8] = 0.0001
    funding_available = np.zeros(n_bars, dtype=np.bool_)
    funding_available[::8] = True
    return RawMarketSeries(
        timestamps=timestamps,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=funding,
        funding_available=funding_available,
        tradable=np.ones(n_bars, dtype=np.bool_),
    )


def _events(
    kind: FeatureKind, *, lookback: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = _trend_series()
    return calculate_feature_events(
        FeatureSpec(name=kind.value, kind=kind, lookback=lookback),
        open_price=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        funding_available=raw.funding_available,
        row_present=np.ones(raw.timestamps.shape, dtype=np.bool_),
        active=np.ones(raw.timestamps.shape, dtype=np.bool_),
    )


def test_indicator_feature_kinds_include_ichimoku_family() -> None:
    assert {
        FeatureKind.ICHIMOKU_TENKAN_DISTANCE,
        FeatureKind.ICHIMOKU_KIJUN_DISTANCE,
        FeatureKind.ICHIMOKU_CLOUD_POSITION,
        FeatureKind.ICHIMOKU_CLOUD_THICKNESS,
    }.issubset(set(FeatureKind))


def test_trend_indicators_are_finite_and_causal() -> None:
    for kind, lookback in (
        (FeatureKind.RSI, 14),
        (FeatureKind.MACD_LINE, 26),
        (FeatureKind.MACD_SIGNAL, 35),
        (FeatureKind.MACD_HISTOGRAM, 35),
        (FeatureKind.ATR_PCT, 14),
        (FeatureKind.ADX, 14),
        (FeatureKind.ICHIMOKU_TENKAN_DISTANCE, 9),
        (FeatureKind.ICHIMOKU_KIJUN_DISTANCE, 26),
        (FeatureKind.ICHIMOKU_CLOUD_POSITION, 52),
        (FeatureKind.ICHIMOKU_CLOUD_THICKNESS, 52),
    ):
        values, valid, source_start = _events(kind, lookback=lookback)
        assert valid[-1]
        assert np.isfinite(values[valid]).all()
        assert np.all(source_start[valid] >= 0)
        assert np.all(source_start[valid] <= np.flatnonzero(valid))

    rsi, rsi_valid, _ = _events(FeatureKind.RSI, lookback=14)
    assert rsi[rsi_valid][-1] > 0.9
    macd, macd_valid, _ = _events(FeatureKind.MACD_LINE, lookback=26)
    assert macd[macd_valid][-1] > 0.0


def test_indicator_prefix_is_invariant_to_future_bars() -> None:
    raw = _trend_series()
    spec = FeatureSpec(
        name="ichimoku_cloud_position",
        kind=FeatureKind.ICHIMOKU_CLOUD_POSITION,
        lookback=52,
    )
    full = calculate_feature_events(
        spec,
        open_price=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        volume=raw.volume,
        funding_rate=raw.funding_rate,
        funding_available=raw.funding_available,
        row_present=np.ones(raw.timestamps.shape, dtype=np.bool_),
        active=np.ones(raw.timestamps.shape, dtype=np.bool_),
    )
    stop = 72
    prefix = calculate_feature_events(
        spec,
        open_price=raw.open[:stop],
        high=raw.high[:stop],
        low=raw.low[:stop],
        close=raw.close[:stop],
        volume=raw.volume[:stop],
        funding_rate=raw.funding_rate[:stop],
        funding_available=raw.funding_available[:stop],
        row_present=np.ones(stop, dtype=np.bool_),
        active=np.ones(stop, dtype=np.bool_),
    )
    np.testing.assert_array_equal(full[1][:stop], prefix[1])
    np.testing.assert_allclose(full[0][:stop], prefix[0], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(full[2][:stop], prefix[2])


def test_feature_spec_identity_binds_indicator_kind() -> None:
    rsi = FeatureSpec(name="x", kind=FeatureKind.RSI, lookback=14)
    atr = FeatureSpec(name="x", kind=FeatureKind.ATR_PCT, lookback=14)
    assert rsi.canonical_payload() != atr.canonical_payload()
    assert InstrumentContract(symbol="BTCUSDT").symbol == "BTCUSDT"
