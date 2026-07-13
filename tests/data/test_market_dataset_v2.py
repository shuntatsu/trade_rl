from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset


def dataset_kwargs(n_bars: int = 8) -> dict[str, object]:
    timestamps = np.datetime64("2026-01-01T01:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.column_stack(
        [
            100.0 + np.arange(n_bars, dtype=np.float64),
            200.0 + 2.0 * np.arange(n_bars, dtype=np.float64),
        ]
    )
    open_price = close - 0.25
    high = np.maximum(open_price, close) + 0.5
    low = np.minimum(open_price, close) - 0.5
    return {
        "dataset_id": "a" * 64,
        "symbols": ("BTCUSDT", "ETHUSDT"),
        "timestamps": timestamps,
        "features": np.zeros((n_bars, 2, 2), dtype=np.float32),
        "global_features": np.zeros((n_bars, 1), dtype=np.float32),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full((n_bars, 2), 1_000.0, dtype=np.float64),
        "funding_rate": np.zeros_like(close),
        "tradable": np.ones((n_bars, 2), dtype=np.bool_),
        "feature_available": np.ones((n_bars, 2, 2), dtype=np.bool_),
        "feature_names": ("ret", "rsi"),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
    }


def test_regular_hourly_dataset_exposes_real_time_conversion() -> None:
    market = MarketDataset(**dataset_kwargs())

    assert market.bar_hours == pytest.approx(1.0)
    assert market.bars_for_hours(4.0) == 4
    with pytest.raises(ValueError, match="integral number of bars"):
        market.bars_for_hours(1.5)


def test_irregular_timestamps_are_rejected() -> None:
    kwargs = dataset_kwargs()
    timestamps = np.asarray(kwargs["timestamps"]).copy()
    timestamps[4:] += np.timedelta64(1, "h")
    kwargs["timestamps"] = timestamps

    with pytest.raises(ValueError, match="regular"):
        MarketDataset(**kwargs)


def test_periods_per_year_must_match_timestamp_cadence() -> None:
    kwargs = dataset_kwargs()
    kwargs["periods_per_year"] = 2_190

    with pytest.raises(ValueError, match="periods_per_year"):
        MarketDataset(**kwargs)


def test_ohlc_invariants_are_enforced() -> None:
    kwargs = dataset_kwargs()
    high = np.asarray(kwargs["high"]).copy()
    high[2, 0] = np.asarray(kwargs["close"])[2, 0] - 1.0
    kwargs["high"] = high

    with pytest.raises(ValueError, match="OHLC"):
        MarketDataset(**kwargs)


def test_availability_masks_must_match_market_shapes() -> None:
    kwargs = dataset_kwargs()
    kwargs["tradable"] = np.ones((8, 1), dtype=np.bool_)
    with pytest.raises(ValueError, match="tradable"):
        MarketDataset(**kwargs)

    kwargs = dataset_kwargs()
    kwargs["feature_available"] = np.ones((8, 2, 1), dtype=np.bool_)
    with pytest.raises(ValueError, match="feature_available"):
        MarketDataset(**kwargs)
