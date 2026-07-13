from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market(*, bar_hours: int, n_bars: int) -> MarketDataset:
    elapsed_hours = np.arange(n_bars, dtype=np.float64) * bar_hours
    close = np.column_stack(
        [
            np.exp(elapsed_hours * 0.002),
            np.exp(-elapsed_hours * 0.001),
            np.exp(elapsed_hours * 0.0005),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id=("a" if bar_hours == 1 else "b") * 64,
        symbols=("UP", "DOWN", "SLOW"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(1, n_bars + 1) * np.timedelta64(bar_hours, "h"),
        features=np.zeros((n_bars, 3, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 3), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 3)),
        tradable=np.ones((n_bars, 3), dtype=np.bool_),
        feature_available=np.ones((n_bars, 3, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760 // bar_hours,
    )


def test_time_lookbacks_resolve_to_equal_physical_horizons() -> None:
    hourly = market(bar_hours=1, n_bars=100)
    four_hourly = market(bar_hours=4, n_bars=30)
    strategy = TrendStrategy(
        TrendConfig(fast_hours=24.0, base_hours=48.0, slow_hours=96.0)
    )

    assert strategy.lookbacks(hourly) == (24, 48, 96)
    assert strategy.lookbacks(four_hourly) == (6, 12, 24)
    assert strategy.minimum_history_for(hourly) == 96
    assert strategy.minimum_history_for(four_hourly) == 24

    hourly_target = strategy.targets(hourly, 96)
    four_hour_target = strategy.targets(four_hourly, 24)
    np.testing.assert_allclose(hourly_target.base, four_hour_target.base, atol=1e-12)
    np.testing.assert_allclose(hourly_target.slow, four_hour_target.slow, atol=1e-12)


def test_legacy_bar_lookbacks_remain_explicitly_supported() -> None:
    strategy = TrendStrategy(
        TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
    )
    assert strategy.lookbacks(market(bar_hours=4, n_bars=30)) == (4, 8, 16)
