from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market() -> MarketDataset:
    n_bars = 24
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.column_stack(
        [
            np.exp(np.arange(n_bars) * 0.01),
            np.exp(np.arange(n_bars) * -0.005),
            np.exp(np.arange(n_bars) * 0.02),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    active = np.ones((n_bars, 3), dtype=np.bool_)
    active[:8, 2] = False
    active[20:, 2] = False
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("A", "B", "NEW"),
        timestamps=timestamps,
        features=np.zeros((n_bars, 3, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 3), 1_000.0),
        funding_rate=np.zeros((n_bars, 3)),
        tradable=active.copy(),
        feature_available=active[:, :, None],
        feature_names=("dummy",),
        global_feature_names=("dummy_global",),
        periods_per_year=8_760,
        symbol_active=active,
    )


def test_trend_excludes_symbol_without_full_point_in_time_history() -> None:
    dataset = market()
    strategy = TrendStrategy(
        TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=6)
    )

    before_full_history = strategy.targets(dataset, 10)
    after_full_history = strategy.targets(dataset, 14)
    after_delisting = strategy.targets(dataset, 21)

    assert before_full_history.base[2] == 0.0
    assert after_full_history.base[2] != 0.0
    assert after_delisting.base[2] == 0.0
