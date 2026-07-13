from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.trend import TrendConfig, TrendMode, TrendStrategy


def market(n_symbols: int) -> MarketDataset:
    n = 40
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(n) * np.timedelta64(
        1, "h"
    )
    paths = [
        100.0 * np.exp(np.arange(n) * (0.002 - i * 0.001)) for i in range(n_symbols)
    ]
    close = np.column_stack(paths)
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=tuple(f"S{i}" for i in range(n_symbols)),
        timestamps=timestamps,
        features=np.zeros((n, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n, n_symbols, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_auto_mode_is_nonzero_for_one_symbol() -> None:
    strategy = TrendStrategy(
        TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
    )
    targets = strategy.targets(market(1), 24)
    assert targets.base[0] > 0.0
    assert np.abs(targets.base).sum() < 1.0


def test_auto_mode_preserves_cross_sectional_behavior_for_multiple_symbols() -> None:
    strategy = TrendStrategy(
        TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
    )
    targets = strategy.targets(market(2), 24)
    assert targets.base.sum() == 0.0
    assert np.abs(targets.base).sum() == 1.0


def test_long_only_never_emits_short_weights() -> None:
    strategy = TrendStrategy(
        TrendConfig(
            fast_lookback=4, base_lookback=8, slow_lookback=16, mode=TrendMode.LONG_ONLY
        )
    )
    assert np.all(strategy.targets(market(2), 24).base >= 0.0)
