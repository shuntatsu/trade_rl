from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def _market() -> MarketDataset:
    n_bars = 5
    shape = (n_bars, 1)
    close = np.full(shape, 100.0)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_slippage_executor_setup() -> None:
    executor = MarketExecutor(
        _market(),
        ExecutionCostConfig(
            slippage_std=0.01,
            tail_slippage_probability=1.0,
            tail_slippage_multiplier=2.0,
            random_seed=7,
        ),
    )
    assert executor.dataset.n_symbols == 1
