from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def crash_market() -> MarketDataset:
    n_bars = 50
    first = np.linspace(100.0, 124.0, 25)
    second = np.linspace(100.0, 76.0, 25)
    close = np.empty((n_bars, 2), dtype=np.float64)
    close[:25, 0] = first
    close[:25, 1] = second
    close[25:, 0] = first[-1] * 0.50
    close[25:, 1] = second[-1] * 1.50
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="e" * 64,
        symbols=("CRASH", "SQUEEZE"),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_drawdown_stop_liquidates_only_policy_book_before_termination() -> None:
    env = ResidualMarketEnv(
        crash_market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=4,
            decision_every=1,
            initial_capital=1_000.0,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    env.reset(options={"start_idx": 24})

    _, reward, terminated, truncated, info = env.step(np.zeros(2))

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "drawdown_stop"
    assert info["emergency_deleverage"] is True
    assert "hybrid_liquidation" in info
    assert "shadow_liquidation" not in info
    np.testing.assert_allclose(env.hybrid.quantities, np.zeros(2), atol=1e-12)
    assert np.any(np.abs(env.shadow.quantities) > 0.0)
    assert info["drawdown_after"] >= env.config.reward.drawdown_stop
    assert reward == info["reward_total_scaled"]
    assert info["reward_growth_raw"] < 0.0
