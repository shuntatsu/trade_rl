from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market(*, bar_hours: int = 1, n_bars: int = 80) -> MarketDataset:
    elapsed = np.arange(n_bars, dtype=np.float64) * bar_hours
    close = np.column_stack(
        [
            np.exp(elapsed * 0.002),
            np.exp(-elapsed * 0.001),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id=("e" if bar_hours == 1 else "f") * 64,
        symbols=("UP", "DOWN"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(1, n_bars + 1) * np.timedelta64(bar_hours, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760 // bar_hours,
    )


def test_environment_resolves_episode_and_decision_hours_to_bars() -> None:
    dataset = market(bar_hours=4, n_bars=80)
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_hours=8.0, base_hours=16.0, slow_hours=24.0)
        ),
        config=ResidualMarketEnvConfig(
            episode_hours=32.0,
            decision_hours=8.0,
            initial_capital=1_000.0,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )

    assert env.episode_bars == 8
    assert env.decision_bars == 2
    env.reset(options={"start_idx": 10})
    env.step(np.zeros(2))
    assert env.current_index == 12


def test_legacy_explicit_bar_config_overrides_hour_defaults() -> None:
    dataset = market()
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=12,
            decision_every=3,
            initial_capital=1_000.0,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )

    assert env.episode_bars == 12
    assert env.decision_bars == 3


def test_end_of_episode_liquidation_charges_cost_and_is_terminal() -> None:
    dataset = market()
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=4,
            decision_every=4,
            initial_capital=1_000.0,
            liquidate_on_end=True,
            execution_cost=ExecutionCostConfig(
                fee_rate=0.001,
                spread_rate=0.0,
                impact_rate=0.0,
                max_participation_rate=1.0,
            ),
        ),
    )
    env.reset(options={"start_idx": 24})

    _, _, terminated, truncated, info = env.step(np.zeros(2))

    assert terminated is True
    assert truncated is False
    np.testing.assert_allclose(env.hybrid.quantities, np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(env.shadow.quantities, np.zeros(2), atol=1e-12)
    assert env.hybrid.total_cost > 0.0
    assert env.shadow.total_cost == pytest.approx(env.hybrid.total_cost)
    assert info["hybrid_liquidation"].fill_count == 2
