from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.rewards import AbsoluteGrowthRewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market() -> MarketDataset:
    n_bars = 40
    close = np.column_stack(
        [
            np.linspace(100.0, 120.0, n_bars),
            np.linspace(100.0, 90.0, n_bars),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 1_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def environment(
    initial_capital: float,
    *,
    reward: AbsoluteGrowthRewardConfig | None = None,
) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=initial_capital,
            episode_bars=8,
            decision_every=2,
            reward=reward or AbsoluteGrowthRewardConfig(),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_environment_config_requires_explicit_initial_capital() -> None:
    with pytest.raises(ValueError, match="explicitly configured"):
        ResidualMarketEnvConfig()


def test_environment_identity_changes_with_aum() -> None:
    small = environment(100_000.0)
    large = environment(1_000_000.0)

    assert small.initial_capital == pytest.approx(100_000.0)
    assert large.initial_capital == pytest.approx(1_000_000.0)
    assert small.environment_digest != large.environment_digest


def test_environment_identity_changes_with_reward_configuration() -> None:
    default = environment(100_000.0)
    stronger_drawdown = environment(
        100_000.0,
        reward=AbsoluteGrowthRewardConfig(drawdown_penalty_weight=0.10),
    )

    assert default.environment_digest != stronger_drawdown.environment_digest
