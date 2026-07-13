from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.rewards import AbsoluteGrowthRewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def dataset(n_bars: int = 160) -> MarketDataset:
    timestamps = np.datetime64("2026-01-01T01:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    up = np.exp(np.arange(n_bars, dtype=np.float64) * 0.002)
    down = np.exp(-np.arange(n_bars, dtype=np.float64) * 0.001)
    close = np.column_stack([up, down])
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("UP", "DOWN"),
        timestamps=timestamps,
        features=np.zeros((n_bars, 2, 2), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0, dtype=np.float64),
        funding_rate=np.zeros_like(close),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 2), dtype=np.bool_),
        feature_names=("ret", "rsi"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def environment() -> ResidualMarketEnv:
    return ResidualMarketEnv(
        dataset(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=32,
            decision_every=4,
            initial_capital=1_000.0,
            reward=AbsoluteGrowthRewardConfig(),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_zero_action_receives_absolute_baseline_growth() -> None:
    env = environment()
    env.reset(options={"start_idx": 24})

    _, reward, _, _, info = env.step(np.zeros(2))

    assert env.hybrid.portfolio_value == pytest.approx(env.shadow.portfolio_value)
    assert info["reward_growth_raw"] == pytest.approx(
        info["hybrid_execution"].interval_log_return
    )
    assert info["reward_baseline_penalty_delta"] == pytest.approx(0.0)
    assert info["reward_drawdown_penalty_delta"] == pytest.approx(0.0)
    assert reward == pytest.approx(100.0 * info["reward_growth_raw"])


def test_reward_info_exposes_complete_breakdown() -> None:
    env = environment()
    env.reset(options={"start_idx": 24})

    _, reward, _, _, info = env.step(np.array([1.0, 0.0]))

    assert reward == pytest.approx(info["reward_total_scaled"])
    assert info["reward_total_raw"] == pytest.approx(
        info["reward_growth_raw"]
        - info["reward_baseline_penalty_weighted"]
        - info["reward_drawdown_penalty_weighted"]
    )
    assert info["rolling_hybrid_log_growth"] == pytest.approx(
        info["reward_context_after"].rolling_hybrid_log_growth
    )
    assert info["rolling_baseline_log_growth"] == pytest.approx(
        info["reward_context_after"].rolling_shadow_log_growth
    )
    assert info["rolling_growth_gap"] == pytest.approx(
        info["reward_context_after"].rolling_growth_gap
    )
    assert info["portfolio_value_after"] == pytest.approx(env.hybrid.portfolio_value)


def test_drawdown_stop_liquidates_policy_and_terminates_without_fixed_jackpot() -> None:
    env = environment()
    env.reset(options={"start_idx": 24})
    env.step(np.zeros(2))
    env.hybrid.peak_value = env.hybrid.portfolio_value / 0.79

    _, reward, terminated, truncated, info = env.step(np.zeros(2))

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "drawdown_stop"
    assert info["emergency_deleverage"] is True
    assert "hybrid_liquidation" in info
    assert "shadow_liquidation" not in info
    np.testing.assert_allclose(env.hybrid.quantities, np.zeros(2), atol=1e-12)
    assert np.any(np.abs(env.shadow.quantities) > 0.0)
    assert reward == pytest.approx(info["reward_total_scaled"])
    assert abs(reward) < env.config.reward.scale
