from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def crash_market(*, liquidation_volume: float = 1_000_000.0) -> MarketDataset:
    n_bars = 50
    first = np.linspace(100.0, 124.0, 25)
    second = np.linspace(100.0, 76.0, 25)
    close = np.empty((n_bars, 2), dtype=np.float64)
    close[:25, 0] = first
    close[:25, 1] = second
    close[25:, 0] = first[-1] * 0.50
    close[25:, 1] = second[-1] * 1.50
    open_price = np.vstack([close[0], close[:-1]])
    volume = np.full((n_bars, 2), 1_000_000.0)
    volume[25] = liquidation_volume
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
        volume=volume,
        funding_rate=np.zeros_like(close),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def environment(
    *,
    market: MarketDataset | None = None,
    execution_cost: ExecutionCostConfig | None = None,
    fail_on_incomplete_emergency_liquidation: bool = True,
) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        market or crash_market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=4,
            decision_every=1,
            initial_capital=1_000.0,
            execution_cost=execution_cost or ExecutionCostConfig.zero(),
            fail_on_incomplete_emergency_liquidation=(
                fail_on_incomplete_emergency_liquidation
            ),
        ),
    )


def test_drawdown_stop_liquidates_only_policy_book_before_termination() -> None:
    env = environment()
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
    assert info["drawdown_after"] >= env.reward_tracker.config.drawdown_stop
    assert reward == info["reward_total_scaled"]
    assert info["reward_growth_raw"] < 0.0


def test_drawdown_stop_includes_realized_liquidation_cost_in_growth() -> None:
    env = environment(
        execution_cost=ExecutionCostConfig(
            fee_rate=0.001,
            spread_rate=0.0005,
            impact_rate=0.0,
            max_participation_rate=1.0,
        )
    )
    env.reset(options={"start_idx": 24})

    _, _, terminated, _, info = env.step(np.zeros(2))

    liquidation = info["hybrid_liquidation"]
    expected_growth = (
        info["hybrid_execution"].interval_log_return + liquidation.interval_log_return
    )
    assert terminated is True
    assert liquidation.interval_cost > 0.0
    assert liquidation.interval_log_return < 0.0
    assert info["reward_growth_raw"] == pytest.approx(expected_growth)
    assert info["portfolio_value_after"] == pytest.approx(env.hybrid.portfolio_value)


def test_drawdown_stop_fails_closed_when_emergency_exit_cannot_fill() -> None:
    env = environment(market=crash_market(liquidation_volume=0.0))
    env.reset(options={"start_idx": 24})

    with pytest.raises(RuntimeError, match="hybrid liquidation"):
        env.step(np.zeros(2))


def test_training_drawdown_stop_terminates_when_emergency_exit_is_partial() -> None:
    env = environment(
        market=crash_market(liquidation_volume=0.0),
        fail_on_incomplete_emergency_liquidation=False,
    )
    env.reset(options={"start_idx": 24})

    _, _, terminated, truncated, info = env.step(np.zeros(2))

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "drawdown_stop"
    assert info["liquidation_complete"] is False
    assert np.any(np.abs(env.hybrid.quantities) > 0.0)
