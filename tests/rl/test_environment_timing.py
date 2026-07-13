from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.observations import observation_layout
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
    high = np.maximum(open_price, close) * 1.001
    low = np.minimum(open_price, close) * 0.999
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("UP", "DOWN"),
        timestamps=timestamps,
        features=np.zeros((n_bars, 2, 2), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0, dtype=np.float64),
        funding_rate=np.zeros_like(close),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 2), dtype=np.bool_),
        feature_names=("ret", "rsi"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def environment(
    *,
    decision_every: int = 4,
    reward_scale: float = 100.0,
    hybrid_insolvency_penalty: float = 1.0,
) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        dataset(),
        trend_strategy=TrendStrategy(
            TrendConfig(
                fast_lookback=4,
                base_lookback=8,
                slow_lookback=16,
            )
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=40,
            decision_every=decision_every,
            reward_scale=reward_scale,
            hybrid_insolvency_penalty=hybrid_insolvency_penalty,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_identity_action_matches_shadow_book_exactly() -> None:
    env = environment()
    env.reset(options={"start_idx": 24})

    for _ in range(5):
        _, reward, terminated, truncated, info = env.step(np.zeros(2))
        assert reward == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(env.hybrid.weights, env.shadow.weights, atol=1e-12)
        assert env.hybrid.portfolio_value == pytest.approx(
            env.shadow.portfolio_value,
            abs=1e-12,
        )
        assert info["bars_advanced"] == 4
        assert info["hybrid_insolvent"] is False
        assert info["shadow_insolvent"] is False
        assert info["rollout_valid"] is True
        if terminated or truncated:
            break

    assert env.shadow.n_trades > 0
    assert env.hybrid.n_trades == env.shadow.n_trades


def test_one_action_receives_one_interval_reward() -> None:
    env = environment(decision_every=8)
    env.reset(options={"start_idx": 24})

    _, reward, _, _, info = env.step(np.array([1.0, 0.0]))

    assert env.current_index == 32
    assert info["bars_advanced"] == 8
    assert info["decision_step_index"] == 1
    assert reward == pytest.approx(100.0 * info["excess_log_return"])


def test_observation_exposes_shadow_relative_state_with_stable_shape() -> None:
    market = dataset()
    env = environment()
    observation, _ = env.reset(options={"start_idx": 24})
    layout = observation_layout(market)

    assert layout.per_symbol_width == market.n_features + 6
    assert layout.global_width == len(market.global_feature_names) + 7
    assert env.action_space.shape == (2,)
    assert observation.shape == env.observation_space.shape == (layout.size,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    np.testing.assert_allclose(observation[-4:], np.zeros(4), atol=1e-12)

    env.hybrid.weights = np.array([0.4, -0.2])
    env.shadow.weights = np.array([0.1, -0.1])
    paired = env._observation()
    symbol_rows = paired[: market.n_symbols * layout.per_symbol_width].reshape(
        market.n_symbols,
        layout.per_symbol_width,
    )
    np.testing.assert_allclose(symbol_rows[:, -1], np.array([0.3, -0.1]))


def test_hybrid_only_insolvency_preserves_excess_and_applies_hybrid_penalty() -> None:
    env = environment(reward_scale=10.0, hybrid_insolvency_penalty=1.25)
    env.reset(options={"start_idx": 24})
    threshold = env.config.initial_capital * env.config.minimum_equity_fraction
    env.hybrid.portfolio_value = threshold / 10.0

    _, reward, terminated, _, info = env.step(np.zeros(2))

    assert terminated is True
    assert info["hybrid_insolvent"] is True
    assert info["shadow_insolvent"] is False
    assert info["rollout_valid"] is True
    assert reward == pytest.approx(10.0 * info["excess_log_return"] - 12.5)


def test_shadow_only_insolvency_terminates_without_hybrid_penalty() -> None:
    env = environment(reward_scale=10.0, hybrid_insolvency_penalty=1.25)
    env.reset(options={"start_idx": 24})
    threshold = env.config.initial_capital * env.config.minimum_equity_fraction
    env.shadow.portfolio_value = threshold / 10.0

    _, reward, terminated, _, info = env.step(np.zeros(2))

    assert terminated is True
    assert info["hybrid_insolvent"] is False
    assert info["shadow_insolvent"] is True
    assert info["rollout_valid"] is False
    assert reward == pytest.approx(10.0 * info["excess_log_return"])


def test_terminal_info_uses_base_bar_return_identity() -> None:
    env = ResidualMarketEnv(
        dataset(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=8,
            decision_every=4,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    env.reset(options={"start_idx": 24})

    info: dict[str, object] = {}
    while True:
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        if terminated or truncated:
            break

    hybrid = info["hybrid_metrics"]
    shadow = info["shadow_metrics"]
    assert hybrid.return_kind.value == "base_bar"
    assert shadow.return_kind.value == "base_bar"
    assert hybrid.n_periods == 8
    assert shadow.n_periods == 8
