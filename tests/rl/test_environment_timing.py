from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def dataset(n_bars: int = 180) -> MarketDataset:
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(n_bars) * np.timedelta64(
        1, "h"
    )
    close = np.column_stack(
        [
            100.0 * np.exp(np.arange(n_bars) * 0.002),
            100.0 * np.exp(-np.arange(n_bars) * 0.001),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("UP", "DOWN"),
        timestamps=timestamps,
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.column_stack([np.sin(np.arange(n_bars) / 10.0)]).astype(
            np.float32
        ),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def environment(**config_overrides: object) -> ResidualMarketEnv:
    config_values: dict[str, object] = {
        "episode_bars": 24,
        "decision_every": 4,
        "initial_capital": 1_000.0,
        "execution_cost": ExecutionCostConfig.zero(),
    }
    config_values.update(config_overrides)
    return ResidualMarketEnv(
        dataset(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(**config_values),
    )


def test_dynamic_action_space_has_no_alpha_dead_dimension() -> None:
    env = environment()
    assert env.action_space.shape == (3,)
    assert env.action_names == ("fast_tilt", "slow_tilt", "risk_tilt")


def test_zero_action_preserves_exact_shadow_book_and_zero_excess() -> None:
    env = environment()
    env.reset(seed=3, options={"start_idx": 24})
    _, _, _, _, info = env.step(np.zeros(env.action_spec.size))
    np.testing.assert_allclose(env.hybrid.quantities, env.shadow.quantities)
    assert info["excess_log_return"] == pytest.approx(0.0)


def test_episode_random_seed_changes_across_resets_but_is_reproducible() -> None:
    env = environment(
        execution_cost=ExecutionCostConfig(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
            slippage_std=0.001,
            random_seed=7,
        )
    )
    _, first = env.reset(seed=11, options={"start_idx": 24})
    _, second = env.reset(options={"start_idx": 24})
    assert first["episode_seed"] != second["episode_seed"]
    other = environment(execution_cost=env.config.execution_cost)
    _, repeated = other.reset(seed=11, options={"start_idx": 24})
    assert repeated["episode_seed"] == first["episode_seed"]


def test_economic_failure_returns_terminal_transition_not_exception() -> None:
    env = environment(
        execution_cost=ExecutionCostConfig(
            fee_rate=2.0, spread_rate=0.0, impact_rate=0.0, max_participation_rate=1.0
        )
    )
    env.reset(options={"start_idx": 24})
    _, reward, terminated, _, info = env.step(np.zeros(env.action_spec.size))
    assert terminated is True
    assert np.isfinite(reward)
    assert info["termination_reason"] in {
        "execution_cost_exhaustion",
        "insolvency",
        "minimum_equity",
    }


def test_episode_curriculum_and_non_cash_initial_states_are_exposed() -> None:
    env = ResidualMarketEnv(
        dataset(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(
            episode_hour_choices=(24.0, 48.0),
            decision_hours=4.0,
            initial_capital=1_000.0,
            initial_state_modes=("baseline", "stress", "partial_fill"),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    _, info = env.reset(seed=5)
    assert info["episode_hours"] in {24.0, 48.0}
    assert info["initial_state_mode"] in {"baseline", "stress", "partial_fill"}


def test_hard_drawdown_stop_overrides_turnover_inside_environment() -> None:
    env = ResidualMarketEnv(
        dataset(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        pre_trade_risk=PreTradeRisk(
            PreTradeRiskConfig(max_turnover=0.0, drawdown_start=0.1, drawdown_stop=0.2)
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=8,
            decision_every=1,
            initial_capital=1_000.0,
            initial_state_modes=("stress",),
            stress_drawdown_fraction=0.25,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    env.reset(options={"start_idx": 24, "initial_state_mode": "stress"})
    _, _, _, _, info = env.step(np.zeros(env.action_spec.size))
    assert info["hybrid_risk"].risk_scale == 0.0
    np.testing.assert_array_equal(info["hybrid_risk"].weights, np.zeros(2))
