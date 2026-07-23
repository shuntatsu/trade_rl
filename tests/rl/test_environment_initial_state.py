from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_initial_state import (
    EnvironmentInitialStateFactory,
    EnvironmentInitialStateRequest,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market() -> MarketDataset:
    n_bars = 40
    close = np.column_stack(
        [
            np.linspace(100.0, 120.0, n_bars),
            np.linspace(80.0, 72.0, n_bars),
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
        contract_multipliers=np.array([1.0, 10.0]),
    )


def config() -> ResidualMarketEnvConfig:
    return ResidualMarketEnvConfig(
        initial_capital=100_000.0,
        episode_bars=8,
        decision_every=2,
        execution_cost=ExecutionCostConfig.zero(),
    )


def test_initial_state_factory_returns_fresh_equivalent_mutable_values() -> None:
    dataset = market()
    action_spec = ActionSpec()
    request = EnvironmentInitialStateRequest(
        dataset=dataset,
        config=config(),
        action_spec=action_spec,
        minimum_start_index=8,
    )

    first = EnvironmentInitialStateFactory.create(request)
    second = EnvironmentInitialStateFactory.create(request)

    assert first.start_index == 8
    assert first.end_index == 9
    assert first.current_index == 8
    assert first.decision_step_index == 0
    assert first.episode_seed == request.config.execution_cost.random_seed
    assert first.episode_hours == request.config.episode_hours
    assert first.initial_state_mode == "cash"
    assert first.pending_hybrid_target is None
    assert first.pending_shadow_target is None
    assert first.has_reset is False

    assert first.hybrid is not first.shadow
    assert first.hybrid is not second.hybrid
    assert first.shadow is not second.shadow
    assert first.hybrid.cash == request.config.initial_capital
    assert first.shadow.cash == request.config.initial_capital
    np.testing.assert_array_equal(first.hybrid.quantities, np.zeros(2))
    np.testing.assert_array_equal(first.shadow.quantities, np.zeros(2))
    np.testing.assert_array_equal(first.hybrid.mark_prices, dataset.close[8])
    np.testing.assert_array_equal(first.shadow.mark_prices, dataset.close[8])
    np.testing.assert_array_equal(first.hybrid.contract_multipliers, [1.0, 10.0])
    np.testing.assert_array_equal(first.shadow.contract_multipliers, [1.0, 10.0])

    assert first.previous_action is not second.previous_action
    assert first.previous_action.dtype == np.float32
    assert first.previous_action.shape == (action_spec.size,)
    np.testing.assert_array_equal(first.previous_action, np.zeros(action_spec.size))

    assert first.position_age is not second.position_age
    assert first.position_age.dtype == np.float64
    assert first.position_age.shape == (dataset.n_symbols,)
    np.testing.assert_array_equal(first.position_age, np.zeros(dataset.n_symbols))

    assert first.hybrid_order_book is not first.shadow_order_book
    assert first.hybrid_order_book is not second.hybrid_order_book
    assert first.shadow_order_book is not second.shadow_order_book
    assert first.hybrid_order_book.active_orders == ()
    assert first.shadow_order_book.active_orders == ()
    assert first.execution_state is not second.execution_state
    assert first.action_diagnostics is not second.action_diagnostics


def test_environment_initial_values_match_factory_contract() -> None:
    dataset = market()
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        config=config(),
    )

    assert env.start_index == env._minimum_start_index
    assert env.end_index == env.start_index + 1
    assert env.current_index == env.start_index
    assert env.hybrid is not env.shadow
    assert env.hybrid.cash == env.config.initial_capital
    assert env.shadow.cash == env.config.initial_capital
    np.testing.assert_array_equal(
        env.hybrid.mark_prices, dataset.close[env.start_index]
    )
    np.testing.assert_array_equal(
        env.shadow.mark_prices, dataset.close[env.start_index]
    )
    assert env._decision_step_index == 0
    assert env._episode_seed == env.config.execution_cost.random_seed
    assert env._episode_hours == env.config.episode_hours
    assert env._initial_state_mode == "cash"
    assert env._previous_action.dtype == np.float32
    assert env._previous_action.shape == (env.action_spec.size,)
    assert env._pending_hybrid_target is None
    assert env._pending_shadow_target is None
    assert env._hybrid_order_book.active_orders == ()
    assert env._shadow_order_book.active_orders == ()
    assert env._position_age.dtype == np.float64
    assert env._position_age.shape == (dataset.n_symbols,)
    assert env._has_reset is False
