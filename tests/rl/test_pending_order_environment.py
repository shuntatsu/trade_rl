from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionMode, ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _dataset() -> MarketDataset:
    n = 40
    close = np.full((n, 1), 100.0)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.ones_like(close),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _env() -> ResidualMarketEnv:
    return ResidualMarketEnv(
        _dataset(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=6)
        ),
        action_spec=ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=1,
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=1_000.0,
            episode_bars=8,
            decision_every=1,
            execution_cost=replace(
                ExecutionCostConfig.zero(),
                max_participation_rate=0.1,
                partial_fill_carry=True,
                path_mode="conservative",
            ),
        ),
    )


def test_environment_carries_pending_order_state_into_next_observation() -> None:
    env = _env()
    env.reset(options={"start_idx": 10, "initial_state_mode": "cash"})
    env.step(np.array([1.0], dtype=np.float32))

    snapshot = env.observation_snapshot()
    assert env.hybrid_order_book.active_orders
    assert snapshot.pending_order_remaining[0] > 0.0
    assert snapshot.pending_order_status[0] > 0.0
    assert snapshot.execution_policy_digest == env.execution_policy_digest
    assert snapshot.pending_order_age_bars[0] >= 1.0


def test_latest_target_reconciles_instead_of_duplicate_resubmission() -> None:
    env = _env()
    env.reset(options={"start_idx": 10, "initial_state_mode": "cash"})
    env.step(np.array([1.0], dtype=np.float32))
    first_ids = {order.order_id for order in env.hybrid_order_book.active_orders}

    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    active = env.hybrid_order_book.active_orders
    assert len(active) == 1
    assert {order.order_id for order in active} == first_ids
    assert info["hybrid_execution"].requested_notional <= 1_000.0 + 1e-9


def test_reset_clears_persistent_order_books() -> None:
    env = _env()
    env.reset(options={"start_idx": 10, "initial_state_mode": "cash"})
    env.step(np.array([1.0], dtype=np.float32))
    assert env.hybrid_order_book.active_orders

    env.reset(options={"start_idx": 10, "initial_state_mode": "cash"})
    assert env.hybrid_order_book.active_orders == ()
    assert env.shadow_order_book.active_orders == ()
    snapshot = env.observation_snapshot()
    np.testing.assert_array_equal(snapshot.pending_order_remaining, np.zeros(1))
