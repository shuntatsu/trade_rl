from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.intent import PPO_OBSERVATION_SCHEMA_V3
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def _flat_price_market() -> MarketDataset:
    prices = np.full((4, 1), 100.0)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.zeros((4, 1, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.full_like(prices, 1_000_000.0),
        funding_rate=np.zeros_like(prices),
        tradable=np.ones((4, 1), dtype=np.bool_),
        feature_available=np.ones((4, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        mark_price=prices.copy(),
    )


def test_ppo_minimum_hold_terminal_close_matches_reference_cash_ledger() -> None:
    initial_cash = 1_000.0
    price = 100.0
    target_fraction = 0.5
    fee_rate = 0.01

    # Independent reference ledger: constant prices, one entry fee, no fill on
    # the locked FLAT request, then one terminal exit fee.
    expected_quantity = target_fraction * initial_cash / price
    entry_notional = expected_quantity * price
    entry_cost = entry_notional * fee_rate
    cash_after_entry = initial_cash - entry_notional - entry_cost
    equity_after_entry = cash_after_entry + expected_quantity * price
    exit_notional = expected_quantity * price
    exit_cost = exit_notional * fee_rate
    final_cash = cash_after_entry + exit_notional - exit_cost
    terminal_exit_turnover = exit_notional / equity_after_entry

    env = PPOTradingEnv(
        _flat_price_market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=target_fraction,
        initial_capital=initial_cash,
        execution_cost=ExecutionCostConfig(
            fee_rate=fee_rate,
            taker_fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
            borrow_rate_multiplier=0.0,
            maintenance_margin_rate=0.0,
        ),
        settle_terminal_position=True,
        minimum_hold_bars=10,
        observation_schema=PPO_OBSERVATION_SCHEMA_V3,
    )
    env.reset(seed=17)

    _, entry_reward, entry_terminated, entry_truncated, entry_info = env.step(2)
    entry_quantity = float(env.book.quantities[0])
    entry_cash = float(env.book.cash)
    entry_equity = float(env.book.portfolio_value)

    _, terminal_reward, terminated, truncated, terminal_info = env.step(1)

    assert entry_info["requested_intent"] is PositionIntent.LONG
    assert entry_info["minimum_hold_suppressed"] is False
    assert entry_info["requested_turnover"] == pytest.approx(
        entry_notional / initial_cash
    )
    assert entry_info["filled_turnover"] == pytest.approx(entry_notional / initial_cash)
    assert entry_info["interval_cost_amount"] == pytest.approx(entry_cost)
    assert entry_quantity == pytest.approx(expected_quantity)
    assert entry_cash == pytest.approx(cash_after_entry)
    assert entry_equity == pytest.approx(equity_after_entry)
    assert entry_reward == pytest.approx(math.log(equity_after_entry / initial_cash))
    assert entry_terminated is False
    assert entry_truncated is False

    assert terminal_info["requested_intent"] is PositionIntent.FLAT
    assert terminal_info["effective_intent"] is PositionIntent.LONG
    assert terminal_info["minimum_hold_suppressed"] is True
    assert terminal_info["target_quantity_override"] == pytest.approx(expected_quantity)
    assert terminal_info["interval_cost_amount"] == pytest.approx(0.0)
    assert terminal_info["requested_turnover"] == pytest.approx(0.0)
    assert terminal_info["filled_turnover"] == pytest.approx(0.0)
    assert terminal_info["terminal_settlement_intervals"] == 1
    assert terminal_info["terminal_settlement_start_weight"] == pytest.approx(
        expected_quantity * price / equity_after_entry
    )
    assert terminal_info["terminal_settlement_requested_turnover"] == pytest.approx(
        terminal_exit_turnover
    )
    assert terminal_info["terminal_settlement_filled_turnover"] == pytest.approx(
        terminal_exit_turnover
    )
    assert terminal_info["terminal_settlement_cost_amount"] == pytest.approx(exit_cost)
    assert terminal_info["terminal_settlement_final_weight"] == pytest.approx(0.0)
    assert terminal_info["terminal_settlement_net_return"] == pytest.approx(
        final_cash / equity_after_entry - 1.0
    )
    assert env.book.quantities[0] == pytest.approx(0.0)
    assert env.book.cash == pytest.approx(final_cash)
    assert env.book.portfolio_value == pytest.approx(final_cash)
    assert terminal_reward == pytest.approx(math.log(final_cash / equity_after_entry))
    assert terminated is True
    assert truncated is False
