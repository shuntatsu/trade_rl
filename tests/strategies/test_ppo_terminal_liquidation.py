from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.test_shared_cash_replay import _market
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def test_terminal_settlement_charges_exit_cost_without_overwriting_action() -> None:
    dataset = _market(np.full((5, 1), 100.0))
    execution_cost = replace(ExecutionCostConfig.zero(), fee_rate=0.001)
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=4,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
    )
    env.reset(seed=7)

    for action in (1, 1):
        _, reward, terminated, truncated, _ = env.step(action)
        assert reward == pytest.approx(0.0)
        assert terminated is False
        assert truncated is False

    _, reward, terminated, truncated, info = env.step(2)

    assert terminated is True
    assert truncated is False
    assert info["intent"] is PositionIntent.LONG
    assert info["terminal_settlement_performed"] is True
    assert info["terminal_settlement_start_index"] == 3
    assert info["terminal_settlement_bars"] == 1
    assert info["interval_cost_amount"] == pytest.approx(0.5)
    assert info["terminal_settlement_cost_amount"] == pytest.approx(0.5)
    assert info["terminal_quantity_before"] == pytest.approx([5.0])
    assert info["terminal_quantity_after"] == pytest.approx([0.0])
    assert info["terminal_settlement_reward"] == pytest.approx(
        math.log(999.0 / 999.5)
    )
    assert reward == pytest.approx(math.log(999.0 / 1_000.0))
    assert env.book.cash == pytest.approx(999.0)
    assert env.book.portfolio_value == pytest.approx(999.0)
    assert env.book.quantities == pytest.approx([0.0])
    assert env.current_intent is PositionIntent.FLAT
