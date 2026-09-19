from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.test_ppo_intent import SequenceStrategy, market
from trade_rl.evaluation.directional import CloseAtEndStrategy
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def _flat_cost_market():
    base = market()
    flat = np.full_like(base.close, 100.0)
    return replace(
        base,
        open=flat.copy(),
        high=flat.copy(),
        low=flat.copy(),
        close=flat.copy(),
        fee_rate=np.full_like(base.close, 0.001),
        spread_rate=np.full_like(base.close, 0.002),
    )


def test_terminal_settlement_matches_directional_close_at_end_economics() -> None:
    dataset = _flat_cost_market()
    replay = run_single_symbol_replay(
        dataset,
        CloseAtEndStrategy(
            SequenceStrategy((PositionIntent.LONG, PositionIntent.LONG)),
            close_index=2,
        ),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
        settle_terminal_position=True,
    )
    env.reset(seed=7)

    _, first_reward, first_terminated, _, _ = env.step(2)
    _, final_reward, final_terminated, _, final_info = env.step(2)

    assert first_terminated is False
    assert final_terminated is True
    assert env.index == 3
    assert env.current_intent is PositionIntent.FLAT
    assert env.desired_quantity == pytest.approx(0.0)
    assert env.book.quantities[0] == pytest.approx(0.0)
    assert env.book.quantities[0] == pytest.approx(replay.book.quantities[0])
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)
    assert env.book.total_cost == pytest.approx(replay.book.total_cost)
    assert final_info["terminal_settlement_intervals"] == 1
    assert float(final_info["terminal_settlement_cost_amount"]) > 0.0

    expected = np.log1p(replay.returns.values)
    assert first_reward == pytest.approx(expected[0])
    assert final_reward == pytest.approx(expected[1] + expected[2])

    with pytest.raises(RuntimeError, match="terminated"):
        env.step(1)
