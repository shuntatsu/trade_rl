from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.test_ppo_intent import SequenceStrategy, market
from trade_rl.evaluation.directional import CloseAtEndStrategy
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOTradingEnv
from trade_rl.strategies.rl.ppo_normalization import fit_ppo_feature_normalizer


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
    # RED: PPOTradingEnv does not yet expose executable terminal settlement.
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
    assert final_info["intent"] is PositionIntent.LONG
    assert env.index == 3
    assert env.current_intent is PositionIntent.FLAT
    assert env.desired_quantity == pytest.approx(0.0)
    assert env.book.quantities[0] == pytest.approx(0.0)
    assert env.book.quantities[0] == pytest.approx(replay.book.quantities[0])
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)
    assert env.book.total_cost == pytest.approx(replay.book.total_cost)
    assert final_info["terminal_settlement_intervals"] == 1
    assert abs(float(final_info["terminal_settlement_start_weight"])) > 0.0
    assert final_info["realized_weight"] == pytest.approx(
        final_info["terminal_settlement_start_weight"]
    )
    assert float(final_info["terminal_settlement_final_weight"]) == pytest.approx(0.0)
    assert float(final_info["terminal_settlement_cost_amount"]) > 0.0

    expected = np.log1p(replay.returns.values)
    assert first_reward == pytest.approx(expected[0])
    assert final_reward == pytest.approx(expected[1] + expected[2])

    with pytest.raises(RuntimeError, match="terminated"):
        env.step(1)


def test_terminal_settlement_reserves_latency_window_and_matches_replay() -> None:
    dataset = _flat_cost_market()
    execution_cost = replace(
        DIRECTIONAL_BASE_EXECUTION_COST,
        order_latency_bars=1,
    )
    replay = run_single_symbol_replay(
        dataset,
        CloseAtEndStrategy(
            SequenceStrategy((PositionIntent.LONG,)),
            close_index=1,
        ),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        settle_terminal_position=True,
    )
    env.reset(seed=11)

    _, reward, terminated, truncated, info = env.step(2)

    assert terminated is True
    assert truncated is False
    assert env.agent_stop_index == 1
    assert info["terminal_settlement_intervals"] == 2
    assert env.book.quantities[0] == pytest.approx(replay.book.quantities[0])
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)
    assert env.book.total_cost == pytest.approx(replay.book.total_cost)
    assert reward == pytest.approx(float(np.log1p(replay.returns.values).sum()))


def test_terminal_settlement_matches_partial_fill_residual() -> None:
    base = _flat_cost_market()
    dataset = replace(
        base,
        max_participation_rate=np.full_like(base.close, 1e-6),
    )
    replay = run_single_symbol_replay(
        dataset,
        CloseAtEndStrategy(
            SequenceStrategy((PositionIntent.LONG, PositionIntent.LONG)),
            close_index=2,
        ),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
        settle_terminal_position=True,
    )
    env.reset(seed=13)

    env.step(2)
    _, _, terminated, _, info = env.step(2)

    assert terminated is True
    assert float(info["terminal_settlement_final_weight"]) != 0.0
    assert env.book.quantities[0] == pytest.approx(replay.book.quantities[0])
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)
    assert env.book.total_cost == pytest.approx(replay.book.total_cost)


def test_terminal_settlement_passes_flat_target_through_turnover_risk() -> None:
    dataset = _flat_cost_market()
    risk_config = PreTradeRiskConfig(
        max_gross=1.0,
        max_abs_weight=1.0,
        max_turnover=0.05,
        drawdown_start=1.0,
        drawdown_stop=1.0,
    )
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
        risk=PreTradeRisk(risk_config),
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
        risk_config=risk_config,
        settle_terminal_position=True,
    )
    env.reset(seed=17)

    env.step(2)
    _, _, terminated, _, info = env.step(2)

    assert terminated is True
    assert "max_turnover" in info["terminal_settlement_risk_reasons"]
    assert env.book.quantities[0] == pytest.approx(replay.book.quantities[0])
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)


def test_terminal_settlement_normalizer_uses_only_agent_decision_rows() -> None:
    dataset = _flat_cost_market()
    normalizer = fit_ppo_feature_normalizer(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=2,
    )

    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
        feature_normalizer=normalizer,
        settle_terminal_position=True,
    )

    assert env.agent_stop_index == 2

    wrong_scope = fit_ppo_feature_normalizer(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
    )
    with pytest.raises(ValueError, match="training scope"):
        PPOTradingEnv(
            dataset,
            feature_indices=(0,),
            start_index=0,
            stop_index=3,
            gross_budget=0.1,
            execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
            feature_normalizer=wrong_scope,
            settle_terminal_position=True,
        )


def test_terminal_settlement_rejects_window_without_agent_interval() -> None:
    with pytest.raises(ValueError, match="at least one agent interval"):
        PPOTradingEnv(
            _flat_cost_market(),
            feature_indices=(0,),
            start_index=0,
            stop_index=1,
            gross_budget=0.1,
            execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
            settle_terminal_position=True,
        )


def test_terminal_settlement_remains_opt_in_for_generic_env() -> None:
    dataset = _flat_cost_market()
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env.reset(seed=23)

    info: dict[str, object] = {}
    terminated = False
    for action in (2, 2, 2):
        _, _, terminated, _, info = env.step(action)

    assert terminated is True
    assert env.current_intent is PositionIntent.LONG
    assert env.book.quantities[0] > 0.0
    assert "terminal_settlement_intervals" not in info
