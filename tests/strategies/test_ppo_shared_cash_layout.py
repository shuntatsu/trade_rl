from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.test_ppo_intent import SequenceStrategy
from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.evaluation.directional import CloseAtEndStrategy
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOSharedCashCoordinator, fit_ppo_strategy


def _risk_config() -> PreTradeRiskConfig:
    return PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=1.0,
        drawdown_stop=1.0,
    )


def test_shared_cash_coordinator_matches_canonical_replay() -> None:
    dataset = pooled_market()
    risk_config = _risk_config()
    strategies = (
        SequenceStrategy(
            (PositionIntent.LONG, PositionIntent.LONG, PositionIntent.FLAT)
        ),
        SequenceStrategy(
            (PositionIntent.LONG, PositionIntent.LONG, PositionIntent.FLAT)
        ),
    )
    replay = run_shared_cash_replay(
        dataset,
        strategies,
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=PreTradeRisk(risk_config),
    )
    coordinator = PPOSharedCashCoordinator(
        dataset,
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk_config=risk_config,
    )

    observations, reset_infos = coordinator.reset(seed=17)

    assert observations.shape == (2, 5)
    assert tuple(info["symbol_index"] for info in reset_infos) == (0, 1)

    rewards: list[float] = []
    target_weights: list[tuple[float, ...]] = []
    for actions in ((2, 2), (2, 2), (1, 1)):
        observations, reward, terminated, infos = coordinator.step(
            np.asarray(actions, dtype=np.int64)
        )
        rewards.append(reward)
        target_weights.append(tuple(float(info["target_weight"]) for info in infos))

    assert terminated is True
    np.testing.assert_allclose(rewards, np.log1p(replay.returns.values))
    np.testing.assert_allclose(
        target_weights,
        [decision.target_weights for decision in replay.decisions],
    )
    np.testing.assert_allclose(coordinator.book.quantities, replay.book.quantities)
    assert coordinator.book.cash == pytest.approx(replay.book.cash)
    assert coordinator.book.portfolio_value == pytest.approx(
        replay.book.portfolio_value
    )


def test_shared_cash_coordinator_constrains_simultaneous_portfolio_proposals() -> None:
    coordinator = PPOSharedCashCoordinator(
        pooled_market(),
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk_config=_risk_config(),
    )
    coordinator.reset(seed=19)

    _, _, _, infos = coordinator.step(np.asarray([2, 2], dtype=np.int64))

    targets = np.asarray(
        [float(info["target_weight"]) for info in infos],
        dtype=np.float64,
    )
    assert np.abs(targets).sum() == pytest.approx(0.5)
    assert all(float(info["portfolio_gross"]) == pytest.approx(0.5) for info in infos)
    assert all(
        float(info["portfolio_team_reward"])
        == pytest.approx(float(infos[0]["portfolio_team_reward"]))
        for info in infos
    )
    assert all("max_gross" in info["risk_reasons"] for info in infos)


def test_shared_cash_terminal_settlement_matches_canonical_endpoint() -> None:
    dataset = pooled_market()
    risk_config = _risk_config()
    replay = run_shared_cash_replay(
        dataset,
        (
            CloseAtEndStrategy(
                SequenceStrategy((PositionIntent.LONG, PositionIntent.LONG)),
                close_index=2,
            ),
            CloseAtEndStrategy(
                SequenceStrategy((PositionIntent.LONG, PositionIntent.LONG)),
                close_index=2,
            ),
        ),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=PreTradeRisk(risk_config),
    )
    coordinator = PPOSharedCashCoordinator(
        dataset,
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk_config=risk_config,
        settle_terminal_position=True,
    )
    coordinator.reset(seed=23)

    _, first_reward, first_terminated, _ = coordinator.step(
        np.asarray([2, 2], dtype=np.int64)
    )
    _, final_reward, final_terminated, final_infos = coordinator.step(
        np.asarray([2, 2], dtype=np.int64)
    )

    expected = np.log1p(replay.returns.values)
    assert first_terminated is False
    assert final_terminated is True
    assert first_reward == pytest.approx(expected[0])
    assert final_reward == pytest.approx(expected[1] + expected[2])
    assert all(info["terminal_settlement_intervals"] == 1 for info in final_infos)
    np.testing.assert_allclose(coordinator.book.quantities, replay.book.quantities)
    assert coordinator.book.portfolio_value == pytest.approx(
        replay.book.portfolio_value
    )


@pytest.mark.parametrize(
    "actions",
    [
        np.asarray([2], dtype=np.int64),
        np.asarray([2, 2, 2], dtype=np.int64),
        np.asarray([0.0, 2.0], dtype=np.float64),
    ],
)
def test_shared_cash_coordinator_rejects_invalid_action_vectors(
    actions: np.ndarray,
) -> None:
    coordinator = PPOSharedCashCoordinator(
        pooled_market(),
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
    )
    coordinator.reset(seed=29)

    with pytest.raises(ValueError, match="action"):
        coordinator.step(actions)


def test_shared_cash_fit_rejects_stochastic_execution_slippage() -> None:
    with pytest.raises(ValueError, match="deterministic execution slippage"):
        fit_ppo_strategy(
            pooled_market(),
            feature_indices=(0,),
            fit_symbol_indices=(0, 1),
            start_index=0,
            stop_index=3,
            gross_budget=0.5,
            total_timesteps=64,
            seed=31,
            execution_cost=ExecutionCostConfig(slippage_std=0.01),
            training_layout="shared_cash",
            rollout_steps_per_env=32,
        )


def test_shared_cash_coordinator_matches_realistic_carry_and_capacity() -> None:
    base = pooled_market()
    dataset = replace(
        base,
        fee_rate=np.full_like(base.close, 0.001),
        spread_rate=np.full_like(base.close, 0.002),
        funding_rate=np.full_like(base.close, 0.001),
        borrow_rate=np.full_like(base.close, 0.1),
        max_participation_rate=np.full_like(base.close, 1e-6),
    )
    strategies = (
        SequenceStrategy(
            (PositionIntent.LONG, PositionIntent.LONG, PositionIntent.FLAT)
        ),
        SequenceStrategy(
            (PositionIntent.SHORT, PositionIntent.SHORT, PositionIntent.FLAT)
        ),
    )
    replay = run_shared_cash_replay(
        dataset,
        strategies,
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
        risk=PreTradeRisk(_risk_config()),
    )
    coordinator = PPOSharedCashCoordinator(
        dataset,
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
        risk_config=_risk_config(),
    )
    coordinator.reset(seed=0)

    rewards: list[float] = []
    first_infos = None
    for actions in ((2, 0), (2, 0), (1, 1)):
        _, reward, _, infos = coordinator.step(np.asarray(actions, dtype=np.int64))
        rewards.append(reward)
        if first_infos is None:
            first_infos = infos

    assert first_infos is not None
    assert float(first_infos[0]["fill_ratio"]) < 1.0
    assert replay.diagnostics.total_cost > 0.0
    assert replay.diagnostics.borrow_cost > 0.0
    np.testing.assert_allclose(rewards, np.log1p(replay.returns.values))
    np.testing.assert_allclose(coordinator.book.quantities, replay.book.quantities)
    assert coordinator.book.cash == pytest.approx(replay.book.cash)
    assert coordinator.book.portfolio_value == pytest.approx(
        replay.book.portfolio_value
    )
    assert coordinator.book.total_cost == pytest.approx(replay.book.total_cost)
    assert coordinator.book.funding_pnl == pytest.approx(replay.book.funding_pnl)
    assert coordinator.book.borrow_cost == pytest.approx(replay.book.borrow_cost)


def test_shared_cash_terminal_settlement_reserves_latency_window() -> None:
    dataset = pooled_market()
    execution_cost = replace(
        DIRECTIONAL_BASE_EXECUTION_COST,
        order_latency_bars=1,
    )
    replay = run_shared_cash_replay(
        dataset,
        (
            CloseAtEndStrategy(
                SequenceStrategy((PositionIntent.LONG,)),
                close_index=1,
            ),
            CloseAtEndStrategy(
                SequenceStrategy((PositionIntent.LONG,)),
                close_index=1,
            ),
        ),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        risk=PreTradeRisk(_risk_config()),
    )
    coordinator = PPOSharedCashCoordinator(
        dataset,
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        risk_config=_risk_config(),
        settle_terminal_position=True,
    )
    coordinator.reset(seed=37)

    _, reward, terminated, infos = coordinator.step(
        np.asarray([2, 2], dtype=np.int64)
    )

    assert terminated is True
    assert coordinator.agent_stop_index == 1
    assert all(info["terminal_settlement_intervals"] == 2 for info in infos)
    assert reward == pytest.approx(float(np.log1p(replay.returns.values).sum()))
    np.testing.assert_allclose(coordinator.book.quantities, replay.book.quantities)
    assert coordinator.book.portfolio_value == pytest.approx(
        replay.book.portfolio_value
    )


def test_shared_cash_slot_order_does_not_change_portfolio_execution() -> None:
    dataset = pooled_market()

    first = PPOSharedCashCoordinator(
        dataset,
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk_config=_risk_config(),
    )
    second = PPOSharedCashCoordinator(
        dataset,
        feature_indices=(0,),
        symbol_indices=(1, 0),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk_config=_risk_config(),
    )
    first.reset(seed=41)
    second.reset(seed=41)

    _, first_reward, _, _ = first.step(np.asarray([2, 0], dtype=np.int64))
    _, second_reward, _, _ = second.step(np.asarray([0, 2], dtype=np.int64))

    assert first_reward == pytest.approx(second_reward)
    np.testing.assert_allclose(first.book.quantities, second.book.quantities)
    assert first.book.cash == pytest.approx(second.book.cash)
    assert first.book.portfolio_value == pytest.approx(second.book.portfolio_value)
