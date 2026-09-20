from __future__ import annotations

import numpy as np
import pytest

from tests.strategies.test_ppo_intent import SequenceStrategy
from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.evaluation.directional import CloseAtEndStrategy
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOSharedCashCoordinator


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
