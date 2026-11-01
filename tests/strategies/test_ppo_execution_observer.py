from __future__ import annotations

import numpy as np

from tests.strategies.test_ppo_intent import market
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.stateful.execution import StatefulExecutionObservation
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def _environment(execution_observer=None) -> PPOTradingEnv:
    return PPOTradingEnv(
        market(),
        feature_indices=(0,),
        symbol_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig(order_latency_bars=1),
        settle_terminal_position=True,
        execution_observer=execution_observer,
    )


def _single_terminal_episode(env: PPOTradingEnv):
    observation, reset_info = env.reset(seed=17)
    result = env.step(2)
    next_observation, reward, terminated, truncated, step_info = result
    assert terminated is True
    assert truncated is False
    return (
        observation.copy(),
        reset_info,
        next_observation.copy(),
        reward,
        step_info,
        env.book.clone(),
    )


def test_execution_observer_none_parity_and_callback_survive_reset() -> None:
    events: list[StatefulExecutionObservation] = []
    baseline = _environment()
    observed = _environment(events.append)

    baseline_result = _single_terminal_episode(baseline)
    observed_result = _single_terminal_episode(observed)

    np.testing.assert_array_equal(baseline_result[0], observed_result[0])
    assert baseline_result[1] == observed_result[1]
    np.testing.assert_array_equal(baseline_result[2], observed_result[2])
    assert baseline_result[3:5] == observed_result[3:5]
    np.testing.assert_array_equal(
        baseline_result[5].quantities,
        observed_result[5].quantities,
    )
    np.testing.assert_array_equal(
        baseline_result[5].mark_prices,
        observed_result[5].mark_prices,
    )
    assert baseline_result[5].cash == observed_result[5].cash
    assert len(events) == 1 + observed_result[4]["terminal_settlement_intervals"]
    assert [event.next_index for event in events] == [1, 2, 3]

    first_episode_events = tuple(events)
    second_result = _single_terminal_episode(observed)
    assert len(events) == 2 * len(first_episode_events)
    assert tuple(events[: len(first_episode_events)]) == first_episode_events
    assert [event.next_index for event in events[len(first_episode_events) :]] == [
        1,
        2,
        3,
    ]
    np.testing.assert_array_equal(
        second_result[5].quantities,
        baseline_result[5].quantities,
    )
