from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.simulation.stateful.execution import StatefulExecutionObservation
from trade_rl.strategies.rl.portfolio_env import (
    ACTION_HOLD,
    ACTION_LONG,
    SharedPortfolioGymEnv,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)
_ACTOR_STEPS = 128
_TERMINAL_ROW = _ACTOR_STEPS + 1
_INITIAL_CAPITAL = 1_000.0


def _delayed_event_env(
    *,
    payoff_transition: int = 72,
    observer: list[StatefulExecutionObservation] | None = None,
) -> SharedPortfolioGymEnv:
    n_bars = _TERMINAL_ROW + 1
    close = np.full((n_bars, 1), 100.0, dtype=np.float64)
    close[payoff_transition:, 0] = 101.5
    open_price = np.vstack((close[0], close[:-1]))
    volume = np.zeros((n_bars, 1), dtype=np.float64)
    volume[1, 0] = 1_000_000.0
    volume[_TERMINAL_ROW, 0] = 1_000_000.0
    cue = np.ones((n_bars, 1, 1), dtype=np.float32)
    zeros = np.zeros((n_bars, 1), dtype=np.float64)
    dataset = MarketDataset(
        dataset_id="f" * 64,
        symbols=("SYNTH",),
        timestamps=(
            np.datetime64("2026-01-01T00:00:00", "ns")
            + np.arange(n_bars) * np.timedelta64(1, "h")
        ),
        features=cue,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=volume,
        funding_rate=zeros.copy(),
        borrow_rate=zeros.copy(),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("persistent_cue",),
        global_feature_names=("constant_regime",),
        global_feature_available=np.ones((n_bars, 1), dtype=np.bool_),
        periods_per_year=8_760,
        fee_rate=zeros.copy(),
        maker_fee_rate=zeros.copy(),
        taker_fee_rate=zeros.copy(),
        spread_rate=zeros.copy(),
        contract_multipliers=np.ones(1, dtype=np.float64),
    )
    objective = ObjectiveContract(
        capital=CapitalContract("shared_portfolio", "USD", (_INITIAL_CAPITAL,)),
        evaluation_start=_START,
        evaluation_stop_exclusive=_START + timedelta(hours=_TERMINAL_ROW),
        terminal_valuation="settled",
        economics_digest="a" * 64,
        risk_digest="b" * 64,
        deployment_recipe_digest="c" * 64,
        maximum_drawdown=0.20,
    )
    clock = FinancialClockContract(
        decision_interval_seconds=3_600,
        execution_interval_seconds=3_600,
        reward_interval_seconds=3_600,
        economic_horizon_seconds=_TERMINAL_ROW * 3_600,
        rollout_steps=64,
        gamma=1.0,
        gae_lambda=0.95,
        reward_schema="equity_delta_v1",
    )
    bound = BoundObjectiveClock(objective, clock)
    return SharedPortfolioGymEnv(
        dataset,
        start_index=0,
        stop_index=_TERMINAL_ROW,
        bound=bound,
        target_abs_weights=(0.5,),
        execution_cost=ExecutionCostConfig(),
        risk_config=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.20,
            drawdown_stop=0.20,
        ),
        execution_observer=None if observer is None else observer.append,
    )


def _events(
    observations: list[StatefulExecutionObservation],
) -> list[OrderEvent]:
    return [event for observation in observations for event in observation.order_events]


@pytest.mark.parametrize("payoff_transition", (48, 72))
def test_delayed_payoff_uses_one_native_account_and_reserved_close(
    payoff_transition: int,
) -> None:
    observations: list[StatefulExecutionObservation] = []
    env = _delayed_event_env(
        payoff_transition=payoff_transition,
        observer=observations,
    )
    observation, _ = env.reset(seed=19)
    rewards: list[float] = []
    terminal_info: dict[str, object] | None = None

    for actor_step in range(_ACTOR_STEPS):
        action = ACTION_LONG if actor_step == 0 else ACTION_HOLD
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.int64)
        )
        rewards.append(reward)
        assert truncated is False
        if actor_step < _ACTOR_STEPS - 1:
            assert terminated is False
            if info["current_index"] in {
                1,
                64,
                71,
                72,
                payoff_transition,
                128,
            }:
                assert observation["features"][0, 0] == pytest.approx(1.0)
            if actor_step == 0:
                assert reward < 0.0
            elif actor_step < payoff_transition - 1:
                assert reward == pytest.approx(0.0, abs=1e-12)
            elif actor_step == payoff_transition - 1:
                assert info["current_index"] == payoff_transition
                assert env.dataset.open[payoff_transition, 0] == pytest.approx(
                    env.dataset.close[payoff_transition - 1, 0]
                )
                assert env.dataset.close[payoff_transition, 0] == pytest.approx(
                    env.dataset.close[payoff_transition - 1, 0] * 1.015
                )
                assert reward > 0.0
        else:
            assert terminated is True
            terminal_info = info

    assert terminal_info is not None
    final_state = terminal_info["terminal_state"]
    assert isinstance(final_state, dict)
    assert terminal_info["elapsed_intervals"] == 2
    assert terminal_info["settlement_complete"] is True
    assert terminal_info["objective_endpoint_valid"] is True
    assert final_state["current_index"] == _TERMINAL_ROW
    assert final_state["exact_quantities"] == ("0",)
    assert final_state["active_orders"] == ()
    assert final_state["portfolio_value"] > _INITIAL_CAPITAL
    assert final_state["total_cost"] > 0.0
    assert sum(rewards) == pytest.approx(
        (final_state["portfolio_value"] - _INITIAL_CAPITAL) / _INITIAL_CAPITAL
    )

    fills = [
        event
        for event in _events(observations)
        if event.event_type in {"filled", "partial_fill"}
    ]
    entry_fills = [event for event in fills if event.processing_index == 1]
    assert len(entry_fills) == 1
    assert entry_fills[0].filled_quantity == pytest.approx(5.0)
    terminal_fills = [
        event for event in fills if event.processing_index == _TERMINAL_ROW
    ]
    assert len(terminal_fills) == 1
    assert all(event.reduce_only for event in terminal_fills)

    # The dataset contributes no embedded fee/spread; the pinned execution
    # policy supplies 5bp fee + 2bp spread + sqrt-participation impact at 1bp.
    assert np.count_nonzero(env.dataset.fee_rate) == 0
    assert np.count_nonzero(env.dataset.spread_rate) == 0
    assert env.execution_cost.fee_rate == pytest.approx(0.0005)
    assert env.execution_cost.spread_rate == pytest.approx(0.0002)
    assert env.execution_cost.impact_rate == pytest.approx(0.0001)

    # Rebuild every execution charge from the native fill facts, independently
    # of the environment's reward and terminal equity calculations.
    independent_notionals: list[float] = []
    for event in fills:
        assert event.execution_price is not None
        independent_notionals.append(abs(event.filled_quantity * event.execution_price))
    expected_cost = math.fsum(
        notional
        * (
            env.execution_cost.fee_rate
            + env.execution_cost.spread_rate
            + env.execution_cost.impact_rate * math.sqrt(event.participation_rate)
        )
        for notional, event in zip(independent_notionals, fills, strict=True)
    )
    expected_cash = (
        _INITIAL_CAPITAL
        - math.fsum(
            event.filled_quantity * event.execution_price
            for event in fills
            if event.execution_price is not None
        )
        - expected_cost
    )
    assert final_state["total_cost"] == pytest.approx(expected_cost, rel=1e-10)
    assert final_state["cash"] == pytest.approx(expected_cash, rel=1e-10)
    assert final_state["portfolio_value"] == pytest.approx(expected_cash, rel=1e-10)
    assert math.log(final_state["portfolio_value"] / _INITIAL_CAPITAL) > 0.0


def test_late_entry_cannot_capture_event_or_reserved_close_liquidity() -> None:
    observations: list[StatefulExecutionObservation] = []
    env = _delayed_event_env(observer=observations)
    env.reset(seed=19)
    late_order_id: str | None = None
    after_late_entry: dict[str, np.ndarray] | None = None
    after_cancel: dict[str, np.ndarray] | None = None
    terminal_info: dict[str, object] | None = None

    for actor_step in range(_ACTOR_STEPS):
        action = ACTION_LONG if actor_step == 71 else ACTION_HOLD
        observation, _, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.int64)
        )
        assert truncated is False
        if actor_step == 71:
            after_late_entry = observation
            assert info["current_index"] == 72
            assert observation["account"][0, 0] == pytest.approx(0.0)
            assert observation["active_orders"][0, 0] == pytest.approx(1.0)
        if actor_step == 72:
            after_cancel = observation
            assert info["current_index"] == 73
            assert observation["active_orders"][0, 0] == pytest.approx(0.0)
        if actor_step < _ACTOR_STEPS - 1:
            assert terminated is False
        else:
            assert terminated is True
            terminal_info = info

    assert after_late_entry is not None
    assert after_cancel is not None
    assert terminal_info is not None
    events = _events(observations)
    late_submission = next(
        event
        for event in events
        if event.event_type == "submitted" and event.processing_index == 71
    )
    late_order_id = late_submission.order_id
    cancellation = next(
        event
        for event in events
        if event.order_id == late_order_id and event.event_type == "cancelled"
    )
    assert cancellation.reason == "superseded"
    assert cancellation.processing_index == 72
    assert not any(
        event.order_id == late_order_id
        and event.processing_index == _TERMINAL_ROW
        and event.event_type in {"filled", "partial_fill"}
        for event in events
    )

    final_state = terminal_info["terminal_state"]
    assert isinstance(final_state, dict)
    assert terminal_info["settlement_complete"] is True
    assert terminal_info["objective_endpoint_valid"] is True
    assert final_state["current_index"] == _TERMINAL_ROW
    assert final_state["portfolio_value"] == pytest.approx(_INITIAL_CAPITAL)
    assert final_state["exact_quantities"] == ("0",)
    assert final_state["active_orders"] == ()


def test_late_pending_entry_is_cancelled_before_terminal_liquidity() -> None:
    observations: list[StatefulExecutionObservation] = []
    env = _delayed_event_env(observer=observations)
    env.reset(seed=19)
    terminal_info: dict[str, object] | None = None

    for actor_step in range(_ACTOR_STEPS):
        action = ACTION_LONG if actor_step == _ACTOR_STEPS - 1 else ACTION_HOLD
        observation, _, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.int64)
        )
        assert truncated is False
        if actor_step < _ACTOR_STEPS - 1:
            assert terminated is False
        else:
            assert terminated is True
            terminal_info = info

    assert terminal_info is not None
    events = _events(observations)
    late_order = next(
        event
        for event in events
        if event.event_type == "submitted"
        and event.processing_index == _ACTOR_STEPS - 1
    )
    terminal_cancel = next(
        event
        for event in events
        if event.order_id == late_order.order_id and event.event_type == "cancelled"
    )
    assert terminal_cancel.reason == "superseded"
    assert terminal_cancel.processing_index == _ACTOR_STEPS
    assert not any(
        event.order_id == late_order.order_id
        and event.processing_index == _TERMINAL_ROW
        and event.event_type in {"filled", "partial_fill"}
        for event in events
    )
    final_state = terminal_info["terminal_state"]
    assert isinstance(final_state, dict)
    assert terminal_info["settlement_complete"] is True
    assert terminal_info["objective_endpoint_valid"] is True
    assert final_state["current_index"] == _TERMINAL_ROW
    assert final_state["portfolio_value"] == pytest.approx(_INITIAL_CAPITAL)
    assert final_state["exact_quantities"] == ("0",)
    assert final_state["active_orders"] == ()


def test_installed_sb3_buffer_bootstraps_after_cost_future_value_at_rollout_edge() -> (
    None
):
    torch = pytest.importorskip("torch")
    buffers = pytest.importorskip("stable_baselines3.common.buffers")
    buffer_type = buffers.DictRolloutBuffer

    observations: list[StatefulExecutionObservation] = []
    env = _delayed_event_env(payoff_transition=72, observer=observations)
    raw_observation, _ = env.reset(seed=19)
    boundary_equity: float | None = None
    boundary_quantity: float | None = None
    terminal_state: dict[str, object] | None = None
    episode_rewards: list[float] = []

    for actor_step in range(_ACTOR_STEPS):
        action = ACTION_LONG if actor_step == 0 else ACTION_HOLD
        raw_observation, reward, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.int64)
        )
        episode_rewards.append(reward)
        assert truncated is False
        if info["current_index"] == 64:
            assert terminated is False
            boundary_equity = float(raw_observation["portfolio"][1])
            boundary_quantity = float(raw_observation["account"][0, 0])
        if terminated:
            terminal = info["terminal_state"]
            assert isinstance(terminal, dict)
            terminal_state = terminal

    assert boundary_equity is not None
    assert boundary_quantity == pytest.approx(5.0)
    assert terminal_state is not None
    assert terminal_state["settlement_complete"] is True
    assert terminal_state["objective_endpoint_valid"] is True

    final_equity = float(terminal_state["portfolio_value"])
    future_after_cost_value = (final_equity - boundary_equity) / _INITIAL_CAPITAL
    realized_episode_return = sum(episode_rewards)
    assert future_after_cost_value > 0.0
    assert realized_episode_return < future_after_cost_value

    fills = [
        event
        for item in observations
        for event in item.order_events
        if event.event_type in {"filled", "partial_fill"}
    ]
    entry_fill = next(event for event in fills if event.processing_index == 1)
    assert entry_fill.execution_price is not None
    entry_notional = abs(entry_fill.filled_quantity * entry_fill.execution_price)
    entry_cost = entry_notional * (
        env.execution_cost.fee_rate
        + env.execution_cost.spread_rate
        + env.execution_cost.impact_rate * math.sqrt(entry_fill.participation_rate)
    )
    assert future_after_cost_value - realized_episode_return == pytest.approx(
        entry_cost / _INITIAL_CAPITAL,
        rel=1e-9,
        abs=1e-12,
    )

    buffer = buffer_type(
        buffer_size=64,
        observation_space=env.observation_space,
        action_space=env.action_space,
        gamma=1.0,
        gae_lambda=0.95,
        n_envs=1,
    )
    batched_observation = {
        key: np.expand_dims(value, axis=0) for key, value in raw_observation.items()
    }
    for index in range(64):
        buffer.add(
            obs=batched_observation,
            action=np.asarray([[ACTION_HOLD]], dtype=np.int64),
            reward=np.asarray([episode_rewards[index]], dtype=np.float32),
            episode_start=np.asarray([index == 0], dtype=np.bool_),
            value=torch.tensor([0.0]),
            log_prob=torch.tensor([0.0]),
        )

    buffer.compute_returns_and_advantage(
        last_values=torch.tensor([future_after_cost_value]),
        dones=np.asarray([False]),
    )
    assert float(buffer.returns[-1, 0]) == pytest.approx(future_after_cost_value)

    buffer.compute_returns_and_advantage(
        last_values=torch.tensor([future_after_cost_value]),
        dones=np.asarray([True]),
    )
    assert float(buffer.returns[-1, 0]) == pytest.approx(episode_rewards[63])
