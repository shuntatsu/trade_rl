from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import gymnasium as gym
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
from trade_rl.simulation.orders.model import OrderStatus, OrderType, TimeInForce
from trade_rl.simulation.stateful.execution import StatefulExecutionObservation
from trade_rl.strategies.portfolio import PortfolioPendingOrderSnapshot
from trade_rl.strategies.rl.portfolio_env import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    SharedPortfolioGymEnv,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _dataset(
    *,
    dataset_id: str = "d" * 64,
    symbols: tuple[str, ...] = ("AAA",),
    close: np.ndarray | None = None,
    open_price: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    features: np.ndarray | None = None,
    funding_rate: np.ndarray | None = None,
    funding_due: np.ndarray | None = None,
    borrow_rate: np.ndarray | None = None,
    split_factor: np.ndarray | None = None,
) -> MarketDataset:
    n_symbols = len(symbols)
    resolved_close = np.asarray(
        np.full((6, n_symbols), 100.0) if close is None else close,
        dtype=np.float64,
    )
    n_bars = resolved_close.shape[0]
    resolved_open_price = np.asarray(
        np.vstack((resolved_close[0], resolved_close[:-1]))
        if open_price is None
        else open_price,
        dtype=np.float64,
    )
    resolved_volume = np.asarray(
        np.full((n_bars, n_symbols), 1_000_000.0) if volume is None else volume,
        dtype=np.float64,
    )
    resolved_features = np.asarray(
        np.broadcast_to(
            np.arange(n_bars, dtype=np.float32)[:, None, None],
            (n_bars, n_symbols, 1),
        ).copy()
        if features is None
        else features,
        dtype=np.float32,
    )
    zeros = np.zeros((n_bars, n_symbols), dtype=np.float64)
    return MarketDataset(
        dataset_id=dataset_id,
        symbols=symbols,
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=resolved_features,
        global_features=np.arange(n_bars, dtype=np.float32).reshape(n_bars, 1),
        open=resolved_open_price,
        high=np.maximum(resolved_open_price, resolved_close),
        low=np.minimum(resolved_open_price, resolved_close),
        close=resolved_close,
        volume=resolved_volume,
        funding_rate=zeros.copy() if funding_rate is None else funding_rate,
        funding_due=funding_due,
        borrow_rate=zeros.copy() if borrow_rate is None else borrow_rate,
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("literal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=zeros.copy(),
        maker_fee_rate=zeros.copy(),
        taker_fee_rate=zeros.copy(),
        spread_rate=zeros.copy(),
        contract_multipliers=np.ones(n_symbols, dtype=np.float64),
        split_factor=split_factor,
    )


def _bound_clock(
    *,
    start_index: int = 0,
    stop_index: int = 5,
    gamma: float = 1.0,
    reward_schema: str = "equity_delta_v1",
    max_drawdown: float = 0.20,
) -> BoundObjectiveClock:
    objective = ObjectiveContract(
        capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
        evaluation_start=_START + timedelta(hours=start_index),
        evaluation_stop_exclusive=_START + timedelta(hours=stop_index),
        terminal_valuation="settled",
        economics_digest="a" * 64,
        risk_digest="b" * 64,
        deployment_recipe_digest="c" * 64,
        maximum_drawdown=max_drawdown,
    )
    clock = FinancialClockContract(
        decision_interval_seconds=3_600,
        execution_interval_seconds=3_600,
        reward_interval_seconds=3_600,
        economic_horizon_seconds=(stop_index - start_index) * 3_600,
        rollout_steps=1,
        gamma=gamma,
        gae_lambda=0.95,
        reward_schema=reward_schema,  # type: ignore[arg-type]
    )
    return BoundObjectiveClock(objective, clock)


def _risk(
    *, drawdown_start: float = 0.20, drawdown_stop: float = 0.20
) -> PreTradeRiskConfig:
    return PreTradeRiskConfig(
        max_gross=1.0,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=drawdown_start,
        drawdown_stop=drawdown_stop,
    )


def _pending_order_snapshot(
    *,
    submission_reference_price: float | None = 100.0,
) -> PortfolioPendingOrderSnapshot:
    return PortfolioPendingOrderSnapshot(
        symbol_index=0,
        order_type=OrderType.MARKET.value,
        time_in_force=TimeInForce.GTC.value,
        requested_quantity=1.0,
        remaining_quantity=1.0,
        status=OrderStatus.SUBMITTED.value,
        submit_index=0,
        eligible_index=1,
        last_processed_index=None,
        trigger_index=None,
        expiry_index=None,
        reduce_only=False,
        submission_reference_price=submission_reference_price,
    )


def _env(
    dataset: MarketDataset,
    *,
    start_index: int = 0,
    stop_index: int = 5,
    execution_cost: ExecutionCostConfig | None = None,
    risk_config: PreTradeRiskConfig | None = None,
    bound: BoundObjectiveClock | None = None,
    target_abs_weights: tuple[float, ...] | None = None,
    execution_observer: Callable[[StatefulExecutionObservation], None] | None = None,
) -> SharedPortfolioGymEnv:
    return SharedPortfolioGymEnv(
        dataset,
        start_index=start_index,
        stop_index=stop_index,
        bound=bound or _bound_clock(start_index=start_index, stop_index=stop_index),
        target_abs_weights=target_abs_weights or (0.2,) * dataset.n_symbols,
        execution_cost=execution_cost or ExecutionCostConfig.zero(),
        risk_config=risk_config or _risk(),
        execution_observer=execution_observer,
    )


def test_fixed_joint_spaces_encode_a_detached_current_snapshot() -> None:
    dataset = _dataset(symbols=("AAA", "BBB"))
    env = _env(dataset)

    observation, info = env.reset(seed=19)

    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert tuple(env.action_space.nvec) == (4, 4)
    assert tuple(observation) == env.observation_keys
    assert env.observation_space.contains(observation)
    assert np.array_equal(observation["features"], dataset.features[0])
    assert np.array_equal(
        observation["feature_available"], dataset.feature_available[0].astype(np.int8)
    )
    assert observation["clock"].tolist() == [
        0,
        int(dataset.timestamps[0].astype("datetime64[ns]").astype("int64")),
        4,
        4,
    ]
    assert observation["terminal"].tolist() == [0]
    assert info["current_index"] == 0

    original = dataset.features[0].copy()
    observation["features"][0, 0] = -999.0
    assert np.array_equal(dataset.features[0], original)


def test_encoder_rejects_multiple_active_orders_for_one_symbol() -> None:
    env = _env(_dataset())
    env.reset()

    with pytest.raises(ValueError, match="at most one active order per symbol"):
        env._encode_active_orders(
            (_pending_order_snapshot(), _pending_order_snapshot())
        )


def test_encoder_rejects_legacy_order_without_price_context() -> None:
    env = _env(_dataset())
    env.reset()

    with pytest.raises(ValueError, match="price context is incomplete"):
        env._encode_active_orders(
            (_pending_order_snapshot(submission_reference_price=None),)
        )


def test_invalid_joint_actions_do_not_advance_the_shared_session() -> None:
    dataset = _dataset(symbols=("AAA", "BBB"))
    cost = replace(ExecutionCostConfig.zero(), slippage_std=0.02, random_seed=41)
    after_invalid = _env(dataset, execution_cost=cost)
    clean = _env(dataset, execution_cost=cost)
    after_invalid.reset(seed=7)
    clean.reset(seed=7)

    invalid_actions: tuple[object, ...] = (
        np.asarray([True, False]),
        [True, 1],
        np.asarray([1.0, 0.0]),
        np.asarray([1]),
        np.asarray([4, 0]),
    )
    for action in invalid_actions:
        with pytest.raises((TypeError, ValueError)):
            after_invalid.step(action)

    expected = clean.step(np.asarray([ACTION_LONG, ACTION_LONG], dtype=np.int64))
    actual = after_invalid.step(np.asarray([ACTION_LONG, ACTION_LONG], dtype=np.int64))

    assert actual[1:4] == expected[1:4]
    assert actual[4]["terminal_state"] == expected[4]["terminal_state"]
    for key in actual[0]:
        assert np.array_equal(actual[0][key], expected[0][key])


def test_environment_rejects_non_telescoping_or_misaligned_objective_clock() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="gamma-one"):
        _env(dataset, bound=_bound_clock(gamma=0.99))
    with pytest.raises(ValueError, match="objective start"):
        _env(dataset, bound=_bound_clock(start_index=1, stop_index=5))


def test_observer_failure_poisoning_prevents_retry_after_possible_mutation() -> None:
    def fail_after_execution(_observation: StatefulExecutionObservation) -> None:
        raise RuntimeError("literal observer failure")

    env = _env(_dataset(), execution_observer=fail_after_execution)
    env.reset()

    with pytest.raises(RuntimeError, match="literal observer failure"):
        env.step(np.asarray([ACTION_LONG], dtype=np.int64))
    with pytest.raises(RuntimeError, match="failed after possible state mutation"):
        env.step(np.asarray([ACTION_LONG], dtype=np.int64))


def test_terminal_reward_aggregates_reserved_close_and_reconciles_literal_fees() -> (
    None
):
    volume = np.full((6, 1), 1_000_000.0)
    volume[1, 0] = 2.0
    cost = replace(
        ExecutionCostConfig.zero(),
        fee_rate=0.01,
        max_participation_rate=0.5,
    )
    env = _env(_dataset(volume=volume), execution_cost=cost)
    observation, _ = env.reset()
    rewards: list[float] = []
    terminal_info: dict[str, object] | None = None

    for action in (ACTION_LONG, ACTION_HOLD, ACTION_HOLD, ACTION_HOLD):
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.int64)
        )
        rewards.append(reward)
        assert truncated is False
        if terminated:
            terminal_info = info
            break

    assert terminal_info is not None
    assert terminated is True
    assert observation["terminal"].tolist() == [1]
    assert env.observation_space.contains(observation)
    final_state = terminal_info["terminal_state"]
    assert isinstance(final_state, dict)
    assert final_state["cash"] == pytest.approx(998.0)
    assert final_state["portfolio_value"] == pytest.approx(998.0)
    assert final_state["total_cost"] == pytest.approx(2.0)
    assert final_state["exact_quantities"] == ("0",)
    assert final_state["active_orders"] == ()
    assert terminal_info["settlement_complete"] is True
    assert terminal_info["objective_endpoint_valid"] is True
    assert terminal_info["elapsed_intervals"] == 2
    assert sum(rewards) == pytest.approx((998.0 - 1_000.0) / 1_000.0)


def test_short_gap_preserves_native_insolvency_floor_and_invalid_endpoint() -> None:
    close = np.asarray([[100.0], [100.0], [500.0], [500.0], [500.0], [500.0]])
    open_price = close.copy()
    cost = replace(ExecutionCostConfig.zero(), fee_rate=0.01)
    env = _env(
        _dataset(close=close, open_price=open_price),
        execution_cost=cost,
        target_abs_weights=(0.5,),
    )
    env.reset()

    first_observation, entry_reward, entry_terminated, _, _ = env.step(
        np.asarray([ACTION_SHORT], dtype=np.int64)
    )
    assert not entry_terminated
    assert first_observation["account"][0, 0] == pytest.approx(-5.0)
    assert entry_reward == pytest.approx(-5.0 / 1_000.0)

    observation, gap_reward, terminated, truncated, info = env.step(
        np.asarray([ACTION_HOLD], dtype=np.int64)
    )

    assert terminated and not truncated
    assert observation["terminal"].tolist() == [1]
    terminal_state = info["terminal_state"]
    assert isinstance(terminal_state, dict)
    assert terminal_state["cash"] == 0.0
    assert terminal_state["portfolio_value"] == 0.0
    assert terminal_state["total_cost"] == pytest.approx(5.0)
    assert terminal_state["max_drawdown"] == pytest.approx(1.0)
    assert terminal_state["termination_reason"] == "insolvency"
    assert terminal_state["settlement_complete"] is False
    assert terminal_state["objective_endpoint_valid"] is False
    assert gap_reward == pytest.approx(-995.0 / 1_000.0)
    assert entry_reward + gap_reward == pytest.approx(-1.0)


def test_split_cancels_partial_remainder_and_retarget_uses_adjusted_book() -> None:
    close = np.asarray(
        [[100.0], [100.0], [50.0], [50.0], [50.0], [50.0], [50.0], [50.0]]
    )
    volume = np.full((8, 1), 1_000_000.0)
    volume[1, 0] = 2.0
    split_factor = np.ones((8, 1), dtype=np.float64)
    split_factor[2, 0] = 2.0
    cost = replace(
        ExecutionCostConfig.zero(),
        order_latency_bars=1,
        max_participation_rate=0.5,
    )
    env = _env(
        _dataset(close=close, volume=volume, split_factor=split_factor),
        stop_index=7,
        execution_cost=cost,
        target_abs_weights=(0.5,),
    )
    env.reset()

    after_submit, *_ = env.step(np.asarray([ACTION_LONG], dtype=np.int64))
    first_order = after_submit["active_orders"][0].copy()
    assert first_order[0] == 1.0
    assert first_order[4] == pytest.approx(5.0)
    assert first_order[5] == pytest.approx(4.0)
    assert after_submit["account"][0, 0] == pytest.approx(1.0)

    after_split, *_ = env.step(np.asarray([ACTION_LONG], dtype=np.int64))
    split_order = after_split["active_orders"][0]
    assert split_order[0] == 0.0
    assert after_split["account"][0, 0] == pytest.approx(2.0)

    after_fill, *_ = env.step(np.asarray([ACTION_LONG], dtype=np.int64))
    assert after_fill["active_orders"][0, 0] == 0.0
    assert after_fill["account"][0, 0] == pytest.approx(10.0)


def test_future_tail_changes_do_not_change_current_observation_or_order() -> None:
    base_features = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    future_features = base_features.copy()
    future_features[2:] += 100.0
    base_close = np.full((6, 1), 100.0)
    future_close = base_close.copy()
    future_close[2:] = np.asarray([80.0, 120.0, 70.0, 140.0])[:, None]
    future_funding = np.zeros((6, 1), dtype=np.float64)
    future_funding[2:] = 0.01
    cost = replace(ExecutionCostConfig.zero(), order_latency_bars=2)
    first = _env(
        _dataset(close=base_close, features=base_features),
        execution_cost=cost,
    )
    second = _env(
        _dataset(
            dataset_id="e" * 64,
            close=future_close,
            features=future_features,
            funding_rate=future_funding,
        ),
        execution_cost=cost,
    )
    first.reset()
    second.reset()

    first_observation, *_ = first.step(np.asarray([ACTION_LONG], dtype=np.int64))
    second_observation, *_ = second.step(np.asarray([ACTION_LONG], dtype=np.int64))

    for key in first.observation_keys:
        np.testing.assert_array_equal(first_observation[key], second_observation[key])
    assert first_observation["active_orders"][0, 0] == 1.0
    assert first_observation["active_orders"][0, 5] == pytest.approx(2.0)


def test_risk_stop_overrides_hold_and_capacity_failure_is_not_settled_success() -> None:
    falling = np.full((6, 1), 100.0)
    falling[1:, 0] = 50.0
    risk = _risk(drawdown_start=0.0, drawdown_stop=0.0)
    env = _env(
        _dataset(close=falling),
        risk_config=risk,
        target_abs_weights=(0.5,),
    )
    env.reset()
    env.step(np.asarray([ACTION_LONG], dtype=np.int64))
    observation, _, terminated, _, _ = env.step(
        np.asarray([ACTION_HOLD], dtype=np.int64)
    )
    assert not terminated
    assert observation["account"][0, 0] == 0.0

    volume = np.full((6, 1), 1_000_000.0)
    volume[5, 0] = 0.0
    capacity_env = _env(
        _dataset(volume=volume),
        target_abs_weights=(0.5,),
    )
    capacity_env.reset()
    for action in (ACTION_LONG, ACTION_HOLD, ACTION_HOLD, ACTION_HOLD):
        observation, _, terminated, _, info = capacity_env.step(
            np.asarray([action], dtype=np.int64)
        )
        if terminated:
            break
    assert terminated
    assert observation["terminal"].tolist() == [1]
    assert info["settlement_complete"] is False
    assert info["objective_endpoint_valid"] is False
    terminal_state = info["terminal_state"]
    assert isinstance(terminal_state, dict)
    assert terminal_state["exact_quantities"] != ("0",)
    assert terminal_state["active_orders"]


def test_settled_endpoint_does_not_hide_drawdown_beyond_configured_stop() -> None:
    close = np.full((6, 1), 100.0)
    close[1:, 0] = 50.0
    env = _env(
        _dataset(close=close),
        target_abs_weights=(0.5,),
        risk_config=_risk(drawdown_start=0.20, drawdown_stop=0.20),
    )
    env.reset()

    for action in (ACTION_LONG, ACTION_HOLD, ACTION_HOLD, ACTION_HOLD):
        _, _, terminated, _, info = env.step(np.asarray([action], dtype=np.int64))
        if terminated:
            break

    assert terminated
    terminal_state = info["terminal_state"]
    assert isinstance(terminal_state, dict)
    assert terminal_state["max_drawdown"] == pytest.approx(0.25)
    assert terminal_state["settlement_complete"] is True
    assert terminal_state["objective_endpoint_valid"] is True
