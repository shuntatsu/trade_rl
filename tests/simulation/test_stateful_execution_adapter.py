"""Stateful compatibility execution contracts introduced test-first."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionRuleStress
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.simulation.targets.execution import execute_target_statefully


def market(*, volume: np.ndarray) -> MarketDataset:
    n = volume.shape[0]
    close = np.full((n, 1), 100.0)
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTC",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def cost(*, participation: float = 0.1) -> ExecutionCostConfig:
    return ExecutionCostConfig(
        fee_rate=0.0,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=participation,
        slippage_std=0.0,
        order_type="market",
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
    )


def test_compatibility_execution_uses_processing_bar_volume() -> None:
    dataset = market(volume=np.array([[1.0], [1_000.0], [1_000.0]]))
    result = MarketExecutor(dataset, cost()).execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )

    assert result.filled_turnover == pytest.approx(1.0)
    assert result.unfilled_turnover == pytest.approx(0.0)


def test_compatibility_execution_matches_shared_stateful_target_path() -> None:
    dataset = market(volume=np.full((4, 1), 5.0))
    initial = BookState.zero(1, 1_000.0, dataset.close[0])
    target = np.array([0.5])
    direct_executor = MarketExecutor(dataset, cost(participation=1.0))
    compatibility_executor = MarketExecutor(dataset, cost(participation=1.0))

    direct = execute_target_statefully(
        direct_executor,
        initial,
        OrderBookState.empty(),
        target,
        start_index=0,
        bars=1,
        target_identity="direct-parity",
    )
    compatibility = compatibility_executor.execute_interval(
        initial,
        target,
        start_index=0,
        bars=1,
    )

    assert compatibility.book.portfolio_value == pytest.approx(
        direct.book.portfolio_value
    )
    assert compatibility.book.quantities == pytest.approx(direct.book.quantities)
    assert compatibility.interval_net_return == pytest.approx(
        direct.interval_net_return
    )
    assert compatibility.filled_turnover == pytest.approx(direct.filled_turnover)


def test_stateful_target_replaces_matching_residual_from_old_policy() -> None:
    dataset = market(volume=np.full((4, 1), 1_000.0))
    executor = MarketExecutor(dataset, cost(participation=1.0))
    old_intent = OrderIntent.create(
        dataset_id=dataset.dataset_id,
        target_identity="old-target",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=5.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    old_order = PendingOrder.from_intent(old_intent)

    result = execute_target_statefully(
        executor,
        BookState.zero(1, 1_000.0, dataset.close[0]),
        OrderBookState(active_orders=(old_order,), terminal_orders=()),
        np.array([0.5]),
        start_index=0,
        bars=1,
        target_identity="current-target",
    )

    assert result.book.quantities[0] == pytest.approx(5.0)
    cancelled = [
        event
        for event in result.order_events
        if event.order_id == old_order.order_id and event.event_type == "cancelled"
    ]
    assert len(cancelled) == 1
    assert cancelled[0].reason == "superseded"
    assert all(
        event.reason != "identity_mismatch"
        for event in result.order_events
        if event.event_type == "rejected"
    )


def test_stateful_target_replaces_matching_residual_from_nominal_policy_under_stress() -> (
    None
):
    n_bars = 4
    volume = np.full((n_bars, 1), 1_000.0)
    dataset = MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTC",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.full((n_bars, 1), 100.0),
        high=np.full((n_bars, 1), 101.0),
        low=np.full((n_bars, 1), 99.0),
        close=np.full((n_bars, 1), 100.0),
        volume=volume,
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        tick_size=np.full((n_bars, 1), 0.1),
    )
    base_cost = cost(participation=1.0)
    nominal = MarketExecutor(dataset, base_cost)
    stressed = MarketExecutor(
        dataset,
        base_cost,
        rule_stress=ExecutionRuleStress(
            name="tick_2x",
            tick_size_factor=2.0,
        ),
    )
    old_intent = OrderIntent.create(
        dataset_id=dataset.dataset_id,
        target_identity="old-nominal-target",
        execution_policy_digest=nominal.execution_policy_digest,
        symbol_index=0,
        requested_quantity=5.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    old_order = PendingOrder.from_intent(old_intent)

    result = execute_target_statefully(
        stressed,
        BookState.zero(1, 1_000.0, dataset.close[0]),
        OrderBookState(active_orders=(old_order,), terminal_orders=()),
        np.array([0.5]),
        start_index=0,
        bars=1,
        target_identity="stressed-target",
    )

    cancelled = [
        event
        for event in result.order_events
        if event.order_id == old_order.order_id and event.event_type == "cancelled"
    ]
    assert len(cancelled) == 1
    assert cancelled[0].reason == "superseded"
    assert any(
        order.intent.execution_policy_digest == stressed.execution_policy_digest
        for order in result.order_book.terminal_orders
        if order.order_id != old_order.order_id
    )


def test_chained_compatibility_calls_keep_one_residual_order() -> None:
    dataset = market(volume=np.full((5, 1), 1.0))
    executor = MarketExecutor(dataset, cost(participation=0.1))
    target = np.array([1.0])

    first = executor.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        target,
        start_index=0,
        bars=1,
    )
    first_active = executor.compatibility_order_book.active_orders
    second = executor.execute_interval(
        first.book,
        target,
        start_index=1,
        bars=1,
    )
    second_active = executor.compatibility_order_book.active_orders

    assert first.filled_turnover == pytest.approx(0.01)
    assert second.filled_turnover == pytest.approx(0.01)
    assert len(first_active) == 1
    assert len(second_active) == 1
    assert second_active[0].order_id == first_active[0].order_id
    assert second_active[0].remaining_quantity < first_active[0].remaining_quantity
