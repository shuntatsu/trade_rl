from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.simulation.quantities import exact_quantity, project_quantity


def _two_symbol_market() -> MarketDataset:
    n_bars = 4
    shape = (n_bars, 2)
    close = np.tile(np.array([100.0, 200.0]), (n_bars, 1))
    volume = np.full(shape, 1_000.0)
    volume[1] = 0.0
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("S0", "S1"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 10.0,
        low=close - 10.0,
        close=close.copy(),
        volume=volume,
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        contract_multipliers=np.array([1.0, 0.8]),
    )


def _market_order(
    dataset: MarketDataset,
    executor: MarketExecutor,
    quantity: float,
    *,
    target_identity: str,
    eligible_index: int,
    reduce_only: bool = False,
) -> PendingOrder:
    intent = OrderIntent.create(
        dataset_id=dataset.dataset_id,
        target_identity=target_identity,
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=eligible_index,
        expiry_index=None,
        submission_reference_price=float(dataset.close[0, 0]),
        decision_equity=1_000.0,
        reduce_only=reduce_only,
    )
    return PendingOrder.from_intent(intent)


def test_same_bar_admission_projects_exposure_in_order_without_mutating_input() -> None:
    dataset = _two_symbol_market()
    config = replace(
        ExecutionCostConfig.zero(),
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        max_leverage=1.0,
    )
    executor = MarketExecutor(dataset, config)
    intents = tuple(
        OrderIntent.create(
            dataset_id=dataset.dataset_id,
            target_identity=f"target-{symbol_index}",
            execution_policy_digest=executor.execution_policy_digest,
            symbol_index=symbol_index,
            requested_quantity=6.0,
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
        for symbol_index in range(2)
    )
    orders = OrderBookState.empty().add(
        *(PendingOrder.from_intent(intent) for intent in intents)
    )
    initial_book = BookState.zero(
        2,
        1_000.0,
        dataset.close[0],
        dataset.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        initial_book,
        orders,
        (),
        start_index=0,
        bars=1,
    )

    rejected = [
        event for event in result.order_events if event.event_type == "rejected"
    ]
    eligible = [
        event for event in result.order_events if event.event_type == "eligible"
    ]
    assert len(rejected) == 1
    assert rejected[0].reason == "pretrade_leverage_exceeded"
    assert len(eligible) == 1
    assert len(result.order_book.active_orders) == 1
    np.testing.assert_array_equal(initial_book.quantities, np.zeros(2))
    assert initial_book.cash == 1_000.0


def test_over_cap_reduce_only_partial_close_is_admitted_at_processing_open() -> None:
    dataset = _two_symbol_market()
    config = replace(
        ExecutionCostConfig.zero(),
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        max_leverage=1.0,
    )
    executor = MarketExecutor(dataset, config)
    closing = _market_order(
        dataset,
        executor,
        -2.0,
        target_identity="over-cap-partial-close",
        eligible_index=1,
        reduce_only=True,
    )
    initial_book = BookState(
        quantities=np.array([14.0, 0.0]),
        cash=-400.0,
        mark_prices=dataset.close[0],
        peak_value=1_000.0,
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        initial_book,
        OrderBookState.empty().add(closing),
        (),
        start_index=0,
        bars=1,
    )

    assert result.fill_count == 0
    assert result.book.exact_quantities[0] == Fraction(14)
    rejected = [
        event
        for event in result.order_events
        if event.event_type == "rejected" and event.order_id == closing.order_id
    ]
    assert not rejected, [event.reason for event in rejected]
    assert any(
        event.event_type == "eligible" and event.order_id == closing.order_id
        for event in result.order_events
    )
    assert any(
        order.order_id == closing.order_id for order in result.order_book.active_orders
    )


def test_close_reserves_exact_inventory_and_does_not_fund_a_later_opening() -> None:
    dataset = _two_symbol_market()
    config = replace(
        ExecutionCostConfig.zero(),
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        max_leverage=1.0,
    )
    executor = MarketExecutor(dataset, config)
    first_close = _market_order(
        dataset,
        executor,
        -8.0,
        target_identity="first-close",
        eligible_index=0,
        reduce_only=True,
    )
    second_close = _market_order(
        dataset,
        executor,
        -8.0,
        target_identity="second-close",
        eligible_index=0,
        reduce_only=True,
    )
    opening = _market_order(
        dataset,
        executor,
        1.0,
        target_identity="later-opening",
        eligible_index=1,
    )
    initial_book = BookState(
        quantities=np.array([10.0, 0.0]),
        cash=-100.0,
        mark_prices=dataset.close[0],
        peak_value=1_000.0,
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        initial_book,
        OrderBookState.empty().add(first_close, second_close, opening),
        (),
        start_index=0,
        bars=1,
    )

    close_ids = {first_close.order_id, second_close.order_id}
    rejected = {
        event.order_id: event.reason
        for event in result.order_events
        if event.event_type == "rejected"
    }
    assert (
        sum(order.order_id in close_ids for order in result.order_book.active_orders)
        == 1
    )
    assert sum(order_id in close_ids for order_id in rejected) == 1
    assert any(
        reason == "reduce_only_exceeds_position"
        for order_id, reason in rejected.items()
        if order_id in close_ids
    )
    assert rejected[opening.order_id] == "pretrade_leverage_exceeded"
    assert result.fill_count == 0


def test_unfilled_no_lot_full_close_reserves_its_exact_fraction() -> None:
    dataset = _two_symbol_market()
    config = replace(
        ExecutionCostConfig.zero(),
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        max_leverage=1.0,
    )
    executor = MarketExecutor(dataset, config)
    position = Fraction(1_000_000, 3)
    projected_position = project_quantity(position)
    closing_quantity = -projected_position
    remaining_fraction = position + exact_quantity(closing_quantity)
    assert remaining_fraction > 0
    second_close_quantity = -project_quantity(remaining_fraction)
    assert abs(second_close_quantity) > 1e-12
    closing = _market_order(
        dataset,
        executor,
        closing_quantity,
        target_identity="rational-full-close",
        eligible_index=0,
        reduce_only=True,
    )
    second_close = _market_order(
        dataset,
        executor,
        second_close_quantity,
        target_identity="rounded-rational-residual-close",
        eligible_index=1,
        reduce_only=True,
    )
    initial_book = BookState(
        quantities=np.array([projected_position, 0.0]),
        cash=30_000_000.0 - projected_position * 100.0,
        mark_prices=dataset.close[0],
        peak_value=30_000_000.0,
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
        _exact_quantities=(str(position), "0"),
    )

    result = executor.execute_orders(
        initial_book,
        OrderBookState.empty().add(closing, second_close),
        (),
        start_index=0,
        bars=1,
    )

    assert result.fill_count == 0
    assert any(
        order.order_id == closing.order_id for order in result.order_book.active_orders
    )
    assert not any(
        order.order_id == second_close.order_id
        for order in result.order_book.active_orders
    )
    rejected = next(
        event
        for event in result.order_events
        if event.order_id == second_close.order_id and event.event_type == "rejected"
    )
    assert rejected.reason == "reduce_only_no_position"


def test_lot_quantized_close_reserves_exact_accepted_lots() -> None:
    dataset = _two_symbol_market()
    lot_size = 3e-15
    config = replace(
        ExecutionCostConfig.zero(),
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        lot_size=lot_size,
        max_leverage=1.0,
    )
    executor = MarketExecutor(dataset, config)
    closing = _market_order(
        dataset,
        executor,
        -100_000.0,
        target_identity="lot-quantized-near-full-close",
        eligible_index=0,
        reduce_only=True,
    )
    second_close = _market_order(
        dataset,
        executor,
        -1e-11,
        target_identity="close-against-reserved-lots",
        eligible_index=1,
        reduce_only=True,
    )
    initial_book = BookState(
        quantities=np.array([100_000.0, 0.0]),
        cash=-1_000_000.0,
        mark_prices=dataset.close[0],
        peak_value=9_000_000.0,
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        initial_book,
        OrderBookState.empty().add(closing, second_close),
        (),
        start_index=0,
        bars=1,
    )

    assert result.fill_count == 0
    assert any(
        order.order_id == closing.order_id for order in result.order_book.active_orders
    )
    assert not any(
        order.order_id == second_close.order_id
        for order in result.order_book.active_orders
    )
    rejected = next(
        event
        for event in result.order_events
        if event.order_id == second_close.order_id and event.event_type == "rejected"
    )
    assert rejected.reason == "reduce_only_exceeds_position"
