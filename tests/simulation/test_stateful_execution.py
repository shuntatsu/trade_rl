from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderIntent,
    OrderStatus,
    OrderType,
    TimeInForce,
)


def _market(**overrides: object) -> MarketDataset:
    n_bars = 6
    shape = (n_bars, 1)
    close = np.full(shape, 100.0)
    values: dict[str, object] = {
        "dataset_id": "d" * 64,
        "symbols": ("S0",),
        "timestamps": np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        "features": np.zeros((n_bars, 1, 1), dtype=np.float32),
        "global_features": np.zeros((n_bars, 1), dtype=np.float32),
        "open": close.copy(),
        "high": close + 10.0,
        "low": close - 10.0,
        "close": close.copy(),
        "volume": np.full(shape, 1_000.0),
        "funding_rate": np.zeros(shape),
        "tradable": np.ones(shape, dtype=np.bool_),
        "feature_available": np.ones((n_bars, 1, 1), dtype=np.bool_),
        "feature_names": ("ret",),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
    }
    values.update(overrides)
    return MarketDataset(**values)


def _executor(
    dataset: MarketDataset,
    **config_overrides: object,
) -> MarketExecutor:
    config = replace(
        ExecutionCostConfig.zero(),
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        **config_overrides,
    )
    return MarketExecutor(dataset, config)


def _intent(
    executor: MarketExecutor,
    quantity: float,
    *,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    stop_price: float | None = None,
    submit_index: int = 0,
    eligible_index: int = 1,
    expiry_index: int | None = None,
    time_in_force: TimeInForce = TimeInForce.GTC,
    target_identity: str = "target-1",
) -> OrderIntent:
    return OrderIntent.create(
        dataset_id=executor.dataset.dataset_id,
        target_identity=target_identity,
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=order_type,
        time_in_force=time_in_force,
        limit_price=limit_price,
        stop_price=stop_price,
        submit_index=submit_index,
        eligible_index=eligible_index,
        expiry_index=expiry_index,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )


def _zero_book(dataset: MarketDataset) -> BookState:
    return BookState.zero(
        1,
        1_000.0,
        dataset.close[0],
        dataset.resolved_array("contract_multipliers"),
    )


def test_partial_limit_fill_carries_to_next_processing_bar() -> None:
    volume = np.full((6, 1), 1_000.0)
    volume[1:3, 0] = 8.0
    dataset = _market(volume=volume)
    executor = _executor(dataset, max_participation_rate=1.0, lot_size=1.0)
    intent = _intent(
        executor,
        3.0,
        order_type=OrderType.LIMIT,
        limit_price=99.0,
    )

    first = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )
    assert first.order_book.active_orders[0].status is OrderStatus.PARTIALLY_FILLED
    assert first.order_book.active_orders[0].remaining_quantity == pytest.approx(1.0)

    second = executor.execute_orders(
        first.book,
        first.order_book,
        (),
        start_index=1,
        bars=1,
    )
    assert second.order_book.active_orders == ()
    assert second.order_book.terminal_orders[-1].status is OrderStatus.FILLED
    assert second.book.quantities[0] == pytest.approx(3.0)


def test_processing_bar_volume_not_preceding_bar_controls_capacity() -> None:
    volume = np.full((6, 1), 1_000.0)
    volume[0, 0] = 100.0
    volume[1, 0] = 1.0
    dataset = _market(volume=volume)
    executor = _executor(dataset, max_participation_rate=1.0)

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (_intent(executor, 10.0),),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities[0] == pytest.approx(1.0)
    assert result.order_book.active_orders[0].remaining_quantity == pytest.approx(9.0)
    assert result.capacity_evidence[0].processing_volume == pytest.approx(1.0)


def test_latency_waits_until_eligible_processing_bar() -> None:
    dataset = _market()
    executor = _executor(dataset, max_participation_rate=1.0)
    intent = _intent(executor, 1.0, eligible_index=2)

    first = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )
    assert first.book.quantities[0] == 0.0
    assert first.order_book.active_orders[0].status is OrderStatus.LATENCY_WAIT

    second = executor.execute_orders(
        first.book,
        first.order_book,
        (),
        start_index=1,
        bars=1,
    )
    assert second.book.quantities[0] == pytest.approx(1.0)
    assert second.order_book.active_orders == ()


def test_ioc_remainder_expires_after_first_eligible_attempt() -> None:
    high = np.full((6, 1), 101.0)
    low = np.full((6, 1), 100.0)
    dataset = _market(high=high, low=low)
    executor = _executor(dataset, max_participation_rate=1.0)
    intent = _intent(
        executor,
        1.0,
        order_type=OrderType.LIMIT,
        limit_price=99.0,
        time_in_force=TimeInForce.IOC,
    )

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities[0] == 0.0
    assert result.order_book.active_orders == ()
    assert result.order_book.terminal_orders[-1].status is OrderStatus.EXPIRED
    assert result.order_book.terminal_orders[-1].terminal_reason == "ioc_remainder"


def test_day_order_expires_after_expiry_bar() -> None:
    high = np.full((6, 1), 101.0)
    low = np.full((6, 1), 100.0)
    dataset = _market(high=high, low=low)
    executor = _executor(dataset, max_participation_rate=1.0)
    intent = _intent(
        executor,
        1.0,
        order_type=OrderType.LIMIT,
        limit_price=99.0,
        time_in_force=TimeInForce.DAY,
        expiry_index=1,
    )

    first = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )
    assert first.order_book.active_orders

    second = executor.execute_orders(
        first.book,
        first.order_book,
        (),
        start_index=1,
        bars=1,
    )
    assert second.order_book.active_orders == ()
    assert (
        second.order_book.terminal_orders[-1].terminal_reason == "time_in_force_expired"
    )


def test_triggered_stop_persists_and_executes_as_market_next_bar() -> None:
    open_prices = np.full((6, 1), 100.0)
    open_prices[2, 0] = 90.0
    high = np.full((6, 1), 110.0)
    low = np.full((6, 1), 95.0)
    low[2, 0] = 85.0
    volume = np.full((6, 1), 1_000.0)
    volume[1, 0] = 1.0
    dataset = _market(open=open_prices, high=high, low=low, volume=volume)
    executor = _executor(dataset, max_participation_rate=1.0)
    intent = _intent(
        executor,
        2.0,
        order_type=OrderType.STOP_MARKET,
        stop_price=105.0,
    )

    first = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )
    pending = first.order_book.active_orders[0]
    assert pending.status is OrderStatus.PARTIALLY_FILLED
    assert pending.trigger_index == 1

    second = executor.execute_orders(
        first.book,
        first.order_book,
        (),
        start_index=1,
        bars=1,
    )
    assert second.order_book.active_orders == ()
    assert second.book.quantities[0] == pytest.approx(2.0)
    assert any(event.trigger_segment == "open" for event in second.order_events)


def test_multiple_orders_share_one_symbol_capacity_pool() -> None:
    volume = np.full((6, 1), 1_000.0)
    volume[1, 0] = 5.0
    dataset = _market(volume=volume)
    executor = _executor(dataset, max_participation_rate=1.0)
    intents = (
        _intent(executor, 3.0, target_identity="a"),
        _intent(executor, 3.0, target_identity="b"),
    )

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        intents,
        start_index=0,
        bars=1,
    )

    assert result.book.quantities[0] == pytest.approx(5.0)
    assert result.filled_notional == pytest.approx(500.0)
    assert result.capacity_evidence[0].consumed_capacity_notional == pytest.approx(
        500.0
    )


def test_quote_notional_volume_is_not_multiplied_by_price_again() -> None:
    volume = np.full((6, 1), 1_000.0)
    volume[1, 0] = 100.0
    dataset = _market(
        volume=volume,
        volume_units=(VolumeUnit.QUOTE_NOTIONAL,),
    )
    executor = _executor(dataset, max_participation_rate=1.0)

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (_intent(executor, 10.0),),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities[0] == pytest.approx(1.0)
    assert result.capacity_evidence[0].market_notional == pytest.approx(100.0)


def test_same_inputs_replay_order_events_and_accounting_identically() -> None:
    volume = np.full((6, 1), 2.0)
    dataset = _market(volume=volume)

    def run_once():
        executor = _executor(dataset, max_participation_rate=1.0)
        return executor.execute_orders(
            _zero_book(dataset),
            OrderBookState.empty(),
            (_intent(executor, 3.0),),
            start_index=0,
            bars=2,
        )

    first = run_once()
    second = run_once()

    np.testing.assert_allclose(first.book.quantities, second.book.quantities)
    assert first.book.portfolio_value == pytest.approx(second.book.portfolio_value)
    assert [event.canonical_payload() for event in first.order_events] == [
        event.canonical_payload() for event in second.order_events
    ]


def test_stateful_result_reports_symbol_level_execution_observation_fields() -> None:
    volume = np.full((6, 1), 2.0)
    dataset = _market(volume=volume)
    executor = _executor(dataset, max_participation_rate=1.0)
    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (_intent(executor, 3.0),),
        start_index=0,
        bars=1,
    )

    assert result.requested_notional_by_symbol.tolist() == pytest.approx([300.0])
    assert result.filled_notional_by_symbol.tolist() == pytest.approx([200.0])
    assert result.participation_by_symbol.tolist() == pytest.approx([1.0])
    assert result.cost_by_symbol.tolist() == pytest.approx([0.0])
    assert np.isfinite(result.interval_gross_return)


def test_open_gap_refreshes_peak_before_projected_book_clone() -> None:
    open_price = np.full((6, 1), 100.0)
    close = np.full((6, 1), 100.0)
    open_price[1, 0] = 110.0
    close[1, 0] = 110.0
    dataset = _market(
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
    )
    executor = _executor(dataset)
    book = BookState.from_weights(
        weights=np.array((1.0,)),
        capital=1_000.0,
        prices=dataset.close[0],
        peak_value=1_000.0,
        max_gross=1.0,
    )

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    assert result.book.peak_value >= result.book.portfolio_value
