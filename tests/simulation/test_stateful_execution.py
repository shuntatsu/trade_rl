from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from trade_rl.simulation.targets.execution import execute_target_statefully


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


def test_valid_sell_limit_is_not_rejected_by_lower_open_notional() -> None:
    shape = (6, 1)
    open_price = np.full(shape, 90.0)
    high = np.full(shape, 110.0)
    low = np.full(shape, 90.0)
    close = np.full(shape, 90.0)
    minimum = np.full(shape, 100.0)
    dataset = _market(
        open=open_price,
        high=high,
        low=low,
        close=close,
        minimum_notional=minimum,
    )
    executor = _executor(dataset, max_participation_rate=1.0)
    intent = OrderIntent.create(
        dataset_id=dataset.dataset_id,
        target_identity="sell-limit",
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=-1.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=110.0,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=90.0,
        decision_equity=1_000.0,
    )
    book = BookState(
        quantities=np.array([1.0]),
        cash=910.0,
        mark_prices=np.array([90.0]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1.0]),
    )

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities[0] == pytest.approx(0.0)
    filled = [event for event in result.order_events if event.event_type == "filled"]
    assert len(filled) == 1
    assert filled[0].execution_price == pytest.approx(110.0)
    assert all(
        event.reason != "below_minimum_notional"
        for event in result.order_events
        if event.event_type in {"rejected", "no_fill"}
    )


def test_marketable_sell_limit_uses_open_for_minimum_notional() -> None:
    shape = (6, 1)
    open_price = np.full(shape, 120.0)
    high = np.full(shape, 120.0)
    low = np.full(shape, 100.0)
    close = np.full(shape, 120.0)
    minimum = np.full(shape, 115.0)
    dataset = _market(
        open=open_price,
        high=high,
        low=low,
        close=close,
        minimum_notional=minimum,
    )
    executor = _executor(dataset, max_participation_rate=1.0)
    intent = OrderIntent.create(
        dataset_id=dataset.dataset_id,
        target_identity="marketable-sell-limit",
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=-1.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=110.0,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=120.0,
        decision_equity=1_000.0,
    )
    book = BookState(
        quantities=np.array([1.0]),
        cash=880.0,
        mark_prices=np.array([120.0]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1.0]),
    )

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities[0] == pytest.approx(0.0)
    filled = [event for event in result.order_events if event.event_type == "filled"]
    assert len(filled) == 1
    assert filled[0].execution_price == pytest.approx(120.0)
    assert all(
        event.reason != "below_minimum_notional"
        for event in result.order_events
        if event.event_type in {"rejected", "no_fill"}
    )


def test_newly_eligible_marketable_limit_uses_taker_costs() -> None:
    shape = (6, 1)
    open_price = np.full(shape, 100.0)
    dataset = _market(
        open=open_price,
        high=np.full(shape, 110.0),
        low=np.full(shape, 90.0),
        close=open_price.copy(),
    )
    executor = _executor(
        dataset,
        max_participation_rate=1.0,
        maker_fee_rate=0.001,
        taker_fee_rate=0.003,
        spread_rate=0.002,
    )
    intent = _intent(
        executor,
        1.0,
        order_type=OrderType.LIMIT,
        limit_price=110.0,
    )

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    filled = [event for event in result.order_events if event.event_type == "filled"]
    assert len(filled) == 1
    assert filled[0].trigger_segment == "open"
    assert filled[0].execution_price == pytest.approx(100.0)
    assert result.filled_notional == pytest.approx(100.0)
    assert result.interval_cost == pytest.approx(0.5)


def test_newly_eligible_resting_limit_touch_uses_maker_costs() -> None:
    shape = (6, 1)
    open_price = np.full(shape, 101.0)
    dataset = _market(
        open=open_price,
        high=np.full(shape, 101.0),
        low=np.full(shape, 99.0),
        close=np.full(shape, 100.0),
    )
    executor = _executor(
        dataset,
        max_participation_rate=1.0,
        maker_fee_rate=0.001,
        taker_fee_rate=0.003,
        spread_rate=0.002,
    )
    intent = _intent(
        executor,
        1.0,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
    )

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    filled = [event for event in result.order_events if event.event_type == "filled"]
    assert len(filled) == 1
    assert filled[0].trigger_segment != "open"
    assert filled[0].execution_price == pytest.approx(100.0)
    assert result.filled_notional == pytest.approx(100.0)
    assert result.interval_cost == pytest.approx(0.2)


def test_carried_resting_limit_remains_maker_when_next_open_crosses() -> None:
    shape = (6, 1)
    open_price = np.full(shape, 101.0)
    open_price[2, 0] = 99.0
    high = np.full(shape, 101.0)
    low = np.full(shape, 101.0)
    low[2, 0] = 99.0
    close = np.full(shape, 101.0)
    close[2, 0] = 100.0
    dataset = _market(
        open=open_price,
        high=high,
        low=low,
        close=close,
    )
    executor = _executor(
        dataset,
        max_participation_rate=1.0,
        maker_fee_rate=0.001,
        taker_fee_rate=0.003,
        spread_rate=0.002,
    )
    intent = _intent(
        executor,
        1.0,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
    )

    first = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )
    assert first.book.quantities[0] == pytest.approx(0.0)
    assert first.order_book.active_orders

    second = executor.execute_orders(
        first.book,
        first.order_book,
        (),
        start_index=1,
        bars=1,
    )

    filled = [event for event in second.order_events if event.event_type == "filled"]
    assert len(filled) == 1
    assert filled[0].trigger_segment == "open"
    assert filled[0].execution_price == pytest.approx(99.0)
    assert second.filled_notional == pytest.approx(99.0)
    assert second.interval_cost == pytest.approx(0.198)


def test_off_tick_buy_limit_is_rejected_before_trigger_rounding() -> None:
    shape = (6, 1)
    open_price = np.full(shape, 101.0)
    high = np.full(shape, 101.0)
    low = np.full(shape, 100.0)
    close = np.full(shape, 100.0)
    tick = np.full(shape, 0.5)
    dataset = _market(
        open=open_price,
        high=high,
        low=low,
        close=close,
        tick_size=tick,
    )
    executor = _executor(dataset, max_participation_rate=1.0)
    intent = _intent(
        executor,
        1.0,
        order_type=OrderType.LIMIT,
        limit_price=100.26,
    )

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities[0] == pytest.approx(0.0)
    assert not any(
        event.event_type in {"filled", "partial_fill"} for event in result.order_events
    )
    rejected = [
        event for event in result.order_events if event.event_type == "rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].reason == "price_not_on_tick"


def test_generated_bound_is_revalidated_against_eligible_tick_grid() -> None:
    shape = (6, 1)
    tick = np.full(shape, 0.5)
    tick[1, 0] = 2.0
    low = np.full(shape, 98.0)
    dataset = _market(
        tick_size=tick,
        low=low,
    )
    executor = _executor(
        dataset,
        max_participation_rate=1.0,
        order_type="limit",
        limit_offset_rate=0.007,
        order_latency_bars=1,
    )

    result = execute_target_statefully(
        executor,
        _zero_book(dataset),
        OrderBookState.empty(),
        np.array([0.5]),
        start_index=0,
        bars=1,
        target_identity="tick-rule-change",
    )

    assert result.book.quantities[0] == pytest.approx(0.0)
    assert result.order_book.active_orders == ()
    assert result.order_book.terminal_orders[-1].intent.limit_price == pytest.approx(
        99.0
    )
    rejected = [
        event for event in result.order_events if event.event_type == "rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].reason == "price_not_on_tick"


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


@pytest.mark.parametrize(
    ("volume_unit", "multiplier", "expected_quantity", "expected_notional"),
    [
        (VolumeUnit.BASE_ASSET, 1.0, 10.0, 900.0),
        (VolumeUnit.CONTRACTS, 2.0, 10.0, 1_800.0),
        (VolumeUnit.QUOTE_NOTIONAL, 1.0, 1_000.0 / 90.0, 1_000.0),
    ],
)
def test_capacity_respects_native_volume_unit_when_fill_price_differs_from_open(
    volume_unit: VolumeUnit,
    multiplier: float,
    expected_quantity: float,
    expected_notional: float,
) -> None:
    shape = (6, 1)
    open_price = np.full(shape, 100.0)
    high = np.full(shape, 100.0)
    low = np.full(shape, 90.0)
    close = np.full(shape, 100.0)
    volume = np.full(shape, 1_000.0)
    volume[1, 0] = 10.0 if volume_unit is not VolumeUnit.QUOTE_NOTIONAL else 1_000.0
    dataset = _market(
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        volume_units=(volume_unit,),
        contract_multipliers=np.array([multiplier]),
    )
    executor = _executor(
        dataset,
        max_participation_rate=1.0,
        max_leverage=5.0,
        trigger_volume_fractions=(1.0, 1.0, 1.0, 1.0),
    )
    intent = _intent(
        executor,
        20.0,
        order_type=OrderType.LIMIT,
        limit_price=90.0,
    )

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities[0] == pytest.approx(expected_quantity)
    assert result.filled_notional == pytest.approx(expected_notional)
    assert result.capacity_evidence[0].processing_volume == pytest.approx(volume[1, 0])
    assert result.capacity_evidence[0].market_notional == pytest.approx(
        1_000.0 if volume_unit is not VolumeUnit.CONTRACTS else 2_000.0
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


def test_fill_ratio_uses_submission_reference_basis() -> None:
    shape = (6, 1)
    open_price = np.full(shape, 100.0)
    close = np.full(shape, 100.0)
    high = np.full(shape, 120.0)
    low = np.full(shape, 90.0)
    volume = np.full(shape, 1_000.0)
    volume[1, 0] = 10.8
    dataset = _market(
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    executor = _executor(
        dataset,
        max_participation_rate=1.0,
        lot_size=1.0,
        trigger_volume_fractions=(1.0, 1.0, 1.0, 1.0),
    )
    intent = _intent(
        executor,
        10.0,
        order_type=OrderType.STOP_MARKET,
        stop_price=105.0,
    )

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    assert result.order_book.active_orders[0].remaining_quantity == pytest.approx(1.0)
    assert result.requested_notional == pytest.approx(1_000.0)
    assert result.filled_notional == pytest.approx(1_080.0)
    assert result.requested_turnover == pytest.approx(1.0)
    assert result.filled_turnover == pytest.approx(1.08)
    assert result.fill_ratio == pytest.approx(0.9)
    assert result.unfilled_turnover == pytest.approx(0.1)


def test_interval_gross_return_uses_actual_fill_price_on_observed_path() -> None:
    open_price = np.full((6, 1), 100.0)
    close = np.full((6, 1), 100.0)
    close[1, 0] = 110.0
    high = np.full((6, 1), 110.0)
    low = np.full((6, 1), 90.0)
    dataset = _market(open=open_price, high=high, low=low, close=close)
    executor = _executor(dataset, max_participation_rate=1.0)
    intent = _intent(
        executor,
        5.0,
        order_type=OrderType.STOP_MARKET,
        stop_price=105.0,
    )

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    fill_events = [
        event
        for event in result.order_events
        if event.event_type in {"filled", "partial_fill"}
    ]
    assert len(fill_events) == 1
    assert fill_events[0].execution_price == pytest.approx(110.0)
    assert result.interval_cost == 0.0
    assert result.interval_funding == 0.0
    assert result.interval_borrow_cost == 0.0
    assert result.interval_dividend == 0.0
    assert result.interval_cash_interest == 0.0
    assert result.interval_net_return == pytest.approx(0.0)
    assert result.interval_gross_return == pytest.approx(result.interval_net_return)


def test_interval_gross_return_reconciles_explicit_execution_cost() -> None:
    close = np.full((6, 1), 100.0)
    close[1, 0] = 110.0
    dataset = _market(close=close)
    executor = _executor(
        dataset,
        max_participation_rate=1.0,
        fee_rate=0.01,
    )

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (_intent(executor, 5.0),),
        start_index=0,
        bars=1,
    )

    assert result.interval_cost == pytest.approx(5.0)
    assert result.interval_funding == 0.0
    assert result.interval_borrow_cost == 0.0
    assert result.interval_dividend == 0.0
    assert result.interval_cash_interest == 0.0
    assert result.interval_net_return == pytest.approx(0.045)
    assert result.interval_gross_return == pytest.approx(0.05)
    gross_value = (
        result.book.portfolio_value
        + result.interval_cost
        - result.interval_funding
        + result.interval_borrow_cost
        - result.interval_dividend
        - result.interval_cash_interest
    )
    assert gross_value / 1_000.0 - 1.0 == pytest.approx(result.interval_gross_return)


def test_interval_gross_return_removes_signed_funding_from_same_fill_path() -> None:
    shape = (6, 1)
    funding_rate = np.zeros(shape)
    funding_rate[1, 0] = 0.001
    funding_due = np.zeros(shape, dtype=np.bool_)
    funding_due[1, 0] = True
    dataset = _market(
        funding_rate=funding_rate,
        funding_due=funding_due,
    )
    executor = _executor(dataset, max_participation_rate=1.0)

    result = executor.execute_orders(
        _zero_book(dataset),
        OrderBookState.empty(),
        (_intent(executor, 10.0),),
        start_index=0,
        bars=1,
    )

    assert result.interval_cost == 0.0
    assert result.interval_funding == pytest.approx(-1.0)
    assert result.interval_net_return == pytest.approx(-0.001)
    assert result.interval_gross_return == pytest.approx(0.0)
    gross_value = (
        result.book.portfolio_value
        + result.interval_cost
        - result.interval_funding
        + result.interval_borrow_cost
        - result.interval_dividend
        - result.interval_cash_interest
    )
    assert gross_value / 1_000.0 - 1.0 == pytest.approx(result.interval_gross_return)


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


def test_cancel_and_replace_emits_cancellation_before_replacement_submission() -> None:
    dataset = _market(volume=np.zeros((6, 1), dtype=np.float64))
    executor = _executor(dataset, max_participation_rate=1.0)

    first = execute_target_statefully(
        executor,
        _zero_book(dataset),
        OrderBookState.empty(),
        np.array((0.5,), dtype=np.float64),
        start_index=0,
        bars=1,
        target_identity="target-a",
    )
    replaced_order = first.order_book.active_orders[0]

    second = execute_target_statefully(
        executor,
        first.book,
        first.order_book,
        np.array((0.2,), dtype=np.float64),
        start_index=1,
        bars=1,
        target_identity="target-b",
    )

    assert second.order_events[0].event_type == "cancelled"
    assert second.order_events[0].order_id == replaced_order.order_id
    assert second.order_events[0].new_status is OrderStatus.CANCELLED
    assert second.order_events[1].event_type == "submitted"
    assert second.order_events[1].replaced_order_id == replaced_order.order_id
    assert tuple(event.sequence for event in second.order_events) == tuple(
        range(len(second.order_events))
    )
