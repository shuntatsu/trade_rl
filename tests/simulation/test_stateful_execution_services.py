from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderIntent,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from trade_rl.simulation.stateful_runtime import StatefulExecutionRuntime


def _market(*, processing_volume: float = 1_000.0) -> MarketDataset:
    n_bars = 4
    shape = (n_bars, 1)
    close = np.full(shape, 100.0)
    volume = np.full(shape, 1_000.0)
    volume[1, 0] = processing_volume
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 10.0,
        low=close - 10.0,
        close=close.copy(),
        volume=volume,
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _executor(
    dataset: MarketDataset,
    *,
    partial_fill_carry: bool = True,
) -> MarketExecutor:
    config = replace(
        ExecutionCostConfig.zero(),
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=partial_fill_carry,
        max_participation_rate=1.0,
        lot_size=1.0,
    )
    return MarketExecutor(dataset, config)


def _intent(executor: MarketExecutor, quantity: float) -> OrderIntent:
    return OrderIntent.create(
        dataset_id=executor.dataset.dataset_id,
        target_identity="service-test",
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=quantity,
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


def _book(dataset: MarketDataset) -> BookState:
    return BookState.zero(
        1,
        1_000.0,
        dataset.close[0],
        dataset.resolved_array("contract_multipliers"),
    )


def test_runtime_owns_submission_event_sequence_and_result_aggregates() -> None:
    dataset = _market()
    executor = _executor(dataset)
    original = _book(dataset)
    runtime = StatefulExecutionRuntime.create(
        executor,
        original,
        OrderBookState.empty(),
    )

    runtime.submit_intents((_intent(executor, 1.0), _intent(executor, 2.0)))
    runtime.initialize_metrics()
    payload = runtime.result_payload(start_index=0, bars=1)

    assert original.quantities.tolist() == [0.0]
    assert [event.sequence for event in runtime.events] == [0, 1]
    assert [event.event_type for event in runtime.events] == ["submitted", "submitted"]
    assert runtime.requested_notional == pytest.approx(300.0)
    assert runtime.requested_by_symbol.tolist() == pytest.approx([300.0])
    assert payload["next_index"] == 1
    assert payload["fill_ratio"] == 0.0
    assert payload["order_events"] == tuple(runtime.events)


def test_runtime_rejects_non_positive_starting_equity_fail_closed() -> None:
    dataset = _market()
    executor = _executor(dataset)
    invalid_book = BookState(
        quantities=np.zeros(1, dtype=np.float64),
        cash=-1.0,
        mark_prices=dataset.close[0],
        peak_value=1_000.0,
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )
    runtime = StatefulExecutionRuntime.create(
        executor,
        invalid_book,
        OrderBookState.empty(),
    )

    with pytest.raises(
        ValueError,
        match="stateful execution requires positive starting equity",
    ):
        runtime.initialize_metrics()


def test_partial_fill_carry_disabled_expires_remainder_after_attempt() -> None:
    dataset = _market(processing_volume=1.0)
    executor = _executor(dataset, partial_fill_carry=False)

    result = executor.execute_orders(
        _book(dataset),
        OrderBookState.empty(),
        (_intent(executor, 3.0),),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities.tolist() == pytest.approx([1.0])
    assert result.order_book.active_orders == ()
    terminal = result.order_book.terminal_orders[-1]
    assert terminal.status is OrderStatus.EXPIRED
    assert terminal.terminal_reason == "partial_fill_carry_disabled"
    assert [event.event_type for event in result.order_events][-2:] == [
        "partial_fill",
        "expired",
    ]


def test_zero_processing_capacity_emits_no_fill_and_preserves_order() -> None:
    dataset = _market(processing_volume=0.0)
    executor = _executor(dataset)

    result = executor.execute_orders(
        _book(dataset),
        OrderBookState.empty(),
        (_intent(executor, 1.0),),
        start_index=0,
        bars=1,
    )

    assert result.book.quantities.tolist() == [0.0]
    assert len(result.order_book.active_orders) == 1
    assert result.order_book.active_orders[0].status is OrderStatus.ELIGIBLE
    assert result.capacity_evidence[0].consumed_capacity_notional == 0.0
    no_fill = [event for event in result.order_events if event.event_type == "no_fill"]
    assert len(no_fill) == 1
    assert no_fill[0].capacity_before == 0.0
    assert no_fill[0].capacity_after == 0.0
