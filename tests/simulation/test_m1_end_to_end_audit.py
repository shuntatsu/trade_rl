"""Independent synthetic M1 order-capacity-ledger and failed-invocation oracles."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    TimeInForce,
)


def _market() -> MarketDataset:
    bars = 4
    close = np.tile(np.array([100.0, 50.0]), (bars, 1))
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("AAA", "BBB"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(bars) * np.timedelta64(1, "h"),
        features=np.zeros((bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.tile(np.array([2.0, 3.0]), (bars, 1)),
        funding_rate=np.zeros((bars, 2)),
        tradable=np.ones((bars, 2), dtype=np.bool_),
        feature_available=np.ones((bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _executor(dataset: MarketDataset, *, slippage: float = 0.0) -> MarketExecutor:
    cost = replace(
        ExecutionCostConfig.zero(),
        fee_rate=0.01 if slippage == 0.0 else 0.0,
        slippage_std=slippage,
        random_seed=7,
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        path_mode="conservative",
    )
    return MarketExecutor(dataset, cost)


def _intents(
    executor: MarketExecutor, targets: tuple[float, float]
) -> tuple[OrderIntent, ...]:
    return tuple(
        OrderIntent.create(
            dataset_id=executor.dataset.dataset_id,
            target_identity="independent-m1-two-symbol",
            execution_policy_digest=executor.execution_policy_digest,
            symbol_index=index,
            requested_quantity=quantity,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            limit_price=None,
            stop_price=None,
            submit_index=0,
            eligible_index=1,
            expiry_index=None,
            submission_reference_price=float(executor.dataset.close[0, index]),
            decision_equity=1_000.0,
        )
        for index, quantity in enumerate(targets)
    )


def _manual_scheduled_book() -> tuple[Decimal, tuple[Decimal, ...], Decimal]:
    # The oracle knows only the independent constant-price venue fixture:
    # two processing bars, per-symbol capacities (2, 3), and fee 1%.
    prices = (Decimal("100"), Decimal("50"))
    targets = (Decimal("4"), Decimal("-5"))
    capacities = (Decimal("2"), Decimal("3"))
    quantities = [Decimal(0), Decimal(0)]
    cash = Decimal("1000")
    fees = Decimal(0)
    for _ in range(2):
        for symbol in range(2):
            remaining = targets[symbol] - quantities[symbol]
            fill = min(abs(remaining), capacities[symbol]).copy_sign(remaining)
            quantity_notional = fill * prices[symbol]
            charged = abs(quantity_notional) * Decimal("0.01")
            cash -= quantity_notional + charged
            fees += charged
            quantities[symbol] += fill
    return cash, tuple(quantities), fees


def test_two_symbol_partial_fills_and_cash_match_independent_decimal_oracle() -> None:
    dataset = _market()
    executor = _executor(dataset)
    book = BookState.zero(2, 1_000.0, dataset.close[0])
    order_book = OrderBookState.empty()
    intents = _intents(executor, (4.0, -5.0))

    result = executor.execute_orders(book, order_book, intents, start_index=0, bars=2)
    manual_cash, manual_quantities, manual_fees = _manual_scheduled_book()
    manual_equity = manual_cash + sum(
        (
            quantity * price
            for quantity, price in zip(
                manual_quantities, (Decimal("100"), Decimal("50")), strict=True
            )
        ),
        Decimal(0),
    )
    assert result.book.cash == float(manual_cash)
    np.testing.assert_array_equal(
        result.book.quantities, [float(value) for value in manual_quantities]
    )
    assert result.book.portfolio_value == float(manual_equity)
    assert result.book.total_cost == float(manual_fees)
    assert result.interval_cost == float(manual_fees)
    assert result.fill_count == 4
    assert result.order_book.active_orders == ()
    assert result.requested_notional == 650.0
    assert result.filled_notional == 650.0
    assert [event.sequence for event in result.order_events] == list(
        range(len(result.order_events))
    )
    fills = [
        event
        for event in result.order_events
        if event.event_type in ("filled", "partial_fill")
    ]
    assert [
        (event.processing_index, event.symbol_index, event.filled_quantity)
        for event in fills
    ] == [(1, 0, 2.0), (1, 1, -3.0), (2, 0, 2.0), (2, 1, -2.0)]
    for event in fills:
        assert event.filled_notional == abs(event.filled_quantity) * (
            100.0 if event.symbol_index == 0 else 50.0
        )
        assert event.capacity_before - event.capacity_after == event.filled_notional
    assert (
        sum(cap.consumed_capacity_notional for cap in result.capacity_evidence) == 650.0
    )
    assert book.cash == 1_000.0
    np.testing.assert_array_equal(book.quantities, [0.0, 0.0])
    assert order_book == OrderBookState.empty()

    replay_executor = _executor(dataset)
    replay = replay_executor.execute_orders(
        book,
        order_book,
        _intents(replay_executor, (4.0, -5.0)),
        start_index=0,
        bars=2,
    )
    assert replay.book.cash == result.book.cash
    assert [event.canonical_payload() for event in replay.order_events] == [
        event.canonical_payload() for event in result.order_events
    ]


@pytest.mark.parametrize("failure_fill", (1, 2))
def test_failed_stochastic_fill_does_not_change_retry_rng_or_inputs(
    monkeypatch: pytest.MonkeyPatch, failure_fill: int
) -> None:
    dataset = _market()
    executor = _executor(dataset, slippage=0.02)
    control = _executor(dataset, slippage=0.02)
    book = BookState.zero(2, 1_000.0, dataset.close[0])
    empty = OrderBookState.empty()
    intents = _intents(executor, (1.0, -1.0))
    control_result = control.execute_orders(
        book, empty, _intents(control, (1.0, -1.0)), start_index=0, bars=1
    )
    rng_before = executor._rng.bit_generator.state

    original_fill = BookState.execute_fill
    calls = 0
    with monkeypatch.context() as patch:

        def failure(self: BookState, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == failure_fill:
                raise RuntimeError("injected persistence failure after stochastic fee")
            original_fill(self, **kwargs)  # type: ignore[arg-type]

        patch.setattr(BookState, "execute_fill", failure)
        with pytest.raises(RuntimeError, match="injected persistence failure"):
            executor.execute_orders(book, empty, intents, start_index=0, bars=1)

    assert book.cash == 1_000.0
    np.testing.assert_array_equal(book.quantities, [0.0, 0.0])
    assert empty == OrderBookState.empty()
    assert executor._rng.bit_generator.state == rng_before

    retry = executor.execute_orders(book, empty, intents, start_index=0, bars=1)
    assert retry.interval_cost == control_result.interval_cost
    assert retry.book.cash == control_result.book.cash
    assert [event.canonical_payload() for event in retry.order_events] == [
        event.canonical_payload() for event in control_result.order_events
    ]
