"""Synthetic order-event accounting counterexamples for Issue #810.

This intentionally independent Decimal calculation uses recorded signed fills,
realized prices, and a declared fixed fee; it does not call the native book
or cost implementation to derive expected results. The forced flatten is not
a fill event, so terminal mark/termination still require separate evidence.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from tests.simulation.test_execution_v2 import two_symbol_market
from tests.simulation.test_insolvency_valuation import cost, market, short_entry
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderEvent,
    OrderIntent,
    OrderType,
    TimeInForce,
)


def _reconstruct_fills(
    events: tuple[OrderEvent, ...],
    *,
    initial_cash: Decimal,
    terminal_marks: tuple[Decimal, ...],
    fee_rate: Decimal,
    policy_digest: str,
    required_fills: int,
) -> tuple[Decimal, Decimal, tuple[Decimal, ...], int]:
    """Calculate pre-flatten equity from event fills, not recorded book totals."""
    cash = initial_cash
    quantities = [Decimal(0) for _ in terminal_marks]
    total_fee = Decimal(0)
    filled = 0
    for sequence, event in enumerate(events):
        if event.sequence != sequence:
            raise ValueError("event sequence is not contiguous")
        if event.execution_policy_digest != policy_digest:
            raise ValueError("event has an unrelated execution policy")
        if event.event_type not in {"filled", "partial_fill"}:
            if event.filled_quantity != 0.0 or event.filled_notional != 0.0:
                raise ValueError("non-fill event carries a fill")
            continue
        if event.execution_price is None:
            raise ValueError("fill event lacks execution price")
        if not 0 <= event.symbol_index < len(quantities):
            raise ValueError("fill symbol is outside the independent account")
        quantity = Decimal(str(event.filled_quantity))
        price = Decimal(str(event.execution_price))
        notional = quantity * price
        if notional == 0 or abs(notional) != Decimal(str(event.filled_notional)):
            raise ValueError("filled notional differs from signed quantity and price")
        fee = abs(notional) * fee_rate
        cash -= notional + fee
        quantities[event.symbol_index] += quantity
        total_fee += fee
        filled += 1
    equity = cash
    for quantity, mark in zip(quantities, terminal_marks, strict=True):
        equity += quantity * mark
    if filled != required_fills:
        raise ValueError("event fill count does not match preregistered count")
    return equity, total_fee, tuple(quantities), filled


def _verify_terminal_order_chain(
    *,
    events: tuple[OrderEvent, ...],
    intents: tuple[OrderIntent, ...],
    order_book: OrderBookState,
    terminal_index: int,
) -> None:
    """Match emitted order transitions to every terminal order record."""
    if order_book.active_orders:
        raise ValueError("terminal order archive still has active orders")
    declared = {intent.order_id: intent for intent in intents}
    terminal = {order.order_id: order for order in order_book.terminal_orders}
    if len(declared) != len(intents) or len(terminal) != len(intents):
        raise ValueError("terminal order roster mismatch")
    if set(declared) != set(terminal):
        raise ValueError("terminal order roster mismatch")

    last_by_order: dict[str, OrderEvent] = {}
    for position, event in enumerate(events):
        if event.sequence != position:
            raise ValueError("order event sequence mismatch")
        intent = declared.get(event.order_id)
        if intent is None:
            raise ValueError("unknown order in event chain")
        if (
            event.dataset_id != intent.dataset_id
            or event.symbol_index != intent.symbol_index
            or event.execution_policy_digest != intent.execution_policy_digest
        ):
            raise ValueError("event identity differs from declared order")
        if event.processing_index > terminal_index:
            raise ValueError("event follows economic termination")

        previous = last_by_order.get(event.order_id)
        if previous is None:
            if event.event_type != "submitted":
                raise ValueError("missing initial order submission")
        elif previous.new_status != event.previous_status:
            raise ValueError("order status chain is broken")
        elif previous.new_status.terminal:
            raise ValueError("order status chain is broken")
        last_by_order[event.order_id] = event

    for order_id, order in terminal.items():
        last = last_by_order.get(order_id)
        if last is None:
            raise ValueError("missing or mismatched terminal order event")
        if last.new_status != order.status:
            raise ValueError("missing or mismatched terminal order event")
        if last.remaining_quantity != order.remaining_quantity:
            raise ValueError("missing or mismatched terminal order event")
        if last.reason != order.terminal_reason:
            raise ValueError("missing or mismatched terminal order event")
        if order.terminal_reason == "economic_termination":
            if last.event_type != "cancelled":
                raise ValueError("economic termination cancellation mismatch")
            if last.processing_index != terminal_index:
                raise ValueError("economic termination cancellation mismatch")


def test_short_gap_raw_debt_is_reconstructed_from_actual_fill_event() -> None:
    data = market()
    executor = MarketExecutor(data, cost(), insolvency_valuation="retain_debt")
    entry = short_entry(executor)

    equity, fee, quantity, count = _reconstruct_fills(
        entry.order_events,
        initial_cash=Decimal("1000"),
        terminal_marks=(Decimal("400"),),
        fee_rate=Decimal("0.002"),
        policy_digest=executor.execution_policy_digest,
        required_fills=1,
    )
    assert fee == Decimal("1")
    assert quantity == (Decimal("-5"),)
    assert equity == Decimal("-501")
    assert count == 1

    terminal = executor.execute_orders(
        entry.book, entry.order_book, (), start_index=1, bars=2
    )
    assert terminal.book.portfolio_value == float(equity)
    assert terminal.book.cash == float(equity)
    assert terminal.book.exact_quantities == (Fraction(0),)
    assert terminal.book.total_cost == float(fee)
    assert not any(
        event.event_type in {"filled", "partial_fill"}
        for event in terminal.order_events
    )


def test_cross_symbol_debt_uses_only_first_realized_fill_in_event_ledger() -> None:
    data = two_symbol_market()
    assert tuple(data.resolved_array("contract_multipliers")) == (1.0, 1.0)
    executor = MarketExecutor(
        data,
        replace(
            ExecutionCostConfig.zero(),
            fee_rate=3.0,
            max_participation_rate=1.0,
        ),
        insolvency_valuation="retain_debt",
    )
    orders = tuple(
        OrderIntent.create(
            dataset_id=data.dataset_id,
            target_identity=f"event-reconcile-debt-{symbol}",
            execution_policy_digest=executor.execution_policy_digest,
            symbol_index=symbol,
            requested_quantity=5.0,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            limit_price=None,
            stop_price=None,
            submit_index=0,
            eligible_index=1,
            expiry_index=None,
            submission_reference_price=100.0,
            decision_equity=1000.0,
        )
        for symbol in range(2)
    )
    result = executor.execute_orders(
        BookState.zero(2, 1000.0, data.close[0]),
        OrderBookState.empty(),
        orders,
        start_index=0,
        bars=3,
    )

    equity, fee, quantities, count = _reconstruct_fills(
        result.order_events,
        initial_cash=Decimal("1000"),
        terminal_marks=(Decimal("100"), Decimal("100")),
        fee_rate=Decimal("3"),
        policy_digest=executor.execution_policy_digest,
        required_fills=1,
    )
    assert equity == Decimal("-500")
    assert fee == Decimal("1500")
    assert quantities == (Decimal("5"), Decimal("0"))
    assert count == 1
    assert result.fill_count == count
    assert result.interval_cost == float(fee)
    assert result.book.cash == float(equity)
    assert result.book.portfolio_value == float(equity)
    assert result.book.exact_quantities == (Fraction(0), Fraction(0))
    assert result.order_book.active_orders == ()

    _verify_terminal_order_chain(
        events=result.order_events,
        intents=orders,
        order_book=result.order_book,
        terminal_index=result.next_index,
    )
    cancelled = None
    for event in result.order_events:
        if event.event_type == "cancelled":
            if event.reason == "economic_termination":
                assert cancelled is None
                cancelled = event
    assert cancelled is not None

    missing = []
    for event in result.order_events:
        if event.sequence != cancelled.sequence:
            missing.append(replace(event, sequence=len(missing)))
    with pytest.raises(ValueError, match="terminal order event"):
        _verify_terminal_order_chain(
            events=tuple(missing),
            intents=orders,
            order_book=result.order_book,
            terminal_index=result.next_index,
        )

    corrupted = list(result.order_events)
    for changes, error in (
        ({"reason": "wrong-reason"}, "terminal order event"),
        ({"processing_index": result.next_index + 1}, "economic termination"),
        ({"previous_status": cancelled.new_status}, "status chain"),
    ):
        corrupted[cancelled.sequence] = replace(cancelled, **changes)
        with pytest.raises(ValueError, match=error):
            _verify_terminal_order_chain(
                events=tuple(corrupted),
                intents=orders,
                order_book=result.order_book,
                terminal_index=result.next_index,
            )


def test_independent_event_oracle_rejects_quantity_notional_or_identity_tamper() -> (
    None
):
    executor = MarketExecutor(market(), cost(), insolvency_valuation="retain_debt")
    entry = short_entry(executor)
    fills = [event for event in entry.order_events if event.filled_quantity != 0.0]
    assert len(fills) == 1
    original = fills[0]

    def replay(events: tuple[OrderEvent, ...]) -> None:
        _reconstruct_fills(
            events,
            initial_cash=Decimal("1000"),
            terminal_marks=(Decimal("400"),),
            fee_rate=Decimal("0.002"),
            policy_digest=executor.execution_policy_digest,
            required_fills=1,
        )

    tampered = list(entry.order_events)
    bad_notional = original.filled_notional + 1.0
    tampered[original.sequence] = replace(original, filled_notional=bad_notional)
    with pytest.raises(ValueError, match="notional"):
        replay(tuple(tampered))
    with pytest.raises(ValueError, match="sequence"):
        replay((*entry.order_events, original))

    # Removing a terminal fill can leave contiguous sequence numbers. The
    # preregistered one-fill count must still reject that missing evidence.
    missing = []
    for event in entry.order_events:
        if event.sequence != original.sequence:
            missing.append(event)
    for index, event in enumerate(missing):
        missing[index] = replace(event, sequence=index)
    with pytest.raises(ValueError, match="fill count"):
        replay(tuple(missing))

    other_policy = "f" * 64
    tampered[original.sequence] = replace(
        original, execution_policy_digest=other_policy
    )
    with pytest.raises(ValueError, match="execution policy"):
        replay(tuple(tampered))
