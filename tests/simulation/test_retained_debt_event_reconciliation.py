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
            raise ValueError("recorded filled notional disagrees with quantity and price")
        fee = abs(notional) * fee_rate
        cash -= notional + fee
        quantities[event.symbol_index] += quantity
        total_fee += fee
        filled += 1
    equity = cash + sum(
        (quantity * mark for quantity, mark in zip(
            quantities, terminal_marks, strict=True
        )),
        Decimal(0),
    )
    return equity, total_fee, tuple(quantities), filled


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


def test_independent_event_oracle_rejects_quantity_notional_or_identity_tamper() -> None:
    executor = MarketExecutor(market(), cost(), insolvency_valuation="retain_debt")
    entry = short_entry(executor)
    fills = tuple(
        event
        for event in entry.order_events
        if event.event_type in {"filled", "partial_fill"}
    )
    assert len(fills) == 1
    original = fills[0]

    def replay(events: tuple[OrderEvent, ...]) -> None:
        _reconstruct_fills(
            events,
            initial_cash=Decimal("1000"),
            terminal_marks=(Decimal("400"),),
            fee_rate=Decimal("0.002"),
            policy_digest=executor.execution_policy_digest,
        )

    with pytest.raises(ValueError, match="notional"):
        replay(
            tuple(
                replace(event, filled_notional=event.filled_notional + 1.0)
                if event.sequence == original.sequence else event
                for event in entry.order_events
            )
        )
    with pytest.raises(ValueError, match="sequence"):
        replay((*entry.order_events, original))
    with pytest.raises(ValueError, match="execution policy"):
        replay(
            tuple(
                replace(event, execution_policy_digest="f" * 64)
                if event.sequence == original.sequence else event
                for event in entry.order_events
            )
        )
