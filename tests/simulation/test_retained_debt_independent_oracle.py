"""Independent accounting counterexample for opt-in native retained debt.

The oracle deliberately does not call the production cost/equity helpers;
it uses Decimal for a fixed artificial short and terminal mark. It cannot
authenticate any saved R2/R3 dataset or real execution venue.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from tests.simulation.test_insolvency_valuation import (
    cost,
    market,
    short_entry,
)
from trade_rl.simulation.execution import MarketExecutor


def test_terminal_short_gap_debt_matches_independent_decimal_account() -> None:
    opening_capital = Decimal("1000")
    short_units = Decimal("5")
    entry_price = Decimal("100")
    terminal_price = Decimal("400")
    fee_rate = Decimal("0.002")

    independent_fill_notional = short_units * entry_price
    independent_fee = independent_fill_notional * fee_rate
    independent_cash_after_fill = opening_capital + independent_fill_notional
    independent_cash_after_fill -= independent_fee
    independent_terminal_equity = (
        independent_cash_after_fill - short_units * terminal_price
    )
    independent_return = (
        independent_terminal_equity - opening_capital
    ) / opening_capital

    assert independent_terminal_equity == Decimal("-501")
    assert independent_return == Decimal("-1.501")

    executor = MarketExecutor(market(), cost(), insolvency_valuation="retain_debt")
    entry = short_entry(executor)
    assert entry.book.exact_quantities == (Fraction(-5),)
    assert entry.book.cash == float(independent_cash_after_fill)
    assert entry.book.total_cost == float(independent_fee)

    terminal = executor.execute_orders(
        entry.book, entry.order_book, (), start_index=1, bars=2
    )
    assert terminal.bars_advanced == 1
    assert terminal.next_index == 2
    assert terminal.book.insolvent
    assert terminal.book.exact_quantities == (Fraction(0),)
    assert terminal.book.cash == float(independent_terminal_equity)
    assert terminal.book.portfolio_value == float(independent_terminal_equity)
    assert terminal.book.total_cost == float(independent_fee)
    assert terminal.book.fill_count == 1
    assert terminal.order_book.active_orders == ()
    assert terminal.interval_cost == 0.0
    assert terminal.interval_funding == 0.0
    assert terminal.interval_borrow_cost == 0.0
    assert terminal.interval_net_return > -1.0
    assert (terminal.book.portfolio_value / float(opening_capital) - 1.0) == (
        pytest.approx(float(independent_return))
    )


def test_default_floor_and_explicit_retained_debt_have_different_terminal_books() -> None:
    default = MarketExecutor(market(), cost())
    opted_in = MarketExecutor(market(), cost(), insolvency_valuation="retain_debt")
    assert default.execution_policy_digest != opted_in.execution_policy_digest
    entry_default = short_entry(default)
    entry_retained = short_entry(opted_in)

    floored = default.execute_orders(
        entry_default.book, entry_default.order_book, (), start_index=1, bars=2
    )
    retained = opted_in.execute_orders(
        entry_retained.book,
        entry_retained.order_book,
        (),
        start_index=1,
        bars=2,
    )
    assert floored.book.portfolio_value == 0.0
    assert floored.bars_advanced == 2
    assert retained.book.portfolio_value == -501.0
    assert retained.bars_advanced == 1
