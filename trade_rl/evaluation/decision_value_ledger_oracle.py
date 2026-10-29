"""Independent interval-by-interval oracle for a *settled* native shared replay."""

from __future__ import annotations

import math
from fractions import Fraction

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import SharedCashPortfolioReplayResult


def _near(a: float, b: float, *, tol: float = 1e-5) -> bool:
    return math.isclose(a, b, abs_tol=tol, rel_tol=0.0)


def verify_paired_economic_ledger(
    result: SharedCashPortfolioReplayResult,
    dataset: MarketDataset,
    *,
    expected_intervals: int,
    initial_capital: float,
) -> bool:
    """Derive each cash/quantity/marked-equity state from raw fills and flows.

    Ledger final totals and BookState are *comparison targets*, never inputs
    to the independent reconstruction. This is an offline test, not a new book.
    """
    wrapped = result.ledger_evidence
    if wrapped is None or not math.isfinite(initial_capital) or initial_capital <= 0:
        return False
    ledger = wrapped.intervals
    intervals = ledger.intervals
    if not intervals or len(intervals) != expected_intervals:
        return False
    try:
        cash = Fraction(str(initial_capital))
        quantities = tuple(Fraction(0) for _ in dataset.symbols)
        cost = funding = borrow = Fraction(0)
        next_index = intervals[0].start_index
        previous_equity = initial_capital
        marks = dataset.resolved_array("mark_price")
        multipliers = dataset.resolved_array("contract_multipliers")
        for iv in intervals:
            if iv.start_index != next_index or iv.next_index != next_index + 1:
                return False
            if not 0 <= iv.next_index < dataset.n_bars:
                return False
            if not _near(iv.cash_before, float(cash)):
                return False
            if not _near(iv.portfolio_value_before, previous_equity):
                return False
            if any(
                abs(float(Fraction(q) - expected)) > 1e-9
                for q, expected in zip(
                    iv.exact_quantities_before, quantities, strict=True
                )
            ):
                return False
            if not all(
                _near(a - b, c)
                for a, b, c in (
                    (iv.total_cost_after, iv.total_cost_before, iv.interval_cost),
                    (iv.funding_pnl_after, iv.funding_pnl_before, iv.interval_funding),
                    (
                        iv.borrow_cost_after,
                        iv.borrow_cost_before,
                        iv.interval_borrow_cost,
                    ),
                )
            ):
                return False
            updated = list(quantities)
            for event in iv.order_events:
                if not event.filled_quantity:
                    continue
                if event.execution_price is None or event.execution_price <= 0:
                    return False
                idx = event.symbol_index
                if (
                    not 0 <= idx < len(updated)
                    or event.processing_index != iv.next_index
                ):
                    return False
                qty = Fraction(str(event.filled_quantity))
                cash -= (
                    qty
                    * Fraction(str(event.execution_price))
                    * Fraction(str(float(multipliers[idx])))
                )
                updated[idx] += qty
            quantities = tuple(updated)
            if any(
                abs(float(Fraction(q) - expected)) > 1e-9
                for q, expected in zip(
                    iv.exact_quantities_after, quantities, strict=True
                )
            ):
                return False
            this_cost = Fraction(iv.interval_cost)
            this_funding = Fraction(iv.interval_funding)
            this_borrow = Fraction(iv.interval_borrow_cost)
            cash += (
                -this_cost
                + this_funding
                - this_borrow
                + Fraction(iv.interval_dividend)
                + Fraction(iv.interval_cash_interest)
            )
            cost += this_cost
            funding += this_funding
            borrow += this_borrow
            expected_equity = float(cash) + sum(
                float(q) * float(marks[iv.next_index, j]) * float(multipliers[j])
                for j, q in enumerate(quantities)
            )
            if not _near(iv.cash_after, float(cash)):
                return False
            if not _near(iv.portfolio_value_after, expected_equity):
                return False
            previous_equity = iv.portfolio_value_after
            next_index = iv.next_index
        book = result.book
        return bool(
            all(abs(float(q)) <= 1e-9 for q in quantities)
            and all(Fraction(q) == 0 for q in ledger.terminal_exact_quantities)
            and all(Fraction(q) == 0 for q in book.exact_quantities)
            and ledger.termination_reason is None
            and book.termination_reason is None
            and not book.insolvent
            and all(
                _near(a, b)
                for a, b in (
                    (float(cash), ledger.final_cash),
                    (float(cash), ledger.final_portfolio_value),
                    (float(cash), float(book.cash)),
                    (float(cash), float(book.portfolio_value)),
                    (float(cost), ledger.final_total_cost),
                    (float(cost), float(book.total_cost)),
                    (float(funding), ledger.final_funding_pnl),
                    (float(funding), float(book.funding_pnl)),
                    (float(borrow), ledger.final_borrow_cost),
                    (float(borrow), float(book.borrow_cost)),
                )
            )
        )
    except (AttributeError, IndexError, ValueError, TypeError, OverflowError):
        return False
