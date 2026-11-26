"""Independent native terminal ledger reconciliation, including valued venue dust."""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Real
from typing import cast

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import SharedCashPortfolioReplayResult
from trade_rl.evaluation.terminal_settlement import terminal_dust
from trade_rl.simulation.execution import ExecutionCostConfig


def reconcile_ledger(
    result: SharedCashPortfolioReplayResult,
    dataset: MarketDataset,
    initial_capital: float,
    execution_cost: ExecutionCostConfig | None = None,
    *,
    reconciliation_tolerance: float = 1e-4,
) -> dict[str, object]:
    """Independently rebuild terminal cash/equity from raw fills and ledger flows.

    ``terminal = initial + sum(-qty * price * multiplier over fills) - costs
    + funding - borrow + dividends + cash interest`` where fills come from the
    raw order events, never from the book's own cash. A flat terminal book makes
    cash and equity coincide.

    The engine keeps the venue minimum notional for every order, so a holding that
    decayed below it can stay open at the end. Such residuals are reported as
    ``terminal_dust`` (symbol, exact quantity, notional, reason
    ``below_minimum_notional``) and valued at mark minus the modelled exit cost
    from ``execution_cost``. ``terminal_flat`` stays false, ``terminal_dust_only``
    is true when every residual is such dust, and the check passes when cash and
    the dust-adjusted equity reconcile. Any other residual still fails; without
    ``execution_cost`` no residual can be valued as dust.
    """

    # The extracted shared contract no longer receives a validated study protocol.
    # An infinite tolerance would make a corrupted ledger appear reconciled.
    amounts: tuple[tuple[str, object, bool], ...] = (
        ("initial_capital", initial_capital, True),
        ("reconciliation_tolerance", reconciliation_tolerance, False),
    )
    for field, value, positive in amounts:
        try:
            valid = (
                not isinstance(value, bool)
                and isinstance(value, Real)
                and math.isfinite(float(value))
                and (float(value) > 0 if positive else float(value) >= 0)
            )
        except (OverflowError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(
                f"{field} must be finite and {'positive' if positive else 'nonnegative'}"
            )

    wrapper = result.ledger_evidence
    if wrapper is None:
        raise ValueError("replay result carries no ledger evidence")
    ledger = wrapper.intervals
    intervals = ledger.intervals
    if not intervals:
        raise ValueError("replay result carries no ledger intervals")
    multipliers = dataset.resolved_array("contract_multipliers")
    initial = Fraction(str(initial_capital))
    signed_fill_cash = Fraction(0)
    filled_notional = Fraction(0)
    n_fills = 0
    cost = funding = borrow = dividend = interest = Fraction(0)
    chained = True
    previous_next: int | None = None
    previous_after: float | None = None
    tolerance = reconciliation_tolerance
    continuity_gap = 0.0
    for interval in intervals:
        if previous_next is not None and interval.start_index != previous_next:
            chained = False
        if previous_after is not None:
            continuity_gap = max(
                continuity_gap, abs(interval.portfolio_value_before - previous_after)
            )
        previous_next = interval.next_index
        previous_after = interval.portfolio_value_after
        for event in interval.order_events:
            if event.filled_quantity == 0.0:
                continue
            if event.execution_price is None:
                raise ValueError("fill without an execution price")
            quantity = Fraction(str(event.filled_quantity))
            price = Fraction(str(event.execution_price))
            multiplier = Fraction(str(float(multipliers[event.symbol_index])))
            signed_fill_cash -= quantity * price * multiplier
            filled_notional += abs(quantity) * price * multiplier
            n_fills += 1
        cost += Fraction(interval.interval_cost)
        funding += Fraction(interval.interval_funding)
        borrow += Fraction(interval.interval_borrow_cost)
        dividend += Fraction(interval.interval_dividend)
        interest += Fraction(interval.interval_cash_interest)
    expected = (
        initial + signed_fill_cash - cost + funding - borrow + dividend + interest
    )
    terminal_flat = all(
        Fraction(value) == 0 for value in ledger.terminal_exact_quantities
    )
    final_equity = ledger.final_portfolio_value
    final_marks = dataset.resolved_array("mark_price")[intervals[-1].next_index]
    residual_quantities = {
        dataset.symbols[index]: str(value)
        for index, value in enumerate(ledger.terminal_exact_quantities)
        if Fraction(value) != 0
    }
    residual_notional = sum(
        abs(float(Fraction(value)))
        * float(final_marks[index])
        * float(multipliers[index])
        for index, value in enumerate(ledger.terminal_exact_quantities)
    )
    dust, non_dust, _ = terminal_dust(ledger, dataset, execution_cost, intervals[-1])
    terminal_dust_only = bool(not terminal_flat and dust and not non_dust)
    dust_mark_value = Fraction(0)
    dust_exit_cost = 0.0
    for item in dust:
        dust_mark_value += Fraction(cast(float, item["mark_value_usdt"]))
        dust_exit_cost += cast(float, item["exit_cost_usdt"])
    cash_residual = float(Fraction(ledger.final_cash) - expected)
    equity_residual = float(Fraction(final_equity) - expected)
    # Equity must equal independent cash plus the dust at mark (its exit cost is
    # applied to the reported valuation, not credited by the ledger).
    dust_adjusted_equity_residual = float(
        Fraction(final_equity) - expected - dust_mark_value
    )
    totals_residual = max(
        abs(float(cost) - ledger.final_total_cost),
        abs(float(funding) - ledger.final_funding_pnl),
        abs(float(borrow) - ledger.final_borrow_cost),
    )
    passed = bool(
        (terminal_flat or terminal_dust_only)
        and chained
        and ledger.termination_reason is None
        and abs(cash_residual) <= tolerance
        and abs(equity_residual if terminal_flat else dust_adjusted_equity_residual)
        <= tolerance
        and continuity_gap <= tolerance
        and totals_residual <= tolerance
    )
    return {
        "passed": passed,
        "terminal_flat": terminal_flat,
        "terminal_dust_only": terminal_dust_only,
        "terminal_dust": dust,
        "terminal_dust_exit_cost_usdt": dust_exit_cost,
        "terminal_equity_dust_adjusted": float(expected)
        + float(dust_mark_value)
        - dust_exit_cost,
        "dust_adjusted_equity_residual": dust_adjusted_equity_residual,
        "terminal_residual_quantities": residual_quantities,
        "terminal_residual_notional_usdt": float(residual_notional),
        "intervals_chained": chained,
        "termination_reason": ledger.termination_reason,
        "n_intervals": len(intervals),
        "n_fills": n_fills,
        "signed_fill_cash": float(signed_fill_cash),
        "filled_notional": float(filled_notional),
        "execution_costs": float(cost),
        "funding_pnl": float(funding),
        "borrow_cost": float(borrow),
        "dividend_pnl": float(dividend),
        "cash_interest_pnl": float(interest),
        "independent_terminal_equity": float(expected),
        "ledger_final_cash": ledger.final_cash,
        "ledger_final_equity": final_equity,
        "cash_residual": cash_residual,
        "equity_residual": equity_residual,
        "continuity_gap": continuity_gap,
        "ledger_total_residual": totals_residual,
    }
