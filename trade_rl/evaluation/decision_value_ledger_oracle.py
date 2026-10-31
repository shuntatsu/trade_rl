"""Source-backed, offline ledger reconstruction for *settled* TD8 replay.

Cash flow, fill quantities, market marks, and final equity are reconstructed
independently of BookState. Fee/funding/borrow/dividend/interest amounts are
read from the observer interval; their economic pricing is not independently
re-derived from contractual market rules. Fill prices are checked against a
conservative source OHLC plus one tick envelope; exact limit/stop trigger
semantics remain outside this checker. Per-interval turnover and a necessary
bar-mark drawdown bound are independently reconstructed. This is a software
diagnostic, not a second authoritative book.
"""

from __future__ import annotations

import math
from fractions import Fraction

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import SharedCashPortfolioReplayResult
from trade_rl.simulation.execution import ExecutionCostConfig


def _near(a: float, b: float, *, tol: float = 1e-5) -> bool:
    return math.isclose(a, b, abs_tol=tol, rel_tol=0.0)


def verify_paired_economic_ledger(
    result: SharedCashPortfolioReplayResult,
    dataset: MarketDataset,
    *,
    expected_intervals: int,
    initial_capital: float,
    expected_start_index: int | None = None,
    execution_cost: ExecutionCostConfig | None = None,
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
    if (
        expected_start_index is not None
        and getattr(ledger, "start_index", None) != expected_start_index
    ):
        return False
    if (
        getattr(ledger, "dataset_id", None) != dataset.dataset_id
        or getattr(ledger, "start_index", None) != intervals[0].start_index
        or getattr(ledger, "stop_index", None) != intervals[-1].next_index
        or getattr(ledger, "active_order_remainders", None) != ()
        or (
            execution_cost is not None
            and ledger.execution_policy_digest != execution_cost.execution_policy_digest
        )
    ):
        return False
    try:
        cash = Fraction(str(initial_capital))
        quantities = tuple(Fraction(0) for _ in dataset.symbols)
        cost = funding = borrow = Fraction(0)
        turnover = 0.0
        peak = float(initial_capital)
        minimum_observed_drawdown = 0.0
        next_index = intervals[0].start_index
        previous_equity = initial_capital
        marks = dataset.resolved_array("mark_price")
        multipliers = dataset.resolved_array("contract_multipliers")
        stamps = dataset.timestamps.astype("datetime64[ns]").astype("int64")
        splits = dataset.resolved_array("split_factor")
        lows = dataset.low
        highs = dataset.high
        tick_sizes = dataset.resolved_array("tick_size")
        configured_tick = 0.0 if execution_cost is None else execution_cost.tick_size
        for iv in intervals:
            if iv.start_index != next_index or iv.next_index != next_index + 1:
                return False
            if not 0 <= iv.next_index < dataset.n_bars:
                return False
            # This diagnostic does not reconstruct corporate-action share
            # conversions; reject rather than silently approving unmodeled
            # split adjustments (including on an empty position).
            if any(float(factor) != 1.0 for factor in splits[iv.next_index]):
                return False
            if not _near(iv.cash_before, float(cash)):
                return False
            if not _near(iv.portfolio_value_before, previous_equity):
                return False
            if (
                not _near(iv.turnover_total_before, turnover)
                or iv.max_drawdown_before + 1e-8 < minimum_observed_drawdown
            ):
                return False
            if iv.termination_reason is not None:
                return False
            if not all(
                _near(a, b)
                for a, b in (
                    (iv.total_cost_before, float(cost)),
                    (iv.funding_pnl_before, float(funding)),
                    (iv.borrow_cost_before, float(borrow)),
                )
            ):
                return False
            if any(
                abs(float(Fraction(q) - expected)) > 1e-9
                for q, expected in zip(
                    iv.exact_quantities_before, quantities, strict=True
                )
            ):
                return False
            if iv.interval_cost < 0 or iv.interval_borrow_cost < 0:
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
            interval_notional = 0.0
            for sequence, event in enumerate(iv.order_events):
                if event.sequence != sequence:
                    return False
                idx = event.symbol_index
                # Submission/cancellation can happen at the decision bar while
                # fills are processed on the next bar. Every event, including a
                # zero-fill event, must bind to the same source and native clock.
                if (
                    not 0 <= idx < len(updated)
                    or event.processing_index not in (iv.start_index, iv.next_index)
                    or event.timestamp_ns != int(stamps[event.processing_index])
                    or event.dataset_id != dataset.dataset_id
                    or event.execution_policy_digest != ledger.execution_policy_digest
                ):
                    return False
                if not event.filled_quantity:
                    if (
                        abs(event.filled_notional) > 1e-9
                        or event.event_type in ("filled", "partial_fill")
                        or event.execution_price is not None
                    ):
                        return False
                    continue
                if (
                    event.execution_price is None
                    or event.execution_price <= 0
                    or event.processing_index != iv.next_index
                    or event.event_type not in ("filled", "partial_fill")
                ):
                    return False
                # A native trigger uses a source OHLC path. Directional price
                # rounding can move the result by one tick outside that path,
                # but an order cannot fill at an arbitrary ledger-only price.
                # This envelope is necessary (not sufficient) for feasible
                # stop/limit/path execution and does not reproduce the executor.
                tick = max(float(tick_sizes[iv.next_index, idx]), configured_tick)
                tolerance = max(1e-9, abs(event.execution_price) * 1e-12)
                if not (
                    float(lows[iv.next_index, idx]) - tick - tolerance
                    <= event.execution_price
                    <= float(highs[iv.next_index, idx]) + tick + tolerance
                ):
                    return False
                qty = Fraction(str(event.filled_quantity))
                fill_notional = (
                    abs(qty)
                    * Fraction(str(event.execution_price))
                    * Fraction(str(float(multipliers[idx])))
                )
                if not _near(float(fill_notional), event.filled_notional):
                    return False
                interval_notional += float(fill_notional)
                cash -= (
                    qty
                    * Fraction(str(event.execution_price))
                    * Fraction(str(float(multipliers[idx])))
                )
                updated[idx] += qty
            quantities = tuple(updated)
            if previous_equity <= 0.0:
                return False
            turnover += interval_notional / previous_equity
            if not _near(iv.turnover_total_after, turnover):
                return False
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
            if not all(
                _near(a, b)
                for a, b in (
                    (iv.total_cost_after, float(cost)),
                    (iv.funding_pnl_after, float(funding)),
                    (iv.borrow_cost_after, float(borrow)),
                )
            ):
                return False
            expected_equity = float(cash) + sum(
                float(q) * float(marks[iv.next_index, j]) * float(multipliers[j])
                for j, q in enumerate(quantities)
            )
            if not _near(iv.cash_after, float(cash)):
                return False
            if not _near(iv.portfolio_value_after, expected_equity):
                return False
            peak = max(peak, expected_equity)
            minimum_observed_drawdown = max(
                minimum_observed_drawdown, 1.0 - expected_equity / peak
            )
            # Native risk may observe additional intrabar equity peaks. The
            # final-bar mark-to-market path is still a necessary lower bound.
            if iv.max_drawdown_after + 1e-8 < minimum_observed_drawdown:
                return False
            interval_return = max(
                max(expected_equity, 0.0) / previous_equity - 1.0,
                -1.0 + 1e-12,
            )
            if not math.isclose(
                iv.interval_net_return,
                interval_return,
                abs_tol=1e-10,
                rel_tol=0.0,
            ):
                return False
            previous_equity = expected_equity
            next_index = iv.next_index
        book = result.book
        return bool(
            all(abs(float(q)) <= 1e-9 for q in quantities)
            and all(Fraction(q) == 0 for q in ledger.terminal_exact_quantities)
            and all(Fraction(q) == 0 for q in book.exact_quantities)
            and ledger.termination_reason is None
            and not ledger.active_order_remainders
            and book.termination_reason is None
            and not book.insolvent
            and ledger.final_max_drawdown + 1e-8 >= minimum_observed_drawdown
            and book.max_drawdown + 1e-8 >= minimum_observed_drawdown
            and _near(ledger.final_turnover_total, turnover)
            and _near(float(book.turnover_total), turnover)
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
                    (ledger.final_turnover_total, float(book.turnover_total)),
                    (ledger.final_max_drawdown, float(book.max_drawdown)),
                )
            )
        )
    except (AttributeError, IndexError, ValueError, TypeError, OverflowError):
        return False
