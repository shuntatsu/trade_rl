"""Stateful OHLCV order execution orchestration."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from trade_rl._validation import _ordered_ns_durations, _require_nominal_session_seconds
from trade_rl.data.market import MarketDataset, _require_available_marks_at_row
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.diagnostics.fill_cash import NativeFillCashObservation
from trade_rl.simulation.diagnostics.funding import FundingBoundaryEvidence
from trade_rl.simulation.liquidity import SymbolCapacityEvidence
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TerminalOrderArchive,
    TimeInForce,
)
from trade_rl.simulation.stateful.bar_lifecycle import StatefulBarLifecycle
from trade_rl.simulation.stateful.order_transitions import (
    StatefulOrderTransitionProcessor,
)
from trade_rl.simulation.stateful.runtime import StatefulExecutionRuntime
from trade_rl.simulation.stateful.symbol_fills import StatefulSymbolFillProcessor

if TYPE_CHECKING:
    from trade_rl.simulation.execution import MarketExecutor


@dataclass(frozen=True, slots=True)
class StatefulExecutionResult:
    book: BookState
    order_book: OrderBookState
    next_index: int
    bars_advanced: int
    order_events: tuple[OrderEvent, ...]
    capacity_evidence: tuple[SymbolCapacityEvidence, ...]
    funding_evidence: tuple[FundingBoundaryEvidence, ...]
    interval_cost: float
    interval_funding: float
    interval_borrow_cost: float
    interval_dividend: float
    interval_cash_interest: float
    interval_gross_return: float
    interval_net_return: float
    interval_log_return: float
    requested_notional: float
    filled_notional: float
    requested_turnover: float
    filled_turnover: float
    unfilled_turnover: float
    fill_ratio: float
    rebalance_events: int
    completed_fill_count: int
    rejected_count: int
    expired_count: int
    fill_count: int
    max_participation: float
    requested_notional_by_symbol: np.ndarray
    filled_notional_by_symbol: np.ndarray
    participation_by_symbol: np.ndarray
    cost_by_symbol: np.ndarray
    termination_reason: str | None


@dataclass(frozen=True, slots=True)
class StatefulExecutionObservation:
    """Detached immutable facts observed after one stateful execution call."""

    next_index: int
    order_events: tuple[OrderEvent, ...]
    capacity_evidence: tuple[SymbolCapacityEvidence, ...]
    funding_evidence: tuple[FundingBoundaryEvidence, ...]
    active_order_remainders: tuple[tuple[str, float], ...]
    terminal_order_reasons: tuple[tuple[str, str], ...]

    @classmethod
    def from_result(
        cls, result: StatefulExecutionResult
    ) -> StatefulExecutionObservation:
        return cls._from_result_with_terminal_reasons(
            result, cls._terminal_reasons(result.order_book.terminal_orders)
        )

    @classmethod
    def from_result_since(
        cls,
        result: StatefulExecutionResult,
        previous_terminal_orders: TerminalOrderArchive,
    ) -> StatefulExecutionObservation:
        """Build interval facts using only terminal orders appended this call."""

        current_terminal_orders = result.order_book.terminal_orders
        if not isinstance(current_terminal_orders, TerminalOrderArchive):
            raise RuntimeError("terminal order archive was not normalized")
        terminal_reasons = cls._terminal_reasons(
            current_terminal_orders.suffix_since(previous_terminal_orders)
        )
        return cls._from_result_with_terminal_reasons(result, terminal_reasons)

    @staticmethod
    def _terminal_reasons(
        terminal_orders: Sequence[PendingOrder],
    ) -> tuple[tuple[str, str], ...]:
        terminal_reasons: list[tuple[str, str]] = []
        for order in terminal_orders:
            reason = order.terminal_reason
            if reason is None:
                raise RuntimeError("terminal order observation is missing its reason")
            terminal_reasons.append((order.order_id, reason))
        return tuple(terminal_reasons)

    @classmethod
    def _from_result_with_terminal_reasons(
        cls,
        result: StatefulExecutionResult,
        terminal_reasons: tuple[tuple[str, str], ...],
    ) -> StatefulExecutionObservation:
        return cls(
            next_index=result.next_index,
            order_events=result.order_events,
            capacity_evidence=result.capacity_evidence,
            funding_evidence=result.funding_evidence,
            active_order_remainders=tuple(
                (order.order_id, float(order.remaining_quantity))
                for order in result.order_book.active_orders
            ),
            terminal_order_reasons=terminal_reasons,
        )


def execute_stateful_orders(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    intents: Sequence[OrderIntent],
    *,
    start_index: int,
    bars: int,
    reconciliation_cancellations: Sequence[tuple[PendingOrder, PendingOrder]] = (),
) -> StatefulExecutionResult:
    """Execute an invocation without consuming RNG on failed book transitions."""

    # BookState and OrderBookState are invocation-local, but the slippage RNG
    # belongs to the reusable executor. Roll it back on an aborted invocation,
    # so a retry of the same explicit inputs cannot price from a later draw.
    rng = executor._rng
    rng_state = (
        deepcopy(rng.bit_generator.state)
        if isinstance(rng, np.random.Generator)
        else None
    )
    try:
        return _execute_stateful_orders_impl(
            executor,
            book,
            order_book,
            intents,
            start_index=start_index,
            bars=bars,
            reconciliation_cancellations=reconciliation_cancellations,
        )
    except BaseException:
        if rng_state is not None:
            rng.bit_generator.state = rng_state
        raise


def _require_close_assigned_linear_funding_rows(
    dataset: MarketDataset, span: slice
) -> None:
    """Validate declared single close events, not inferred intrabar timing."""
    if dataset.funding_price_rate is None:
        raise ValueError("linear carry requires explicit funding price-rate data")
    count = dataset.resolved_array("funding_event_count")[span]
    due = dataset.resolved_array("funding_due")[span]
    rate = dataset.funding_rate[span]
    price_rate = dataset.funding_price_rate[span]
    if (
        np.any(count < 0)
        or np.any(count > 1)
        or np.any(due != (count == 1))
        or not np.isfinite(rate).all()
        or not np.isfinite(price_rate).all()
        or np.any(~due & ((rate != 0.0) | (price_rate != 0.0)))
        or np.any((rate == 0.0) != (price_rate == 0.0))
    ):
        raise ValueError("linear carry requires consistent single-event funding rows")
    nonzero = due & (rate != 0.0)
    # A nonzero single-event rate requires a representable positive event
    # price. Do not substitute the close or try to reconstruct event timing.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        implied_price = price_rate[nonzero] / rate[nonzero]
    if not np.isfinite(implied_price).all() or np.any(implied_price <= 0.0):
        raise ValueError("linear carry requires a finite positive funding event price")


def _require_native_execution_rows(
    dataset: MarketDataset,
    *,
    start_index: int,
    stop_index: int,
    native_linear_carry: bool = False,
    native_session_linear_carry: bool = False,
) -> None:
    """Check the complete declared native window without invoking an actor.

    The first row supplies the decision mark and liquidity; only subsequent
    processing rows incur funding, corporate actions and cash carry. Use the
    same admission predicates for an individual transition and an up-front
    learning window. The validated data are not a provider attestation.
    """
    if (
        type(start_index) is not int
        or type(stop_index) is not int
        or not 0 <= start_index < stop_index < dataset.n_bars
    ):
        raise ValueError("native execution window is outside the Dataset")
    span = slice(start_index, stop_index + 1)
    processing = slice(start_index + 1, stop_index + 1)
    if native_session_linear_carry:
        nominal_hours = dataset.bar_hours
        seconds = nominal_hours * 3600
        if not math.isfinite(seconds):
            raise ValueError("session requires finite whole-second nominal bars")
        nominal_seconds = round(seconds)
        if nominal_seconds / 3600 != nominal_hours:
            raise ValueError("session requires whole-second nominal bars")
        timestamps = dataset.timestamps[span].astype("datetime64[ns]")
        boundaries = tuple(int(x) for x in timestamps.view(np.int64))
        _require_nominal_session_seconds(
            nominal_seconds, _ordered_ns_durations(boundaries)
        )
    unavailable = np.any(
        (dataset.resolved_array("available_at")[span] > dataset.timestamps[span, None])
        | ~dataset.resolved_array("information_available")[span],
        axis=1,
    )
    if np.any(unavailable):
        # Reuse the common row diagnostics for the earliest unavailable mark.
        _require_available_marks_at_row(
            dataset,
            row_index=start_index + int(np.argmax(unavailable)),
            selected_symbol_indices=tuple(range(dataset.n_symbols)),
        )
    if (
        (
            not native_linear_carry
            and np.any(dataset.resolved_array("funding_event_count")[processing] != 0)
        )
        or np.any(dataset.resolved_array("dividend")[processing] != 0.0)
        or np.any(dataset.resolved_array("split_factor")[processing] != 1.0)
        or np.any(~dataset.resolved_array("asset_active")[span])
        or np.any(~dataset.tradable[span])
        or np.any(dataset.resolved_array("cash_rate")[processing] < 0.0)
    ):
        raise ValueError(
            ("native session" if native_session_linear_carry else "native fixed")
            + " execution refuses unsupported funding, corporate actions, gaps or negative carry"
        )
    if native_linear_carry:
        _require_close_assigned_linear_funding_rows(dataset, processing)
        # A coarse bar cannot reconstruct a recall or replacement borrow.
        if np.any(~dataset.resolved_array("borrow_available")[span]):
            raise ValueError("native linear carry requires available borrow")


def _execute_stateful_orders_impl(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    intents: Sequence[OrderIntent],
    *,
    start_index: int,
    bars: int,
    reconciliation_cancellations: Sequence[tuple[PendingOrder, PendingOrder]] = (),
) -> StatefulExecutionResult:
    dataset = executor.dataset
    if bars <= 0:
        raise ValueError("bars must be positive")
    if start_index < 0 or start_index + bars >= dataset.n_bars:
        raise ValueError("execution interval is outside the dataset")
    if book.quantities.shape != (dataset.n_symbols,):
        raise ValueError("book quantities do not match market symbols")
    if executor.insolvency_valuation == "retain_debt" and book.insolvent:
        raise ValueError(
            "retain_debt execution cannot restart from an already terminated book"
        )

    executor._validate_native_fixed_configuration()
    native_fixed = dataset.native_clock is not None
    if native_fixed or executor.native_session_linear_carry:
        if executor._native_linear_carry:
            _require_native_execution_rows(
                dataset,
                start_index=start_index,
                stop_index=start_index + bars,
                native_linear_carry=True,
                native_session_linear_carry=executor.native_session_linear_carry,
            )
        else:
            book.require_nonnegative_spot_cash()
        all_intents = (*intents, *(order.intent for order in order_book.active_orders))
        if any(
            intent.order_type is not OrderType.MARKET
            or intent.time_in_force is TimeInForce.DAY
            for intent in all_intents
        ):
            raise ValueError(
                "native fixed execution admits only MARKET IOC/GTC order intents"
            )
    if executor.strict_spot_cash:
        executor._validate_strict_spot_configuration()
        book.require_nonnegative_spot_cash()

    runtime = StatefulExecutionRuntime.create(executor, book, order_book)
    cancellation_ids: set[str] = set()
    for previous, updated in reconciliation_cancellations:
        if previous.order_id in cancellation_ids:
            raise ValueError("duplicate reconciliation cancellation transition")
        cancellation_ids.add(previous.order_id)
        if previous.terminal:
            raise ValueError("reconciliation cancellation source is terminal")
        if previous.order_id != updated.order_id:
            raise ValueError("reconciliation cancellation changed order identity")
        if updated.status is not OrderStatus.CANCELLED:
            raise ValueError("reconciliation transition is not a cancellation")
        processing_index = updated.last_processed_index
        reason = updated.terminal_reason
        if processing_index is None or reason is None:
            raise ValueError("reconciliation cancellation evidence is incomplete")
        if updated != previous.cancel(
            processing_index=processing_index,
            reason=reason,
        ):
            raise ValueError("reconciliation cancellation state is inconsistent")
        if updated not in runtime.order_book.terminal_orders:
            raise ValueError("reconciliation cancellation is absent from order book")
        runtime.append_event(
            previous=previous,
            updated=updated,
            event_type="cancelled",
            processing_index=processing_index,
            reason=reason,
        )
    runtime.submit_intents(intents)
    runtime.initialize_metrics()
    lifecycle = StatefulBarLifecycle(executor)
    transitions = StatefulOrderTransitionProcessor(executor)
    fills = StatefulSymbolFillProcessor(executor)

    bars_advanced = 0
    for offset in range(bars):
        previous_index = start_index + offset
        processing_index = previous_index + 1
        if native_fixed and not executor._native_linear_carry:
            _require_native_execution_rows(
                dataset, start_index=previous_index, stop_index=processing_index
            )
        if executor.strict_spot_cash:
            executor._validate_strict_spot_row(processing_index)
        context = lifecycle.begin_bar(
            runtime,
            previous_index=previous_index,
            processing_index=processing_index,
        )
        if executor.strict_spot_cash:
            runtime.book.require_nonnegative_spot_cash()
        absorbing = executor.insolvency_valuation == "retain_debt"
        if not (absorbing and runtime.book.insolvent):
            accepted = transitions.prepare_orders(runtime, context)
            attempted = fills.process_symbols(runtime, context, accepted)
            if not (absorbing and runtime.book.insolvent):
                transitions.expire_attempted_remainders(
                    runtime,
                    processing_index=processing_index,
                    attempted_order_ids=attempted,
                )
        lifecycle.finish_bar(runtime, context)
        if executor.strict_spot_cash or (
            native_fixed and not executor._native_linear_carry
        ):
            runtime.book.require_nonnegative_spot_cash()
        bars_advanced += 1
        if absorbing and runtime.book.insolvent:
            break

    result = StatefulExecutionResult(
        **runtime.result_payload(start_index=start_index, bars=bars_advanced)
    )
    if runtime.fill_cash_records is not None:
        observer = executor._fill_cash_observer
        if observer is None:
            raise RuntimeError("native fill cash observer was removed during execution")
        records = tuple(runtime.fill_cash_records)
        if tuple(row.event for row in records) != tuple(
            event
            for event in result.order_events
            if event.event_type in ("filled", "partial_fill")
        ):
            raise RuntimeError("native fill cash observation omitted an order fill")
        observer(
            NativeFillCashObservation(
                dataset_id=dataset.dataset_id,
                execution_policy_digest=executor.execution_policy_digest,
                start_index=start_index,
                next_index=result.next_index,
                fills=records,
                interval_cost=result.interval_cost,
            )
        )
    return result
