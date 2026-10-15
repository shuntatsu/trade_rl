"""Shared native reserve execution; callers own declarations and evidence.

The iterator yields a returned native step before validating it. Callers must
retain that frontier and exhaust the iterator, including after its last yield,
before claiming full coverage. It never resets or writes an environment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from trade_rl.evaluation.allocation import _execute_allocation_target
from trade_rl.evaluation.rl_allocation._native_progress import (
    AllocationAdvanceObserver,
    notify_advance,
    validate_advance_observer,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    native_allocation_execution_facts,
)
from trade_rl.risk import PreTradeRisk
from trade_rl.risk.pretrade import RiskConstrainedTarget
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.simulation.stateful.execution import StatefulExecutionResult


@dataclass(frozen=True, slots=True)
class _ReservedAllocationClosingStep:
    """Actual native return, not a validated settlement or immutable account."""

    start_index: int
    opening_equity: float
    before_state_digest: str
    risk_target: RiskConstrainedTarget
    execution: StatefulExecutionResult
    native: dict[str, Any]
    processing_time_ns: int | None
    clock_valid: bool


def _iter_reserved_allocation_closing(
    executor: MarketExecutor,
    book: BookState,
    orders: OrderBookState,
    *,
    risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
    stop_index: int,
    decision_digest: Callable[[int], str],
    before_state: Callable[[int, BookState, OrderBookState], str],
    validate_binding: Callable[[], None],
    note_stage: Callable[[int, str], None],
    retain_returned: Callable[[StatefulExecutionResult], None] | None = None,
    on_advance: AllocationAdvanceObserver | None = None,
) -> Iterator[_ReservedAllocationClosingStep]:
    """Continue one actual account through every declared reserve bar.

    Hooks bind the caller's frozen declaration and preserve its failure frontier;
    they supply no execution implementation or replacement accounting. A flat
    account still processes native carry. Only actual economic termination stops
    the reserve early, and never establishes full settlement coverage.

    A caller owning a live account can retain each actual return before any
    fallible diagnostics. The hook supplies no replacement execution.
    """
    validate_advance_observer(on_advance)
    if (
        type(start_index) is not int
        or type(stop_index) is not int
        or not 0 <= start_index < stop_index < executor.dataset.n_bars
    ):
        raise ValueError("reserved closing requires a nonempty native Dataset window")
    index = start_index
    while index < stop_index:
        note_stage(index, "closing_preflight")
        validate_binding()
        before = book.portfolio_value
        before_digest = before_state(index, book, orders)
        note_stage(index, "closing_execution")
        target, execution = _execute_allocation_target(
            executor,
            book,
            orders,
            target_weight=0.0,
            decision_digest=decision_digest(index),
            pretrade_risk=risk,
            symbol_index=symbol_index,
            start_index=index,
        )
        if retain_returned is not None:
            retain_returned(execution)
        note_stage(index, "closing_returned")
        facts = native_allocation_execution_facts(execution)
        clock_valid = (
            type(execution.next_index) is int
            and type(execution.bars_advanced) is int
            and execution.bars_advanced == 1
            and execution.next_index == index + 1
        )
        processing_time = (
            int(
                executor.dataset.timestamps[execution.next_index]
                .astype("datetime64[ns]")
                .astype("int64")
            )
            if type(execution.next_index) is int
            and 0 <= execution.next_index < executor.dataset.n_bars
            else None
        )
        yield _ReservedAllocationClosingStep(
            index,
            before,
            before_digest,
            target,
            execution,
            facts,
            processing_time,
            clock_valid,
        )
        # Resuming after the final yield is mandatory: do not lose its rejection.
        validation_start = index
        book, orders, index = execution.book, execution.order_book, execution.next_index
        note_stage(validation_start, "closing_validation")
        if not clock_valid:
            raise ValueError("terminal native execution changed its one-bar clock")
        validate_binding()
        if on_advance is not None:
            notify_advance(
                on_advance,
                "reserve",
                validation_start,
                index,
                executor=executor,
                book=book,
                orders=orders,
                validate_binding=validate_binding,
            )
        if book.termination_reason is not None:
            break


__all__: list[str] = []
