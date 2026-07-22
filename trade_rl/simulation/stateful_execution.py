"""Stateful OHLCV order execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.liquidity import SymbolCapacityEvidence
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderEvent,
    OrderIntent,
)
from trade_rl.simulation.stateful_bar_lifecycle import StatefulBarLifecycle
from trade_rl.simulation.stateful_order_transitions import (
    StatefulOrderTransitionProcessor,
)
from trade_rl.simulation.stateful_runtime import StatefulExecutionRuntime
from trade_rl.simulation.stateful_symbol_fills import StatefulSymbolFillProcessor

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


def execute_stateful_orders(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    intents: Sequence[OrderIntent],
    *,
    start_index: int,
    bars: int,
) -> StatefulExecutionResult:
    """Execute persistent orders over one or more processing bars."""

    dataset = executor.dataset
    if bars <= 0:
        raise ValueError("bars must be positive")
    if start_index < 0 or start_index + bars >= dataset.n_bars:
        raise ValueError("execution interval is outside the dataset")
    if book.quantities.shape != (dataset.n_symbols,):
        raise ValueError("book quantities do not match market symbols")

    runtime = StatefulExecutionRuntime.create(executor, book, order_book)
    runtime.submit_intents(intents)
    runtime.initialize_metrics()
    lifecycle = StatefulBarLifecycle(executor)
    transitions = StatefulOrderTransitionProcessor(executor)
    fills = StatefulSymbolFillProcessor(executor)

    for offset in range(bars):
        previous_index = start_index + offset
        processing_index = previous_index + 1
        context = lifecycle.begin_bar(
            runtime,
            previous_index=previous_index,
            processing_index=processing_index,
        )
        accepted = transitions.prepare_orders(runtime, context)
        attempted = fills.process_symbols(runtime, context, accepted)
        transitions.expire_attempted_remainders(
            runtime,
            processing_index=processing_index,
            attempted_order_ids=attempted,
        )
        lifecycle.finish_bar(runtime, context)

    return StatefulExecutionResult(
        **runtime.result_payload(start_index=start_index, bars=bars)
    )
