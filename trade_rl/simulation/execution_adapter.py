"""Maintained compatibility executor backed by the stateful order engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import (
    ExecutionResult,
    MarketExecutor as _BaseMarketExecutor,
)
from trade_rl.simulation.orders import OrderBookState
from trade_rl.simulation.target_execution import execute_target_statefully

_ATTEMPT_EVENT_TYPES = frozenset(
    {"eligible", "triggered", "no_fill", "partial_fill", "filled", "rejected"}
)


class StatefulCompatibilityMarketExecutor(_BaseMarketExecutor):
    """Preserve the legacy API while sharing stateful execution semantics.

    A compatibility chain is continued only when the caller passes the exact
    ``BookState`` returned by the preceding call on this executor. Unrelated
    books begin with an empty order book, preventing cross-run contamination.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._compatibility_order_book = OrderBookState.empty()
        self._compatibility_last_book: BookState | None = None

    @property
    def compatibility_order_book(self) -> OrderBookState:
        return self._compatibility_order_book

    def _reset_compatibility_chain(self) -> None:
        self._compatibility_order_book = OrderBookState.empty()
        self._compatibility_last_book = None

    def reset_random_state(self, seed: int | None = None) -> None:
        super().reset_random_state(seed)
        self._reset_compatibility_chain()

    def execute_interval(
        self,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
    ) -> ExecutionResult:
        state = (
            self._compatibility_order_book
            if book is self._compatibility_last_book
            else OrderBookState.empty()
        )
        target_vector = np.asarray(target, dtype=np.float64).reshape(-1)
        target_identity = content_digest(
            {
                "bars": bars,
                "dataset_id": self.dataset.dataset_id,
                "schema_version": "compatibility_target_execution_v1",
                "start_index": start_index,
                "target": tuple(float(value) for value in target_vector),
            }
        )
        stateful = execute_target_statefully(
            self,
            book,
            state,
            target_vector,
            start_index=start_index,
            bars=bars,
            target_identity=target_identity,
        )
        self._compatibility_order_book = stateful.order_book
        self._compatibility_last_book = stateful.book

        attempted = any(
            event.event_type in _ATTEMPT_EVENT_TYPES for event in stateful.order_events
        )
        requested_turnover = stateful.requested_turnover if attempted else 0.0
        unfilled_turnover = stateful.unfilled_turnover if attempted else 0.0
        requested_by_symbol = (
            stateful.requested_notional_by_symbol
            if attempted
            else np.zeros_like(stateful.requested_notional_by_symbol)
        )
        fill_ratio = stateful.fill_ratio if attempted else 1.0
        return ExecutionResult(
            book=stateful.book,
            next_index=stateful.next_index,
            bars_advanced=stateful.bars_advanced,
            interval_gross_return=stateful.interval_gross_return,
            interval_cost=stateful.interval_cost,
            interval_funding=stateful.interval_funding,
            interval_net_return=stateful.interval_net_return,
            interval_log_return=stateful.interval_log_return,
            requested_turnover=requested_turnover,
            filled_turnover=stateful.filled_turnover,
            unfilled_turnover=unfilled_turnover,
            fill_count=stateful.fill_count,
            rebalance_events=stateful.rebalance_events,
            fill_ratio=fill_ratio,
            max_participation=stateful.max_participation,
            interval_borrow_cost=stateful.interval_borrow_cost,
            interval_dividend=stateful.interval_dividend,
            interval_cash_interest=stateful.interval_cash_interest,
            margin_utilization=stateful.book.margin_utilization,
            termination_reason=stateful.termination_reason,
            requested_notional_by_symbol=requested_by_symbol,
            filled_notional_by_symbol=stateful.filled_notional_by_symbol,
            participation_by_symbol=stateful.participation_by_symbol,
            cost_by_symbol=stateful.cost_by_symbol,
        )


__all__ = ["StatefulCompatibilityMarketExecutor"]
