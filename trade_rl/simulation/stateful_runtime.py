"""Invocation-local state and evidence for stateful execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.bar_path import BarPath
from trade_rl.simulation.liquidity import SymbolCapacityEvidence
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderEvent,
    OrderIntent,
    PendingOrder,
)

if TYPE_CHECKING:
    from trade_rl.simulation.execution import MarketExecutor

_TOLERANCE = 1e-12


def _timestamp_ns(executor: MarketExecutor, index: int) -> int:
    timestamp = executor.dataset.timestamps[index]
    return int(timestamp.astype("datetime64[ns]").astype(np.int64))


@dataclass(slots=True)
class StatefulExecutionRuntime:
    """Own all mutation and aggregate evidence for one execution invocation."""

    executor: MarketExecutor
    book: BookState
    order_book: OrderBookState
    events: list[OrderEvent]
    capacities: list[SymbolCapacityEvidence]
    starting_value: float
    starting_rebalance_events: int
    requested_notional: float
    filled_notional: float
    total_cost: float
    total_funding: float
    total_borrow: float
    total_dividend: float
    total_cash_interest: float
    completed_fills: int
    rejected_count: int
    expired_count: int
    fill_count: int
    max_participation: float
    gross_factor: float
    requested_by_symbol: np.ndarray
    filled_by_symbol: np.ndarray
    participation_by_symbol: np.ndarray
    cost_by_symbol: np.ndarray

    @classmethod
    def create(
        cls,
        executor: MarketExecutor,
        book: BookState,
        order_book: OrderBookState,
    ) -> StatefulExecutionRuntime:
        result_book = book.clone()
        n_symbols = executor.dataset.n_symbols
        return cls(
            executor=executor,
            book=result_book,
            order_book=order_book,
            events=[],
            capacities=[],
            starting_value=0.0,
            starting_rebalance_events=result_book.rebalance_events,
            requested_notional=0.0,
            filled_notional=0.0,
            total_cost=0.0,
            total_funding=0.0,
            total_borrow=0.0,
            total_dividend=0.0,
            total_cash_interest=0.0,
            completed_fills=0,
            rejected_count=0,
            expired_count=0,
            fill_count=0,
            max_participation=0.0,
            gross_factor=1.0,
            requested_by_symbol=np.zeros(n_symbols, dtype=np.float64),
            filled_by_symbol=np.zeros(n_symbols, dtype=np.float64),
            participation_by_symbol=np.zeros(n_symbols, dtype=np.float64),
            cost_by_symbol=np.zeros(n_symbols, dtype=np.float64),
        )

    def submit_intents(self, intents: Sequence[OrderIntent]) -> None:
        for intent in intents:
            pending = PendingOrder.from_intent(intent)
            self.order_book = self.order_book.add(pending)
            self.append_event(
                previous=pending,
                updated=pending,
                event_type="submitted",
                processing_index=intent.submit_index,
            )

    def initialize_metrics(self) -> None:
        self.starting_value = self.book.portfolio_value
        if not math.isfinite(self.starting_value) or self.starting_value <= 0.0:
            raise ValueError("stateful execution requires positive starting equity")
        multipliers = self.book.contract_multipliers
        if multipliers is None:
            raise ValueError("stateful execution requires contract multipliers")
        for order in self.order_book.active_orders:
            symbol = order.intent.symbol_index
            order_notional = (
                abs(order.remaining_quantity)
                * order.intent.submission_reference_price
                * float(multipliers[symbol])
            )
            self.requested_notional += order_notional
            self.requested_by_symbol[symbol] += order_notional

    def append_event(
        self,
        *,
        previous: PendingOrder,
        updated: PendingOrder,
        event_type: str,
        processing_index: int,
        filled_quantity: float = 0.0,
        execution_price: float | None = None,
        filled_notional: float = 0.0,
        capacity_before: float = 0.0,
        capacity_after: float = 0.0,
        participation_rate: float = 0.0,
        trigger_segment: str | None = None,
        available_volume_fraction: float = 0.0,
        reason: str | None = None,
        path: BarPath | None = None,
    ) -> None:
        executor = self.executor
        self.events.append(
            OrderEvent(
                schema_version="order_event_v1",
                sequence=len(self.events),
                order_id=updated.order_id,
                replaced_order_id=updated.intent.replaced_order_id,
                dataset_id=updated.intent.dataset_id,
                execution_policy_digest=updated.intent.execution_policy_digest,
                symbol_index=updated.intent.symbol_index,
                event_type=event_type,
                processing_index=processing_index,
                timestamp_ns=_timestamp_ns(executor, processing_index),
                previous_status=previous.status,
                new_status=updated.status,
                requested_quantity=updated.intent.requested_quantity,
                remaining_quantity=updated.remaining_quantity,
                filled_quantity=filled_quantity,
                execution_price=execution_price,
                filled_notional=filled_notional,
                capacity_before=capacity_before,
                capacity_after=capacity_after,
                participation_rate=participation_rate,
                trigger_segment=trigger_segment,
                available_volume_fraction=available_volume_fraction,
                reason=reason,
                path_mode=(
                    executor.cost.path_mode if path is None else path.mode.value
                ),
                path_points=() if path is None else path.points,
            )
        )

    def cancel_active_orders(
        self,
        *,
        processing_index: int,
        reason: str,
        symbol_mask: np.ndarray | None = None,
    ) -> None:
        for order in tuple(self.order_book.active_orders):
            if symbol_mask is not None and not symbol_mask[order.intent.symbol_index]:
                continue
            updated = order.cancel(processing_index=processing_index, reason=reason)
            self.order_book = self.order_book.replace(updated)
            self.append_event(
                previous=order,
                updated=updated,
                event_type="cancelled",
                processing_index=processing_index,
                reason=reason,
            )

    def active_order(self, order_id: str) -> PendingOrder | None:
        return next(
            (
                order
                for order in self.order_book.active_orders
                if order.order_id == order_id
            ),
            None,
        )

    def require_active_order(self, order_id: str) -> PendingOrder:
        return next(
            order
            for order in self.order_book.active_orders
            if order.order_id == order_id
        )

    def result_payload(self, *, start_index: int, bars: int) -> dict[str, Any]:
        ending_value = max(self.book.portfolio_value, 0.0)
        interval_net_return = max(
            ending_value / self.starting_value - 1.0,
            -1.0 + 1e-12,
        )
        requested_turnover = self.requested_notional / max(
            self.starting_value, _TOLERANCE
        )
        filled_turnover = self.filled_notional / max(
            self.starting_value, _TOLERANCE
        )
        reason = (
            None
            if self.book.termination_reason is None
            else EconomicTerminationReason(self.book.termination_reason).value
        )
        return {
            "book": self.book,
            "order_book": self.order_book,
            "next_index": start_index + bars,
            "bars_advanced": bars,
            "order_events": tuple(self.events),
            "capacity_evidence": tuple(self.capacities),
            "interval_cost": self.total_cost,
            "interval_funding": self.total_funding,
            "interval_borrow_cost": self.total_borrow,
            "interval_dividend": self.total_dividend,
            "interval_cash_interest": self.total_cash_interest,
            "interval_gross_return": self.gross_factor - 1.0,
            "interval_net_return": interval_net_return,
            "interval_log_return": math.log1p(interval_net_return),
            "requested_notional": self.requested_notional,
            "filled_notional": self.filled_notional,
            "requested_turnover": requested_turnover,
            "filled_turnover": filled_turnover,
            "unfilled_turnover": max(0.0, requested_turnover - filled_turnover),
            "fill_ratio": (
                1.0
                if self.requested_notional <= _TOLERANCE
                else min(1.0, self.filled_notional / self.requested_notional)
            ),
            "rebalance_events": (
                self.book.rebalance_events - self.starting_rebalance_events
            ),
            "completed_fill_count": self.completed_fills,
            "rejected_count": self.rejected_count,
            "expired_count": self.expired_count,
            "fill_count": self.fill_count,
            "max_participation": self.max_participation,
            "requested_notional_by_symbol": self.requested_by_symbol,
            "filled_notional_by_symbol": self.filled_by_symbol,
            "participation_by_symbol": self.participation_by_symbol,
            "cost_by_symbol": self.cost_by_symbol,
            "termination_reason": reason,
        }
