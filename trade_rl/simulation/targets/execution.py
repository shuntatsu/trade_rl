"""Shared target-to-order execution used by environments and compatibility APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders.model import OrderBookState, OrderType, TimeInForce
from trade_rl.simulation.orders.reconciliation import reconcile_target
from trade_rl.simulation.stateful.execution import (
    StatefulExecutionResult,
    execute_stateful_orders,
)

if TYPE_CHECKING:
    from trade_rl.simulation.execution import MarketExecutor

_EQUITY_TOLERANCE = 1e-12


def execute_target_statefully(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    target: np.ndarray,
    *,
    start_index: int,
    bars: int,
    target_identity: str,
    time_in_force: TimeInForce = TimeInForce.GTC,
    expiry_index: int | None = None,
) -> StatefulExecutionResult:
    """Reconcile one target against holdings and active residual orders."""

    if not isinstance(target_identity, str) or not target_identity:
        raise ValueError("target_identity must be non-empty")
    if bars <= 0:
        raise ValueError("bars must be positive")
    if start_index < 0 or start_index + bars >= executor.dataset.n_bars:
        raise ValueError("target execution interval is outside the dataset")
    if book.weights.shape != (executor.dataset.n_symbols,):
        raise ValueError("book weights shape does not match market symbols")
    if not np.array_equal(
        np.asarray(book.contract_multipliers),
        executor.dataset.resolved_array("contract_multipliers"),
    ):
        raise ValueError("book contract multipliers do not match market dataset")
    if not np.isfinite(book.portfolio_value) or book.portfolio_value <= 0.0:
        raise ValueError("stateful execution requires positive starting equity")

    target_vector = np.asarray(target, dtype=np.float64).reshape(-1)
    valuation_prices = book.mark_prices
    if not any(book.exact_quantities):
        valuation_prices = executor.dataset.resolved_array("mark_price")[start_index]
    submit_tick_sizes, _, _ = executor._effective_rule_array_views(index=start_index)
    reconciliation = reconcile_target(
        dataset_id=executor.dataset.dataset_id,
        target_identity=target_identity,
        execution_policy_digest=executor.execution_policy_digest,
        target_weights=target_vector,
        book=book,
        order_book=order_book,
        reference_prices=executor.dataset.close[start_index],
        valuation_prices=valuation_prices,
        decision_equity=max(book.portfolio_value, _EQUITY_TOLERANCE),
        submit_index=start_index,
        latency_bars=executor.cost.order_latency_bars,
        order_type=OrderType(executor.cost.order_type),
        time_in_force=time_in_force,
        expiry_index=expiry_index,
        limit_offset_rate=executor.cost.limit_offset_rate,
        tick_sizes=submit_tick_sizes,
        maximum_gross=executor.cost.max_leverage,
        reduce_only_symbols=executor.reduce_only_symbols,
    )
    active_by_id = {order.order_id: order for order in order_book.active_orders}
    cancellation_transitions = []
    for cancelled in reconciliation.cancelled_orders:
        previous = active_by_id.get(cancelled.order_id)
        if previous is None:
            raise RuntimeError("reconciliation cancelled an unknown active order")
        cancellation_transitions.append((previous, cancelled))
    return execute_stateful_orders(
        executor,
        book,
        reconciliation.order_book,
        reconciliation.new_intents,
        start_index=start_index,
        bars=bars,
        reconciliation_cancellations=tuple(cancellation_transitions),
    )


def execute_quantity_hold_statefully(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    symbol_index: int,
    start_index: int,
) -> StatefulExecutionResult:
    """Cancel every selected-symbol MARKET residual and advance without sizing.

    Includes reduce-only residuals: retaining one is a different action from
    actual quantity HOLD. The caller must apply hard risk before choosing this
    path. Canonical corporate actions and carry still run on the next bar.
    """

    if not 0 <= symbol_index < executor.dataset.n_symbols:
        raise ValueError("hold symbol is outside the dataset")
    if not 0 <= start_index < executor.dataset.n_bars - 1:
        raise ValueError("hold execution interval is outside the dataset")
    if book.quantities.shape != (executor.dataset.n_symbols,):
        raise ValueError("book quantities do not match market symbols")
    if not np.array_equal(
        np.asarray(book.contract_multipliers),
        executor.dataset.resolved_array("contract_multipliers"),
    ):
        raise ValueError("book contract multipliers do not match market dataset")
    if not np.isfinite(book.portfolio_value) or book.portfolio_value <= 0.0:
        raise ValueError("quantity hold requires positive starting equity")
    selected = order_book.active_for_symbol(symbol_index)
    if any(order.intent.order_type is not OrderType.MARKET for order in selected):
        raise ValueError("quantity hold supports only MARKET residuals")
    transitions = []
    updated_book = order_book
    for previous in selected:
        cancelled = previous.cancel(
            processing_index=start_index, reason="quantity_hold"
        )
        updated_book = updated_book.replace(cancelled)
        transitions.append((previous, cancelled))
    return execute_stateful_orders(
        executor,
        book,
        updated_book,
        (),
        start_index=start_index,
        bars=1,
        reconciliation_cancellations=tuple(transitions),
    )


__all__ = ["execute_target_statefully", "execute_quantity_hold_statefully"]
