"""Shared target-to-order execution used by environments and compatibility APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.order_reconciliation import reconcile_target
from trade_rl.simulation.orders import OrderBookState, OrderType, TimeInForce
from trade_rl.simulation.stateful_execution import StatefulExecutionResult

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
    if not 0 <= start_index < executor.dataset.n_bars:
        raise ValueError("target execution start index is outside the dataset")
    if bars <= 0 or start_index + bars >= executor.dataset.n_bars:
        raise ValueError("target execution interval is outside the dataset")

    target_vector = np.asarray(target, dtype=np.float64).reshape(-1)
    reconciliation = reconcile_target(
        dataset_id=executor.dataset.dataset_id,
        target_identity=target_identity,
        execution_policy_digest=executor.execution_policy_digest,
        target_weights=target_vector,
        book=book,
        order_book=order_book,
        reference_prices=executor.dataset.close[start_index],
        decision_equity=max(book.portfolio_value, _EQUITY_TOLERANCE),
        submit_index=start_index,
        latency_bars=executor.cost.order_latency_bars,
        order_type=OrderType(executor.cost.order_type),
        time_in_force=time_in_force,
        expiry_index=expiry_index,
        limit_offset_rate=executor.cost.limit_offset_rate,
        maximum_gross=executor.cost.max_leverage,
    )
    return executor.execute_orders(
        book,
        reconciliation.order_book,
        reconciliation.new_intents,
        start_index=start_index,
        bars=bars,
    )


__all__ = ["execute_target_statefully"]
