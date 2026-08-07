"""Causal target-to-order reconciliation with cancel-and-replace semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderIntent,
    OrderType,
    PendingOrder,
    TimeInForce,
)

_TOLERANCE = 1e-12


class OrderReconciliationError(ValueError):
    """Raised when a target cannot be reconciled causally."""


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    order_book: OrderBookState
    new_intents: tuple[OrderIntent, ...]
    cancelled_orders: tuple[PendingOrder, ...]
    desired_quantities: np.ndarray
    residual_quantities: np.ndarray


def _validate_digest(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise OrderReconciliationError(
            "execution_policy_digest must be a lowercase SHA-256 digest"
        )


def _type_prices(
    *,
    order_type: OrderType,
    quantity: float,
    reference_price: float,
    offset_rate: float,
) -> tuple[float | None, float | None]:
    buy = quantity > 0.0
    if order_type is OrderType.MARKET:
        return None, None
    if order_type is OrderType.LIMIT:
        factor = 1.0 - offset_rate if buy else 1.0 + offset_rate
        return reference_price * factor, None
    if order_type is OrderType.STOP_MARKET:
        factor = 1.0 + offset_rate if buy else 1.0 - offset_rate
        return None, reference_price * factor
    raise OrderReconciliationError(f"unsupported order type: {order_type}")


def reconcile_target(
    *,
    dataset_id: str,
    target_identity: str,
    execution_policy_digest: str,
    target_weights: np.ndarray,
    book: BookState,
    order_book: OrderBookState,
    reference_prices: np.ndarray,
    decision_equity: float,
    submit_index: int,
    latency_bars: int,
    order_type: OrderType,
    time_in_force: TimeInForce,
    expiry_index: int | None,
    limit_offset_rate: float,
    maximum_gross: float = 1.0,
) -> ReconciliationResult:
    """Reconcile a latest target against holdings and active residual orders."""

    if not dataset_id:
        raise OrderReconciliationError("dataset_id must be non-empty")
    if not target_identity:
        raise OrderReconciliationError("target_identity must be non-empty")
    _validate_digest(execution_policy_digest)
    if (
        isinstance(submit_index, bool)
        or not isinstance(submit_index, int)
        or submit_index < 0
    ):
        raise OrderReconciliationError("submit_index must be a non-negative integer")
    if (
        isinstance(latency_bars, bool)
        or not isinstance(latency_bars, int)
        or latency_bars < 0
    ):
        raise OrderReconciliationError("latency_bars must be a non-negative integer")
    if not math.isfinite(limit_offset_rate) or not 0.0 <= limit_offset_rate < 1.0:
        raise OrderReconciliationError("limit/stop offset must be within [0, 1)")
    if not math.isfinite(maximum_gross) or maximum_gross <= 0.0:
        raise OrderReconciliationError("maximum_gross must be finite and positive")
    if not math.isfinite(decision_equity) or decision_equity <= 0.0:
        raise OrderReconciliationError("decision_equity must be finite and positive")

    eligible_index = submit_index + latency_bars
    if time_in_force is TimeInForce.DAY and expiry_index is None:
        raise OrderReconciliationError("day orders require an expiry index")
    if expiry_index is not None and expiry_index < eligible_index:
        raise OrderReconciliationError("expiry index must not precede eligibility")

    weights = np.asarray(target_weights, dtype=np.float64).reshape(-1)
    prices = np.asarray(reference_prices, dtype=np.float64).reshape(-1)
    quantities = np.asarray(book.quantities, dtype=np.float64).reshape(-1)
    multipliers = np.asarray(book.contract_multipliers, dtype=np.float64).reshape(-1)
    expected_shape = quantities.shape
    if (
        weights.shape != expected_shape
        or prices.shape != expected_shape
        or multipliers.shape != expected_shape
    ):
        raise OrderReconciliationError(
            "target, price, multiplier and holdings shapes must match"
        )
    if (
        weights.size == 0
        or not np.isfinite(weights).all()
        or not np.isfinite(prices).all()
        or np.any(prices <= 0.0)
        or not np.isfinite(multipliers).all()
        or np.any(multipliers <= 0.0)
    ):
        raise OrderReconciliationError("target and pricing inputs must be finite")
    gross = float(np.abs(weights).sum())
    if gross > maximum_gross + _TOLERANCE:
        raise OrderReconciliationError("target gross exceeds maximum_gross")

    desired = weights * decision_equity / (prices * multipliers)
    state = order_book
    cancelled: list[PendingOrder] = []
    intents: list[OrderIntent] = []
    residuals = np.zeros_like(desired)

    for symbol_index in range(weights.size):
        target_delta = desired[symbol_index] - quantities[symbol_index]
        active = state.active_for_symbol(symbol_index)
        active_residual = float(sum(order.remaining_quantity for order in active))
        if math.isclose(
            target_delta,
            active_residual,
            rel_tol=0.0,
            abs_tol=_TOLERANCE,
        ):
            continue

        replaced_order_id: str | None = None
        if active:
            ordered_active = tuple(
                sorted(
                    active,
                    key=lambda order: (
                        order.intent.eligible_index,
                        order.order_id,
                    ),
                )
            )
            replaced_order_id = ordered_active[0].order_id
            for order in ordered_active:
                cancelled_order = order.cancel(
                    processing_index=submit_index,
                    reason="superseded",
                )
                state = state.replace(cancelled_order)
                cancelled.append(cancelled_order)

        residual = target_delta
        if abs(residual) <= _TOLERANCE:
            continue
        residuals[symbol_index] = residual
        limit_price, stop_price = _type_prices(
            order_type=order_type,
            quantity=residual,
            reference_price=float(prices[symbol_index]),
            offset_rate=limit_offset_rate,
        )
        intents.append(
            OrderIntent.create(
                dataset_id=dataset_id,
                target_identity=target_identity,
                execution_policy_digest=execution_policy_digest,
                symbol_index=symbol_index,
                requested_quantity=float(residual),
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                submit_index=submit_index,
                eligible_index=eligible_index,
                expiry_index=expiry_index,
                submission_reference_price=float(prices[symbol_index]),
                decision_equity=decision_equity,
                replaced_order_id=replaced_order_id,
            )
        )

    return ReconciliationResult(
        order_book=state,
        new_intents=tuple(intents),
        cancelled_orders=tuple(cancelled),
        desired_quantities=desired,
        residual_quantities=residuals,
    )
