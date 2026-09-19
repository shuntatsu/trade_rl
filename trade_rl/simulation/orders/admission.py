"""Deterministic pre-trade admission decisions for persistent orders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders.model import OrderIntent, OrderType
from trade_rl.simulation.quantities import (
    accepted_fill_quantity,
    exact_quantity,
    quantize_quantity,
)

_TOLERANCE = 1e-12


class OrderAdmissionError(ValueError):
    """Raised when admission inputs are structurally invalid."""


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    accepted: bool
    reason: str | None
    admitted_quantity: float
    admitted_notional: float


@dataclass(frozen=True, slots=True)
class OrderAdmissionPolicy:
    expected_dataset_id: str
    expected_execution_policy_digest: str
    allow_short: bool
    max_leverage: float

    def __post_init__(self) -> None:
        if not self.expected_dataset_id:
            raise OrderAdmissionError("expected_dataset_id must be non-empty")
        if len(self.expected_execution_policy_digest) != 64:
            raise OrderAdmissionError(
                "expected_execution_policy_digest must be a SHA-256 digest"
            )
        if not isinstance(self.allow_short, bool):
            raise OrderAdmissionError("allow_short must be a boolean")
        if not math.isfinite(self.max_leverage) or self.max_leverage <= 0.0:
            raise OrderAdmissionError("max_leverage must be finite and positive")

    @staticmethod
    def _reject(reason: str) -> AdmissionDecision:
        return AdmissionDecision(
            accepted=False,
            reason=reason,
            admitted_quantity=0.0,
            admitted_notional=0.0,
        )

    def evaluate(
        self,
        intent: OrderIntent,
        *,
        remaining_quantity: float | None = None,
        book: BookState,
        processing_index: int,
        asset_active: bool,
        tradable: bool,
        buy_allowed: bool,
        sell_allowed: bool,
        borrow_available: bool,
        tick_size: float,
        lot_size: float,
        minimum_notional: float,
        reference_prices: np.ndarray,
        actual_position: Fraction | None = None,
        minimum_quantity: float = 0.0,
        maximum_quantity: float | None = None,
        market_only: bool = False,
    ) -> AdmissionDecision:
        """Return an explicit admission or economic rejection reason."""

        if actual_position is not None and not isinstance(actual_position, Fraction):
            raise OrderAdmissionError("actual_position must be an exact Fraction")

        prices = np.asarray(reference_prices, dtype=np.float64).reshape(-1)
        quantities = np.asarray(book.quantities, dtype=np.float64).reshape(-1)
        multipliers = np.asarray(book.contract_multipliers, dtype=np.float64).reshape(
            -1
        )
        if (
            prices.shape != quantities.shape
            or multipliers.shape != quantities.shape
            or prices.size == 0
            or not np.isfinite(prices).all()
            or np.any(prices <= 0.0)
        ):
            raise OrderAdmissionError(
                "reference_prices must match book symbols and be finite"
            )
        if intent.symbol_index >= quantities.size:
            raise OrderAdmissionError("order symbol is outside the book")
        if (
            isinstance(processing_index, bool)
            or not isinstance(processing_index, int)
            or processing_index < 0
        ):
            raise OrderAdmissionError("processing_index must be a non-negative integer")
        if intent.dataset_id != self.expected_dataset_id or (
            intent.execution_policy_digest != self.expected_execution_policy_digest
        ):
            return self._reject("identity_mismatch")
        if processing_index < intent.eligible_index:
            return self._reject("not_eligible")
        if intent.expiry_index is not None and processing_index > intent.expiry_index:
            return self._reject("expired")
        if not asset_active:
            return self._reject("inactive_asset")
        if not tradable:
            return self._reject("non_tradable_market")

        requested = (
            intent.requested_quantity
            if remaining_quantity is None
            else remaining_quantity
        )
        if (
            not math.isfinite(requested)
            or abs(requested) <= _TOLERANCE
            or math.copysign(1.0, requested)
            != math.copysign(1.0, intent.requested_quantity)
            or abs(requested) > abs(intent.requested_quantity) + _TOLERANCE
        ):
            raise OrderAdmissionError(
                "remaining_quantity must preserve the order direction and bound"
            )
        if requested > 0.0 and not buy_allowed:
            return self._reject("buy_disabled")
        if requested < 0.0 and not sell_allowed:
            return self._reject("sell_disabled")

        if market_only and intent.order_type is not OrderType.MARKET:
            return self._reject("profile_requires_market_order")
        for value in (tick_size, lot_size, minimum_notional, minimum_quantity):
            if not math.isfinite(value) or value < 0.0:
                return self._reject("invalid_execution_rule")
        if maximum_quantity is not None:
            if (
                not math.isfinite(maximum_quantity)
                or maximum_quantity <= 0
                or maximum_quantity < minimum_quantity
            ):
                return self._reject("invalid_execution_rule")
            if abs(exact_quantity(requested)) > exact_quantity(maximum_quantity):
                return self._reject("above_maximum_quantity")

        symbol = intent.symbol_index
        current = quantities[symbol]
        if intent.reduce_only:
            # Stateful admission projects other pending orders for economics,
            # but an unfilled opening is not inventory available to close.
            position = (
                book.exact_quantities[symbol]
                if actual_position is None
                else actual_position
            )
            request = exact_quantity(requested)
            if not position:
                return self._reject("reduce_only_no_position")
            if position * request >= 0:
                return self._reject("reduce_only_wrong_direction")
            if abs(request) > abs(position):
                return self._reject("reduce_only_exceeds_position")
        projected_unrounded = current + requested
        if projected_unrounded < -_TOLERANCE and not self.allow_short:
            return self._reject("shorting_disabled")
        if (
            projected_unrounded < min(current, 0.0) - _TOLERANCE
            and not borrow_available
        ):
            return self._reject("borrow_unavailable")

        admitted_quantity, count = quantize_quantity(requested, lot_size)
        if abs(admitted_quantity) <= _TOLERANCE:
            return self._reject("zero_quantity_after_rounding")
        admitted_exact = accepted_fill_quantity(
            admitted_quantity,
            lot_size=lot_size if count is not None else 0.0,
            lot_count=count,
        )
        if abs(admitted_exact) < exact_quantity(minimum_quantity):
            return self._reject("below_minimum_quantity")

        price = prices[symbol]
        multiplier = multipliers[symbol]
        notional_price = price
        if intent.order_type is OrderType.LIMIT:
            assert intent.limit_price is not None
            limit_price = intent.limit_price
            marketable_at_open = (
                admitted_quantity > 0.0 and price <= limit_price
            ) or (admitted_quantity < 0.0 and price >= limit_price)
            if not marketable_at_open:
                notional_price = limit_price
        admitted_notional = abs(admitted_quantity) * notional_price * multiplier
        if admitted_notional + _TOLERANCE < minimum_notional:
            return self._reject("below_minimum_notional")

        projected = quantities.copy()
        projected[symbol] += admitted_quantity
        gross_notional = float(np.sum(np.abs(projected * prices * multipliers)))
        equity = book.portfolio_value
        if (
            book.insolvent
            or not math.isfinite(equity)
            or equity <= _TOLERANCE
            or gross_notional > equity * self.max_leverage + _TOLERANCE
        ):
            return self._reject("pretrade_leverage_exceeded")

        return AdmissionDecision(
            accepted=True,
            reason=None,
            admitted_quantity=admitted_quantity,
            admitted_notional=admitted_notional,
        )
