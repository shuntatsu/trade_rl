"""Deterministic pre-trade admission decisions for persistent orders."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

import numpy as np

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders.model import OrderIntent, OrderType
from trade_rl.simulation.quantities import (
    accepted_fill_quantity,
    exact_quantity,
    quantize_quantity,
)

_TOLERANCE = 1e-12


class _AdmissionBookView(Protocol):
    quantities: np.ndarray
    insolvent: bool

    @property
    def contract_multipliers(self) -> np.ndarray | None: ...

    @property
    def portfolio_value(self) -> float: ...


def _projection_tolerance(price: float, projected: float, tick: float) -> float:
    """Allow one ULP on resolvable grids; retain dense-grid projection slack."""

    ulp = max(math.ulp(price), math.ulp(projected))
    # When a tick spans more than eight ULPs, a multi-ULP discrepancy is
    # economically distinguishable from one binary64 rounding step.
    return ulp if tick > 8.0 * ulp else 4.0 * ulp


def _needs_exact_tick_count(price: float, tick: float, quotient: float) -> bool:
    """Use decimal ticks where binary64 cannot preserve the grid identity."""

    return (
        not math.isfinite(quotient)
        or tick < sys.float_info.min
        or tick <= 8.0 * math.ulp(price)
    )


def price_on_tick_grid(price: float, tick_size: float) -> bool:
    """Return whether a float price represents a tick-grid point."""

    resolved_price = float(price)
    resolved_tick = float(tick_size)
    if (
        not math.isfinite(resolved_price)
        or resolved_price <= 0.0
        or not math.isfinite(resolved_tick)
        or resolved_tick < 0.0
    ):
        return False
    if resolved_tick == 0.0:
        return True

    price_exact = Fraction(str(resolved_price))
    tick_exact = Fraction(str(resolved_tick))
    if price_exact % tick_exact == 0:
        return True

    # With multiple decimal ticks per binary64 ULP, the shortest decimal
    # spelling may sit exactly between two ticks even though an integer
    # decimal-grid tick correctly rounds to this floating-point price.
    # Only admit the exact binary64 image of a positive tick count: neither
    # a broad ULP tolerance nor a decimal half-tick guess may prove it.
    if resolved_tick <= 2.0 * math.ulp(resolved_price):
        candidate_count = round(Fraction.from_float(resolved_price) / tick_exact)
        if candidate_count > 0:
            try:
                if float(candidate_count * tick_exact) == resolved_price:
                    return True
            except OverflowError:
                # At float64's largest finite price, the closest exact tick
                # can round beyond the representable range. That cannot
                # establish membership; never leak an OverflowError.
                return False

    # A finite float quotient can still lose more than one whole tick near
    # binary64's unit-precision limit. Use the decimal authority when one
    # tick is subnormal or within the 4-ULP tolerance, or division overflows.
    quotient = resolved_price / resolved_tick
    use_exact = _needs_exact_tick_count(resolved_price, resolved_tick, quotient)
    nearest_count = round(price_exact / tick_exact) if use_exact else round(quotient)
    if nearest_count <= 0:
        return False
    # A float count * float tick can move by an ULP at large notionals.
    # Project the accepted decimal tick count in the canonical rational grid.
    try:
        projected = float(nearest_count * tick_exact)
    except OverflowError:
        return False
    # A positive off-grid quote cannot be represented by tick zero, and an
    # infinite nearest price must never grant membership via inf <= inf.
    if nearest_count <= 0 or not math.isfinite(projected) or projected <= 0.0:
        return False
    # Four ULPs can exceed half a tick at binary64's subnormal boundary.
    # Such an allowance is not representation noise: it would grant a
    # genuinely off-tick midpoint a valid grid identity.
    if 2 * abs(price_exact - nearest_count * tick_exact) >= tick_exact:
        return False
    return abs(resolved_price - projected) <= _projection_tolerance(
        resolved_price,
        projected,
        resolved_tick,
    )


def snap_price_to_tick(
    price: float,
    tick_size: float,
    *,
    round_up: bool,
) -> float:
    """Snap a generated order bound conservatively to a tick grid."""

    resolved_price = float(price)
    resolved_tick = float(tick_size)
    if not math.isfinite(resolved_price) or resolved_price <= 0.0:
        raise ValueError("price must be finite and positive")
    if not math.isfinite(resolved_tick) or resolved_tick < 0.0:
        raise ValueError("tick_size must be finite and non-negative")
    if resolved_tick == 0.0:
        return resolved_price

    # The decimal venue grid is authoritative. A point already on that
    # grid must remain unchanged, rather than drifting by an ULP through
    # a binary64 count * tick projection (possibly against round_up).
    if Fraction(str(resolved_price)) % Fraction(str(resolved_tick)) == 0:
        return resolved_price

    exact_tick = Fraction(str(resolved_tick))
    quotient = resolved_price / resolved_tick
    use_exact = _needs_exact_tick_count(resolved_price, resolved_tick, quotient)
    if use_exact:
        # Floating division can lose complete tick counts near binary64's
        # integer precision limit. Use the canonical decimal grid instead.
        nearest_count = round(Fraction(str(resolved_price)) / exact_tick)
    else:
        nearest_count = round(quotient)
    # Even in the finite-quotient path, never project via float count*tick.
    try:
        nearest = float(nearest_count * exact_tick)
    except OverflowError:
        nearest = math.inf
    # The float-ULP tolerance is only meaningful for a *positive, finite*
    # grid point. Count zero and an infinite nearest must use conservative
    # directional snapping, not become false grid matches.
    nearest_is_valid = nearest_count > 0 and math.isfinite(nearest) and nearest > 0.0
    if nearest_is_valid:
        tolerance = _projection_tolerance(resolved_price, nearest, resolved_tick)
        if abs(resolved_price - nearest) > tolerance:
            nearest_is_valid = False
        else:
            # Any floated nearest price within the ULP tolerance must still
            # be closer than half a decimal tick. The former 2*tolerance
            # gate allowed a false nearest just above that boundary and
            # could reverse conservative ceil/floor snapping.
            exact_tick = Fraction(str(resolved_tick))
            nearest_is_valid = (
                2 * abs(Fraction(str(resolved_price)) - nearest_count * exact_tick)
                < exact_tick
            )
    if nearest_is_valid:
        # The current float is already an accepted grid representation. A
        # one-ULP projection cannot be allowed to reverse the order bound.
        if (round_up and nearest < resolved_price) or (
            not round_up and nearest > resolved_price
        ):
            return resolved_price
        count = nearest_count
    else:
        # Preserve the actual binary64 order bound: decimal str(price) may
        # lie up to half an ULP on the opposite side of its float value.
        direction_ratio = Fraction.from_float(resolved_price) / exact_tick
        count = math.ceil(direction_ratio) if round_up else math.floor(direction_ratio)

    if count <= 0:
        raise ValueError("positive order price has no valid tick-grid projection")
    try:
        snapped = float(count * exact_tick)
    except OverflowError as error:
        raise ValueError(
            "tick-grid projection must remain finite and positive"
        ) from error
    if not math.isfinite(snapped) or snapped <= 0.0:
        raise ValueError("tick-grid projection must remain finite and positive")
    return float(snapped)


class OrderAdmissionError(ValueError):
    """Raised when admission inputs are structurally invalid."""


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    accepted: bool
    reason: str | None
    admitted_quantity: float
    admitted_notional: float
    admitted_exact_quantity: Fraction


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
            admitted_exact_quantity=Fraction(0),
        )

    def evaluate(
        self,
        intent: OrderIntent,
        *,
        remaining_quantity: float | None = None,
        book: BookState | _AdmissionBookView,
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
        allow_over_cap_reduce_only: bool = False,
    ) -> AdmissionDecision:
        """Return an explicit admission or economic rejection reason."""

        if actual_position is not None and not isinstance(actual_position, Fraction):
            raise OrderAdmissionError("actual_position must be an exact Fraction")
        if not isinstance(allow_over_cap_reduce_only, bool):
            raise OrderAdmissionError("allow_over_cap_reduce_only must be a boolean")

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
        bound_price = None
        if intent.order_type is OrderType.LIMIT:
            bound_price = intent.limit_price
        elif intent.order_type is OrderType.STOP_MARKET:
            bound_price = intent.stop_price
        if bound_price is not None and not price_on_tick_grid(bound_price, tick_size):
            return self._reject("price_not_on_tick")
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
            if actual_position is None:
                if not isinstance(book, BookState):
                    raise OrderAdmissionError(
                        "actual_position is required for a projected book"
                    )
                position = book.exact_quantities[symbol]
            else:
                position = actual_position
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
            buy_marketable = admitted_quantity > 0.0 and price <= limit_price
            sell_marketable = admitted_quantity < 0.0 and price >= limit_price
            if not (buy_marketable or sell_marketable):
                notional_price = limit_price
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            admitted_notional = abs(admitted_quantity) * notional_price * multiplier
        if not math.isfinite(admitted_notional):
            return self._reject("pretrade_leverage_exceeded")
        # BookState rejects a nonzero fill whose left-associated notional
        # underflows to zero. Reject at admission rather than fail during fill.
        if admitted_notional == 0.0:
            return self._reject("unrepresentable_notional")
        if admitted_notional + _TOLERANCE < minimum_notional:
            return self._reject("below_minimum_notional")

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            projected = quantities.copy()
            projected[symbol] += admitted_quantity
            projected_values = projected * prices * multipliers
            gross_notional = float(np.sum(np.abs(projected_values)))
            current_gross_notional = float(
                np.sum(np.abs(quantities * prices * multipliers))
            )
        if (
            not np.isfinite(projected).all()
            or not math.isfinite(gross_notional)
            or not math.isfinite(current_gross_notional)
        ):
            return self._reject("pretrade_leverage_exceeded")
        # A nonmarketable limit uses its limit for minimum notional, but the
        # sequential admission book reserves at the processing-open mark.
        # Reject before the caller emits an eligible event if that reservation
        # would have an unrepresentable nonzero position value.
        if np.any((projected != 0.0) & (projected_values == 0.0)):
            return self._reject("unrepresentable_notional")
        equity = book.portfolio_value
        leverage_exceeded = gross_notional > equity * self.max_leverage + _TOLERANCE
        over_cap_reduction = (
            allow_over_cap_reduce_only
            and intent.reduce_only
            and intent.order_type is OrderType.MARKET
            and gross_notional < current_gross_notional - _TOLERANCE
        )
        if (
            book.insolvent
            or not math.isfinite(equity)
            or equity <= _TOLERANCE
            or (leverage_exceeded and not over_cap_reduction)
        ):
            return self._reject("pretrade_leverage_exceeded")

        return AdmissionDecision(
            accepted=True,
            reason=None,
            admitted_quantity=admitted_quantity,
            admitted_notional=admitted_notional,
            admitted_exact_quantity=admitted_exact,
        )
