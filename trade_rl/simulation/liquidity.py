"""Deterministic symbol-level processing-bar liquidity allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import Sequence

from trade_rl._validation import require_sha256
from trade_rl.simulation.quantities import (
    accepted_fill_quantity,
    exact_quantity,
    project_quantity,
    quantize_quantity,
)

_TOLERANCE = 1e-12


class LiquidityAllocationError(ValueError):
    """Raised when a liquidity request or shared capacity is invalid."""


class LiquidityPriority(IntEnum):
    PREVIOUSLY_TRIGGERED_STOP = 0
    MARKET = 1
    NEWLY_TRIGGERED_STOP = 2
    OLDER_LIMIT = 3
    NEWER_LIMIT = 4


def _validate_digest(name: str, value: str) -> None:
    try:
        require_sha256(value, field=name)
    except ValueError as error:
        raise LiquidityAllocationError(str(error)) from error


@dataclass(frozen=True, slots=True)
class LiquidityRequest:
    order_id: str
    remaining_quantity: float
    execution_price: float
    available_volume_fraction: float
    priority: LiquidityPriority
    eligible_index: int
    reduce_only: bool = False
    minimum_notional: float | None = None
    minimum_quantity: float = 0.0
    maximum_quantity: float | None = None

    def __post_init__(self) -> None:
        for value in (
            self.minimum_notional,
            self.minimum_quantity,
            self.maximum_quantity,
        ):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise LiquidityAllocationError(
                    "request rules must be finite and nonnegative"
                )
        if self.maximum_quantity is not None and (
            self.maximum_quantity <= 0 or self.maximum_quantity < self.minimum_quantity
        ):
            raise LiquidityAllocationError("invalid request quantity bounds")
        if not isinstance(self.reduce_only, bool):
            raise LiquidityAllocationError("reduce_only must be a boolean")
        _validate_digest("order_id", self.order_id)
        if (
            not math.isfinite(self.remaining_quantity)
            or abs(self.remaining_quantity) <= _TOLERANCE
        ):
            raise LiquidityAllocationError(
                "remaining_quantity must be finite and non-zero"
            )
        if not math.isfinite(self.execution_price) or self.execution_price <= 0.0:
            raise LiquidityAllocationError(
                "execution_price must be finite and positive"
            )
        if (
            not math.isfinite(self.available_volume_fraction)
            or not 0.0 <= self.available_volume_fraction <= 1.0
        ):
            raise LiquidityAllocationError(
                "available volume fraction must be within [0, 1]"
            )
        if not isinstance(self.priority, LiquidityPriority):
            raise LiquidityAllocationError("priority must be a LiquidityPriority")
        if (
            isinstance(self.eligible_index, bool)
            or not isinstance(self.eligible_index, int)
            or self.eligible_index < 0
        ):
            raise LiquidityAllocationError(
                "eligible_index must be a non-negative integer"
            )

    @property
    def priority_key(self) -> tuple[int, int, str]:
        return (int(self.priority), self.eligible_index, self.order_id)


@dataclass(frozen=True, slots=True)
class LiquidityAllocation:
    order_id: str
    requested_quantity: float
    filled_quantity: float
    requested_notional: float
    filled_notional: float
    capacity_before: float
    accessible_capacity_notional: float
    capacity_after: float
    participation_rate: float
    no_fill_reason: str | None = None
    filled_lot_count: int | None = None
    lot_size: float = 0.0
    reduce_only_exhausted: bool = False


@dataclass(frozen=True, slots=True)
class SymbolCapacityEvidence:
    processing_volume: float
    capacity_reference_price: float
    contract_multiplier: float
    participation_limit: float
    market_notional: float
    initial_capacity_notional: float
    consumed_capacity_notional: float
    remaining_capacity_notional: float


def _capacity_quantity(
    request: LiquidityRequest,
    capacity: float,
    lot_size: float,
    multiplier: float,
    position: Fraction | None = None,
    quantity_capacity: float | None = None,
) -> tuple[float, int | None]:
    quantity, count = quantize_quantity(request.remaining_quantity, lot_size)
    if request.reduce_only:
        assert position is not None and position * exact_quantity(quantity) <= 0
        if count is None:
            quantity = math.copysign(
                min(abs(quantity), project_quantity(abs(position))), quantity
            )
        else:
            bound = abs(position) // exact_quantity(lot_size)
            count = min(abs(count), bound) * (-1 if count < 0 else 1)
            quantity = project_quantity(count * exact_quantity(lot_size))
    if quantity_capacity is not None:
        if not math.isfinite(quantity_capacity) or quantity_capacity < 0.0:
            raise LiquidityAllocationError(
                "quantity_capacity must be finite and non-negative"
            )
        if count is None:
            quantity = math.copysign(
                min(abs(quantity), quantity_capacity),
                quantity,
            )
        else:
            step = exact_quantity(lot_size)
            quantity_bound = exact_quantity(quantity_capacity) // step
            count = min(abs(count), quantity_bound) * (-1 if count < 0 else 1)
            quantity = project_quantity(count * step)
    if abs(quantity) * request.execution_price * multiplier <= capacity:
        return quantity, count
    if count is None:
        raw = capacity / (request.execution_price * multiplier)
        return math.copysign(min(raw, abs(quantity)), quantity), None

    # Capacity-derived division is not a quantity authority. Search integer
    # lots using the same monetary arithmetic as the final allocation, with
    # the original (strictly quantized) request as the upper bound.
    step = exact_quantity(lot_size)
    lower, upper = 0, abs(count) - 1
    while lower < upper:
        middle = (lower + upper + 1) // 2
        projected = project_quantity(middle * step)
        if projected * request.execution_price * multiplier <= capacity:
            lower = middle
        else:
            upper = middle - 1
    signed_count = lower if count > 0 else -lower
    return project_quantity(signed_count * step), signed_count


def _zero_allocation(
    request: LiquidityRequest,
    *,
    requested_notional: float,
    capacity_before: float,
    accessible_capacity: float,
    reason: str,
) -> LiquidityAllocation:
    return LiquidityAllocation(
        order_id=request.order_id,
        requested_quantity=request.remaining_quantity,
        filled_quantity=0.0,
        requested_notional=requested_notional,
        filled_notional=0.0,
        capacity_before=capacity_before,
        accessible_capacity_notional=accessible_capacity,
        capacity_after=capacity_before,
        participation_rate=0.0,
        no_fill_reason=reason,
    )


def allocate_symbol_capacity(
    requests: Sequence[LiquidityRequest],
    *,
    processing_volume: float,
    processing_market_notional: float | None = None,
    processing_quantity_capacity: float | None = None,
    price: float,
    contract_multiplier: float,
    participation_limit: float,
    lot_size: float,
    minimum_notional: float,
    initial_position: Fraction | None = None,
) -> tuple[tuple[LiquidityAllocation, ...], SymbolCapacityEvidence]:
    """Allocate one deterministic shared capacity pool among symbol orders."""

    for name, value in (
        ("processing_volume", processing_volume),
        ("price", price),
        ("contract_multiplier", contract_multiplier),
        ("participation_limit", participation_limit),
        ("lot_size", lot_size),
        ("minimum_notional", minimum_notional),
    ):
        if not math.isfinite(value):
            raise LiquidityAllocationError(f"{name} must be finite")
    if processing_volume < 0.0:
        raise LiquidityAllocationError("processing_volume must be non-negative")
    if processing_market_notional is not None and (
        not math.isfinite(processing_market_notional)
        or processing_market_notional < 0.0
    ):
        raise LiquidityAllocationError(
            "processing_market_notional must be finite and non-negative"
        )
    if processing_quantity_capacity is not None and (
        not math.isfinite(processing_quantity_capacity)
        or processing_quantity_capacity < 0.0
    ):
        raise LiquidityAllocationError(
            "processing_quantity_capacity must be finite and non-negative"
        )
    if price <= 0.0:
        raise LiquidityAllocationError("price must be positive")
    if contract_multiplier <= 0.0:
        raise LiquidityAllocationError("contract_multiplier must be positive")
    if not 0.0 < participation_limit <= 1.0:
        raise LiquidityAllocationError("participation_limit must be within (0, 1]")
    if lot_size < 0.0:
        raise LiquidityAllocationError("lot_size must be non-negative")
    if minimum_notional < 0.0:
        raise LiquidityAllocationError("minimum_notional must be non-negative")

    ordered = tuple(sorted(requests, key=lambda request: request.priority_key))
    if any(request.reduce_only for request in ordered) and not isinstance(
        initial_position, Fraction
    ):
        raise LiquidityAllocationError(
            "reduce-only allocation requires exact Fraction initial_position"
        )
    position = initial_position
    order_ids = tuple(request.order_id for request in ordered)
    if len(order_ids) != len(set(order_ids)):
        raise LiquidityAllocationError(
            "duplicate liquidity request order IDs are not allowed"
        )

    market_notional = (
        processing_volume * price * contract_multiplier
        if processing_market_notional is None
        else processing_market_notional
    )
    initial_capacity = market_notional * participation_limit
    remaining_capacity = initial_capacity
    initial_quantity_capacity = (
        None
        if processing_quantity_capacity is None
        else processing_quantity_capacity * participation_limit
    )
    remaining_quantity_capacity = initial_quantity_capacity
    allocations: list[LiquidityAllocation] = []

    for request in ordered:
        capacity_before = remaining_capacity
        fraction_cap = initial_capacity * request.available_volume_fraction
        accessible_capacity = min(capacity_before, fraction_cap)
        accessible_quantity_capacity = None
        if initial_quantity_capacity is not None:
            assert remaining_quantity_capacity is not None
            fraction_quantity_cap = (
                initial_quantity_capacity * request.available_volume_fraction
            )
            accessible_quantity_capacity = min(
                remaining_quantity_capacity,
                fraction_quantity_cap,
            )
        requested_notional = (
            abs(request.remaining_quantity)
            * request.execution_price
            * contract_multiplier
        )

        if request.maximum_quantity is not None and abs(
            exact_quantity(request.remaining_quantity)
        ) > exact_quantity(request.maximum_quantity):
            allocations.append(
                _zero_allocation(
                    request,
                    requested_notional=requested_notional,
                    capacity_before=capacity_before,
                    accessible_capacity=accessible_capacity,
                    reason="above_maximum_quantity",
                )
            )
            continue

        if request.reduce_only:
            assert position is not None
            if position * exact_quantity(request.remaining_quantity) >= 0:
                allocations.append(
                    _zero_allocation(
                        request,
                        requested_notional=requested_notional,
                        capacity_before=capacity_before,
                        accessible_capacity=accessible_capacity,
                        reason="reduce_only_exhausted",
                    )
                )
                continue

        if request.available_volume_fraction <= _TOLERANCE:
            allocations.append(
                _zero_allocation(
                    request,
                    requested_notional=requested_notional,
                    capacity_before=capacity_before,
                    accessible_capacity=accessible_capacity,
                    reason="zero_volume_fraction",
                )
            )
            continue
        if accessible_capacity <= _TOLERANCE or (
            accessible_quantity_capacity is not None
            and accessible_quantity_capacity <= _TOLERANCE
        ):
            allocations.append(
                _zero_allocation(
                    request,
                    requested_notional=requested_notional,
                    capacity_before=capacity_before,
                    accessible_capacity=accessible_capacity,
                    reason="no_capacity",
                )
            )
            continue

        filled_quantity, filled_lot_count = _capacity_quantity(
            request,
            accessible_capacity,
            lot_size,
            contract_multiplier,
            position,
            accessible_quantity_capacity,
        )
        if abs(filled_quantity) > abs(request.remaining_quantity):
            raise LiquidityAllocationError(
                "rounded fill exceeds the order remaining quantity"
            )
        if abs(filled_quantity) <= _TOLERANCE:
            allocations.append(
                _zero_allocation(
                    request,
                    requested_notional=requested_notional,
                    capacity_before=capacity_before,
                    accessible_capacity=accessible_capacity,
                    reason="below_lot_size",
                )
            )
            continue

        # Currency retains the shared float convention. BookState debits this
        # same projected signed fill; exact state is the quantity authority.
        exact_notional = (
            abs(filled_quantity) * request.execution_price * contract_multiplier
        )
        filled_exact = accepted_fill_quantity(
            filled_quantity,
            lot_size=lot_size if filled_lot_count is not None else 0.0,
            lot_count=filled_lot_count,
        )
        if abs(filled_exact) < exact_quantity(request.minimum_quantity):
            allocations.append(
                _zero_allocation(
                    request,
                    requested_notional=requested_notional,
                    capacity_before=capacity_before,
                    accessible_capacity=accessible_capacity,
                    reason="below_minimum_quantity",
                )
            )
            continue
        applicable_minimum = (
            minimum_notional
            if request.minimum_notional is None
            else request.minimum_notional
        )
        if exact_notional + _TOLERANCE < applicable_minimum:
            allocations.append(
                _zero_allocation(
                    request,
                    requested_notional=requested_notional,
                    capacity_before=capacity_before,
                    accessible_capacity=accessible_capacity,
                    reason="below_minimum_notional",
                )
            )
            continue
        if exact_notional > capacity_before + _TOLERANCE:
            raise LiquidityAllocationError(
                "rounded fill exceeds remaining symbol capacity"
            )

        remaining_capacity = max(0.0, capacity_before - exact_notional)
        if remaining_quantity_capacity is not None:
            remaining_quantity_capacity = max(
                0.0,
                remaining_quantity_capacity - abs(filled_quantity),
            )
        if position is not None:
            position += accepted_fill_quantity(
                filled_quantity,
                lot_size=lot_size if filled_lot_count is not None else 0.0,
                lot_count=filled_lot_count,
            )
        participation_rate = (
            0.0 if market_notional <= _TOLERANCE else exact_notional / market_notional
        )
        allocations.append(
            LiquidityAllocation(
                order_id=request.order_id,
                requested_quantity=request.remaining_quantity,
                filled_quantity=filled_quantity,
                requested_notional=requested_notional,
                filled_notional=exact_notional,
                capacity_before=capacity_before,
                accessible_capacity_notional=accessible_capacity,
                capacity_after=remaining_capacity,
                participation_rate=participation_rate,
                no_fill_reason=None,
                filled_lot_count=filled_lot_count,
                lot_size=lot_size if filled_lot_count is not None else 0.0,
                reduce_only_exhausted=request.reduce_only and position == 0,
            )
        )

    consumed = initial_capacity - remaining_capacity
    filled_total = sum(allocation.filled_notional for allocation in allocations)
    if initial_quantity_capacity is not None:
        assert remaining_quantity_capacity is not None
        filled_quantity_total = sum(
            abs(allocation.filled_quantity) for allocation in allocations
        )
        consumed_quantity = initial_quantity_capacity - remaining_quantity_capacity
        quantity_tolerance = max(
            _TOLERANCE,
            8.0 * math.ulp(initial_quantity_capacity),
        )
        if not math.isclose(
            consumed_quantity,
            filled_quantity_total,
            rel_tol=0.0,
            abs_tol=quantity_tolerance,
        ):
            raise LiquidityAllocationError(
                "native quantity capacity accounting is inconsistent"
            )
        if filled_quantity_total > initial_quantity_capacity + quantity_tolerance:
            raise LiquidityAllocationError(
                "native quantity capacity was over-allocated"
            )
    capacity_tolerance = max(
        1e-9,
        8.0 * math.ulp(initial_capacity),
        8.0 * math.ulp(remaining_capacity),
    )
    if not math.isclose(
        consumed,
        filled_total,
        rel_tol=0.0,
        abs_tol=capacity_tolerance,
    ):
        raise LiquidityAllocationError("symbol capacity accounting is inconsistent")
    if filled_total > initial_capacity + capacity_tolerance:
        raise LiquidityAllocationError("symbol capacity was over-allocated")

    evidence = SymbolCapacityEvidence(
        processing_volume=processing_volume,
        capacity_reference_price=price,
        contract_multiplier=contract_multiplier,
        participation_limit=participation_limit,
        market_notional=market_notional,
        initial_capacity_notional=initial_capacity,
        consumed_capacity_notional=consumed,
        remaining_capacity_notional=remaining_capacity,
    )
    return tuple(allocations), evidence
