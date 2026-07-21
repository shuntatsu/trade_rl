"""Deterministic symbol-level processing-bar liquidity allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

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
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LiquidityAllocationError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class LiquidityRequest:
    order_id: str
    remaining_quantity: float
    execution_price: float
    available_volume_fraction: float
    priority: LiquidityPriority
    eligible_index: int

    def __post_init__(self) -> None:
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


def _rounded_toward_zero(quantity: float, lot_size: float) -> float:
    if lot_size <= 0.0:
        return quantity
    lots = math.floor((abs(quantity) + _TOLERANCE) / lot_size)
    rounded = lots * lot_size
    return math.copysign(rounded, quantity) if rounded > _TOLERANCE else 0.0


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
    price: float,
    contract_multiplier: float,
    participation_limit: float,
    lot_size: float,
    minimum_notional: float,
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
    allocations: list[LiquidityAllocation] = []

    for request in ordered:
        capacity_before = remaining_capacity
        fraction_cap = initial_capacity * request.available_volume_fraction
        accessible_capacity = min(capacity_before, fraction_cap)
        requested_notional = (
            abs(request.remaining_quantity)
            * request.execution_price
            * contract_multiplier
        )

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
        if accessible_capacity <= _TOLERANCE:
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

        candidate_notional = min(requested_notional, accessible_capacity)
        raw_quantity = math.copysign(
            candidate_notional / (request.execution_price * contract_multiplier),
            request.remaining_quantity,
        )
        filled_quantity = _rounded_toward_zero(raw_quantity, lot_size)
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

        exact_notional = (
            abs(filled_quantity) * request.execution_price * contract_multiplier
        )
        if exact_notional + _TOLERANCE < minimum_notional:
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
            )
        )

    consumed = initial_capacity - remaining_capacity
    filled_total = sum(allocation.filled_notional for allocation in allocations)
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
