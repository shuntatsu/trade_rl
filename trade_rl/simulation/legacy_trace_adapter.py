"""Canonicalization helpers for the maintained legacy stateful execution engine."""

from __future__ import annotations

import math
from typing import Iterable

from trade_rl.simulation.execution_canonicalization import CanonicalFillSignature
from trade_rl.simulation.orders import OrderEvent


def canonicalize_legacy_fill_events(
    events: Iterable[OrderEvent],
    *,
    price_tick: float,
    lot_size: float,
) -> tuple[CanonicalFillSignature, ...]:
    """Convert legacy fill events to integer structural signatures.

    Fees and funding are intentionally excluded here because the legacy stateful
    engine records them at interval/accounting scope, not on ``OrderEvent``. They
    belong in a separate ``CanonicalEconomicClosure``.
    """

    if not math.isfinite(price_tick) or price_tick <= 0.0:
        raise ValueError("price_tick must be finite and positive")
    if not math.isfinite(lot_size) or lot_size <= 0.0:
        raise ValueError("lot_size must be finite and positive")

    fills: list[CanonicalFillSignature] = []
    position_lots = 0
    for event in events:
        if event.event_type not in {"partial_fill", "filled"}:
            continue
        if abs(event.filled_quantity) <= 1e-12:
            continue
        if event.execution_price is None:
            raise ValueError("legacy fill event must define execution_price")
        quantity_lots = _to_grid_units(event.filled_quantity, lot_size, "quantity")
        price_ticks = _to_grid_units(event.execution_price, price_tick, "price")
        position_lots += quantity_lots
        fills.append(
            CanonicalFillSignature(
                sequence=len(fills) + 1,
                timestamp_ns=event.timestamp_ns,
                price_ticks=price_ticks,
                quantity_lots=quantity_lots,
                position_lots=position_lots,
            )
        )
    return tuple(fills)


def _to_grid_units(value: float, increment: float, name: str) -> int:
    units = value / increment
    rounded = round(units)
    if not math.isclose(units, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} is not aligned to canonical increment")
    return int(rounded)


__all__ = ["canonicalize_legacy_fill_events"]
