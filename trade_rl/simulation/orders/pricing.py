"""Tick-grid validation and conservative order-bound snapping."""

from __future__ import annotations

import math
from fractions import Fraction


def _validated_price_and_tick(price: float, tick_size: float) -> tuple[float, float]:
    resolved_price = float(price)
    resolved_tick = float(tick_size)
    if not math.isfinite(resolved_price) or resolved_price <= 0.0:
        raise ValueError("price must be finite and positive")
    if not math.isfinite(resolved_tick) or resolved_tick < 0.0:
        raise ValueError("tick_size must be finite and non-negative")
    return resolved_price, resolved_tick


def _projection_tolerance(price: float, projected: float) -> float:
    return 4.0 * max(math.ulp(price), math.ulp(projected))


def price_on_tick_grid(price: float, tick_size: float) -> bool:
    """Return whether a float price represents a tick-grid point."""

    resolved_price, resolved_tick = _validated_price_and_tick(price, tick_size)
    if resolved_tick == 0.0:
        return True

    price_exact = Fraction(str(resolved_price))
    tick_exact = Fraction(str(resolved_tick))
    if price_exact % tick_exact == 0:
        return True

    nearest_count = round(resolved_price / resolved_tick)
    projected = nearest_count * resolved_tick
    return abs(resolved_price - projected) <= _projection_tolerance(
        resolved_price,
        projected,
    )


def snap_price_to_tick(
    price: float,
    tick_size: float,
    *,
    round_up: bool,
) -> float:
    """Snap a generated order bound conservatively to the configured tick grid."""

    resolved_price, resolved_tick = _validated_price_and_tick(price, tick_size)
    if resolved_tick == 0.0:
        return resolved_price

    ratio = resolved_price / resolved_tick
    nearest_count = round(ratio)
    nearest = nearest_count * resolved_tick
    if abs(resolved_price - nearest) <= _projection_tolerance(
        resolved_price,
        nearest,
    ):
        count = nearest_count
    else:
        count = math.ceil(ratio) if round_up else math.floor(ratio)

    if count <= 0:
        raise ValueError("positive order price has no valid tick-grid projection")
    snapped = count * resolved_tick
    if not math.isfinite(snapped) or snapped <= 0.0:
        raise ValueError("tick-grid projection must remain finite and positive")
    return float(snapped)


__all__ = ["price_on_tick_grid", "snap_price_to_tick"]
