"""Exact decimal quantity arithmetic and conservative float projections."""

from __future__ import annotations

import math
from fractions import Fraction


def exact_quantity(value: float) -> Fraction:
    if not math.isfinite(value):
        raise ValueError("quantity must be finite")
    return Fraction(str(float(value)))


def parse_quantity(value: str) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("exact quantity must be a canonical rational string")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid exact quantity") from error
    if str(parsed) != value:
        raise ValueError("exact quantity must be canonical")
    project_quantity(parsed)
    return parsed


def project_quantity(value: Fraction) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("quantity is outside finite float range") from error
    if not math.isfinite(result):
        raise ValueError("quantity is outside finite float range")
    while abs(exact_quantity(result)) > abs(value):
        result = math.nextafter(result, 0.0)
    return result


def quantize_quantity(value: float, lot_size: float) -> tuple[float, int | None]:
    quantity, step = exact_quantity(value), exact_quantity(lot_size)
    if step < 0:
        raise ValueError("lot size must be non-negative")
    if not step:
        return value, None
    count = int(abs(quantity) // step) * (-1 if quantity < 0 else 1)
    return project_quantity(count * step), count


def accepted_fill_quantity(
    quantity: float, *, lot_size: float = 0.0, lot_count: int | None = None
) -> Fraction:
    value = exact_quantity(quantity)
    if lot_count is None:
        if lot_size != 0.0:
            raise ValueError("quantized fill requires its lot count")
        return value
    if isinstance(lot_count, bool) or not isinstance(lot_count, int) or not lot_count:
        raise ValueError("filled lot count must be a nonzero integer")
    step = exact_quantity(lot_size)
    if step <= 0:
        raise ValueError("quantized fill requires a positive lot size")
    exact = lot_count * step
    if project_quantity(exact) != quantity:
        raise ValueError("fill quantity differs from its exact lot evidence")
    return exact
