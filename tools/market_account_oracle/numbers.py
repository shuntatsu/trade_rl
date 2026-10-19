"""Standard-library numeric boundary, independent of execution code."""

from __future__ import annotations

import math
import re
from decimal import Context, Decimal, InvalidOperation
from fractions import Fraction

Number = int | float | str | Decimal
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def arithmetic_context() -> Context:
    """Create private precision and flags without depending on caller context."""
    return Context(prec=50)


def number(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError("expected a finite non-boolean decimal number")
    if isinstance(value, str) and _DECIMAL.fullmatch(value) is None:
        raise ValueError("invalid decimal number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("invalid decimal number") from error
    if not result.is_finite():
        raise ValueError("number must be finite")
    return result


def vector(
    values: tuple[Number, ...], size: int, *, positive: bool = False
) -> tuple[Decimal, ...]:
    if not isinstance(values, tuple) or len(values) != size:
        raise ValueError("expected a matching immutable vector")
    result = tuple(number(value) for value in values)
    if positive and any(value <= 0 for value in result):
        raise ValueError("vector must be strictly positive")
    return result


def quantity(value: object) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("quantity must be a canonical rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid rational quantity") from error
    if str(result) != value:
        raise ValueError("quantity must be canonical")
    projection(result)
    return result


def projection(value: Fraction) -> Decimal:
    """Report a decimal float inside the rational magnitude, without native helpers."""
    try:
        reported = float(value)
    except OverflowError as error:
        raise ValueError("quantity has no finite reporting projection") from error
    if not math.isfinite(reported):
        raise ValueError("quantity has no finite reporting projection")
    while abs(Fraction(str(reported))) > abs(value):
        reported = math.nextafter(reported, 0.0)
    return Decimal(str(reported))
