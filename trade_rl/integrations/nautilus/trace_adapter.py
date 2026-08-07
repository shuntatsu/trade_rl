"""Canonicalize Nautilus order fills without leaking upstream objects downstream."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable

from trade_rl.simulation.execution_canonicalization import CanonicalFillSignature


@dataclass(frozen=True, slots=True)
class NautilusCanonicalFillResult:
    fills: tuple[CanonicalFillSignature, ...]
    fee_minor: int


def canonicalize_nautilus_fill_events(
    events: Iterable[Any],
    *,
    price_tick: Decimal,
    lot_size: Decimal,
    currency_precision: int,
) -> NautilusCanonicalFillResult:
    """Convert ``OrderFilled``-compatible objects to integer canonical identity."""

    if not price_tick.is_finite() or price_tick <= 0:
        raise ValueError("price_tick must be finite and positive")
    if not lot_size.is_finite() or lot_size <= 0:
        raise ValueError("lot_size must be finite and positive")
    if (
        isinstance(currency_precision, bool)
        or not isinstance(currency_precision, int)
        or currency_precision < 0
        or currency_precision > 18
    ):
        raise ValueError("currency_precision must be an integer within [0, 18]")

    position_lots = 0
    fee_minor = 0
    fills: list[CanonicalFillSignature] = []
    scale = Decimal(10) ** currency_precision

    for event in events:
        price = _decimal_value(event.last_px, name="last_px")
        quantity = _decimal_value(event.last_qty, name="last_qty")
        side = _order_side_name(event.order_side)
        if side == "BUY":
            signed_quantity = quantity
        elif side == "SELL":
            signed_quantity = -quantity
        else:
            raise ValueError(f"unsupported Nautilus order side: {event.order_side!r}")

        price_ticks = _exact_units(price, price_tick, name="price")
        quantity_lots = _exact_units(signed_quantity, lot_size, name="quantity")
        position_lots += quantity_lots

        fills.append(
            CanonicalFillSignature(
                sequence=len(fills) + 1,
                timestamp_ns=int(event.ts_event),
                price_ticks=price_ticks,
                quantity_lots=quantity_lots,
                position_lots=position_lots,
            )
        )

        commission = getattr(event, "commission", None)
        if commission is not None:
            commission_decimal = _decimal_value(commission, name="commission")
            scaled = commission_decimal * scale
            integral = scaled.to_integral_value()
            if scaled != integral:
                raise ValueError("commission is not aligned to settlement minor unit")
            fee_minor += int(integral)

    return NautilusCanonicalFillResult(fills=tuple(fills), fee_minor=fee_minor)


def _order_side_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    text = str(value).upper()
    if text.endswith("BUY"):
        return "BUY"
    if text.endswith("SELL"):
        return "SELL"
    return text


def _decimal_value(value: Any, *, name: str) -> Decimal:
    if hasattr(value, "as_decimal"):
        result = value.as_decimal()
    else:
        result = Decimal(str(value))
    if not isinstance(result, Decimal):
        result = Decimal(str(result))
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _exact_units(value: Decimal, increment: Decimal, *, name: str) -> int:
    units = value / increment
    integral = units.to_integral_value()
    if units != integral:
        raise ValueError(f"{name} is not aligned to canonical increment")
    return int(integral)


__all__ = [
    "NautilusCanonicalFillResult",
    "canonicalize_nautilus_fill_events",
]
