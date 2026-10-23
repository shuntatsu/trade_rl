"""Independent MARKET price and raw capacity primitives, without replay authority."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext

from tools.market_account_oracle.numbers import Number, arithmetic_context, number


def _primitive(value: Number) -> Decimal:
    parsed = number(value)
    reported = float(parsed)
    if not math.isfinite(reported) or (parsed != 0 and reported == 0):
        raise ValueError("market primitives require a finite binary64 representation")
    return parsed


@dataclass(frozen=True, slots=True)
class MarketRow:
    """Explicit raw scalars for one selected symbol, with an absolute row index."""

    index: int
    open_price: Number
    close_price: Number
    volume: Number

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("market row index must be a non-negative integer")
        if _primitive(self.open_price) <= 0 or _primitive(self.close_price) <= 0:
            raise ValueError("market prices must be positive")
        if _primitive(self.volume) < 0:
            raise ValueError("raw volume must be non-negative")


@dataclass(frozen=True, slots=True)
class MarketFacts:
    execution_price: Decimal
    reference_index: int | None
    reference_price: Decimal
    raw_volume: Decimal
    market_notional: Decimal
    quantity_capacity: Decimal | None


def _rounded_open(
    raw_open: Decimal, tick: Decimal, direction: int, adverse: bool
) -> Decimal:
    # Preserve the declared binary64 division/round/multiply contract. Decimal
    # half-even would change a boundary such as 0.35 / 0.1 before rounding.
    price = float(raw_open)
    if tick:
        step = float(tick)
        scaled = price / step
        if not math.isfinite(scaled):
            raise ValueError("execution price divided by tick must remain finite")
        if adverse:
            units = math.ceil(scaled) if direction > 0 else math.floor(scaled)
        else:
            units = round(scaled)
        price = max(units * step, step)
    if not math.isfinite(price) or price <= 0:
        raise ValueError("rounded execution price must be finite and positive")
    return Decimal(str(price))


def market_order_facts(
    current: MarketRow,
    *,
    previous: MarketRow | None = None,
    volume_unit: str,
    contract_multiplier: Number,
    tick_size: Number = 0,
    direction: int,
    adverse_tick_rounding: bool = False,
    processing_bar_volume_capacity: bool = False,
) -> MarketFacts:
    """Derive MARKET open price and capacity from caller-bound raw primitives.

    No participation cap, order admission, fill allocation, OHLC trigger path,
    Dataset authentication or effective-rule lookup is implied. Capacity values
    are raw denominators, not an amount granted to an order.
    """
    if type(current) is not MarketRow:
        raise ValueError("requires an exact raw market row")
    current.__post_init__()
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("market order direction must be -1 or 1")
    if (
        type(adverse_tick_rounding) is not bool
        or type(processing_bar_volume_capacity) is not bool
    ):
        raise ValueError("market execution controls must be boolean")
    if type(volume_unit) is not str or volume_unit not in {
        "base_asset",
        "contracts",
        "quote_notional",
    }:
        raise ValueError("unknown raw market volume unit")
    multiplier = _primitive(contract_multiplier)
    tick = _primitive(tick_size)
    if multiplier <= 0 or tick < 0:
        raise ValueError("contract multiplier must be positive and tick non-negative")
    raw_open = number(current.open_price)
    price = _rounded_open(raw_open, tick, direction, adverse_tick_rounding)

    if processing_bar_volume_capacity:
        if previous is not None:
            raise ValueError("current-bar capacity does not consume a previous row")
        reference = current
        reference_price = raw_open
    elif current.index == 0:
        if previous is not None:
            raise ValueError("the first market row cannot have previous history")
        return MarketFacts(price, None, raw_open, Decimal(0), Decimal(0), None)
    else:
        if type(previous) is not MarketRow:
            raise ValueError("prior-bar capacity requires the adjacent previous row")
        previous.__post_init__()
        if previous.index != current.index - 1:
            raise ValueError("prior-bar capacity requires adjacent absolute indices")
        reference = previous
        reference_price = number(previous.close_price)

    volume = number(reference.volume)
    with localcontext(arithmetic_context()):
        if volume_unit == "quote_notional":
            notional, quantity_capacity = volume, None
        elif volume_unit == "base_asset":
            notional = volume * reference_price
            quantity_capacity = volume / multiplier
        else:
            notional = volume * multiplier * reference_price
            quantity_capacity = volume
    return MarketFacts(
        price, reference.index, reference_price, volume, notional, quantity_capacity
    )
