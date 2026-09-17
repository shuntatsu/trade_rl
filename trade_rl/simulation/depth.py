"""Conservative recorded-depth paper fills on the canonical account.

One call consumes one instrument's snapshot. The journal must prevent reuse of
that snapshot's capacity. Displayed liquidity is not a guaranteed exchange fill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction

from trade_rl._validation import require_aware_datetime
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.quantities import exact_quantity, project_quantity

Levels = tuple[tuple[float, float], ...]


def _finite(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


@dataclass(frozen=True)
class DepthOrderRules:
    lot_size: float
    minimum_quantity: float
    maximum_quantity: float
    minimum_notional: float
    maximum_notional: float | None

    def __post_init__(self) -> None:
        for field in (
            "lot_size",
            "minimum_quantity",
            "maximum_quantity",
            "minimum_notional",
        ):
            _finite(getattr(self, field), field)
        if (
            self.lot_size <= 0
            or self.minimum_quantity < 0
            or self.maximum_quantity <= 0
            or self.maximum_quantity < self.minimum_quantity
            or self.minimum_notional < 0
        ):
            raise ValueError("invalid depth order rules")
        if self.maximum_notional is not None:
            _finite(self.maximum_notional, "maximum_notional")
            if (
                self.maximum_notional <= 0
                or self.maximum_notional < self.minimum_notional
            ):
                raise ValueError("invalid maximum notional")


@dataclass(frozen=True)
class DepthFill:
    filled_lots: int
    unfilled_lots: int
    price: float | None
    fee: float
    reason: str | None


def _validate_depth(bids: Levels, asks: Levels) -> None:
    for levels, descending in ((bids, True), (asks, False)):
        if not levels:
            raise ValueError("both sides of the depth must be present")
        previous = None
        for price, quantity in levels:
            _finite(price, "depth price")
            _finite(quantity, "depth quantity")
            if price <= 0 or quantity <= 0:
                raise ValueError("depth prices and quantities must be positive")
            if previous is not None and (
                price >= previous if descending else price <= previous
            ):
                raise ValueError("depth prices must be strictly ordered")
            previous = price
    if bids[0][0] >= asks[0][0]:
        raise ValueError("depth must not be crossed or locked")


def execute_depth_order(
    *,
    book: BookState,
    symbol_index: int,
    lot_count: int,
    rules: DepthOrderRules,
    bids: Levels,
    asks: Levels,
    decision_at: datetime,
    quote_at: datetime,
    execution_at: datetime,
    fee_bps: float,
    adverse_bps: float = 5.0,
    depth_fraction: float = 0.1,
) -> DepthFill:
    """Match signed lots against later recorded depth, preserving unfilled lots.

    Notional admission uses the adverse-adjusted best quote. A maximum-notional
    guard also checks the resulting fill. Venue average-price filters and account
    restrictions are outside this paper primitive. Marks come from the book;
    fill prices never replace them or create transient valuation peaks.
    """
    for name, timestamp in (
        ("decision_at", decision_at),
        ("quote_at", quote_at),
        ("execution_at", execution_at),
    ):
        require_aware_datetime(timestamp, field=name)
    if not decision_at < quote_at <= execution_at:
        raise ValueError("decision must precede quote receipt and execution")
    if (execution_at - quote_at).total_seconds() > 5 or (
        execution_at - decision_at
    ).total_seconds() > 10:
        raise ValueError("paper execution quote or decision is stale")
    if isinstance(lot_count, bool) or not isinstance(lot_count, int) or not lot_count:
        raise ValueError("order must contain a nonzero integer lot count")
    if (
        isinstance(symbol_index, bool)
        or not isinstance(symbol_index, int)
        or not (0 <= symbol_index < len(book.quantities))
    ):
        raise ValueError("symbol index is outside the book")
    for name, value in (
        ("fee_bps", fee_bps),
        ("adverse_bps", adverse_bps),
        ("depth_fraction", depth_fraction),
    ):
        _finite(value, name)
    if not 0 <= fee_bps < 10_000 or not 0 <= adverse_bps < 10_000:
        raise ValueError("cost basis points must be within [0, 10000)")
    if not 0 < depth_fraction <= 1:
        raise ValueError("depth fraction must be within (0, 1]")
    _validate_depth(bids, asks)
    if book.insolvent or book.termination_reason is not None:
        raise ValueError("cannot execute on a terminated account")
    side = 1 if lot_count > 0 else -1
    levels = asks if side == 1 else bids
    multiplier = (
        float(book.contract_multipliers[symbol_index])
        if book.contract_multipliers is not None
        else 1.0
    )
    exact_multiplier = exact_quantity(multiplier)
    step = exact_quantity(rules.lot_size)
    requested = abs(lot_count) * step
    adverse = Fraction(1) + side * exact_quantity(adverse_bps) / 10_000
    notional = requested * exact_quantity(levels[0][0]) * adverse * exact_multiplier

    def rejected(reason: str) -> DepthFill:
        return DepthFill(0, lot_count, None, 0.0, reason)

    if requested < exact_quantity(rules.minimum_quantity):
        return rejected("quantity_below_minimum")
    if requested > exact_quantity(rules.maximum_quantity):
        return rejected("quantity_above_maximum")
    if notional < exact_quantity(rules.minimum_notional):
        return rejected("notional_below_minimum")
    if rules.maximum_notional is not None and notional > exact_quantity(
        rules.maximum_notional
    ):
        return rejected("notional_above_maximum")
    participation = exact_quantity(depth_fraction)
    capacity = sum((exact_quantity(q) * participation for _, q in levels), Fraction(0))
    filled_count = min(abs(lot_count), int(capacity // step))
    if filled_count == 0:
        return rejected("insufficient_depth")
    remaining = filled_count * step
    filled = remaining
    weighted_value = Fraction(0)
    for price, quantity in levels:
        consumed = min(remaining, exact_quantity(quantity) * participation)
        weighted_value += consumed * exact_quantity(price) * adverse
        remaining -= consumed
        if not remaining:
            break
    assert not remaining
    if (
        rules.maximum_notional is not None
        and weighted_value * exact_multiplier > exact_quantity(rules.maximum_notional)
    ):
        return rejected("notional_above_maximum")
    price = float(weighted_value / filled)
    signed_lots = side * filled_count
    quantity = project_quantity(signed_lots * step)
    executed_notional = abs(quantity) * price * multiplier
    fee = executed_notional * fee_bps / 10_000
    fill_prices = book.mark_prices.copy()
    fill_prices[symbol_index] = price
    book.execute_fill(
        symbol_index=symbol_index,
        quantity=quantity,
        fill_prices=fill_prices,
        valuation_prices=book.mark_prices,
        cost_amount=fee,
        turnover=executed_notional / book.portfolio_value,
        lot_size=rules.lot_size,
        lot_count=signed_lots,
    )
    return DepthFill(signed_lots, lot_count - signed_lots, price, fee, None)
