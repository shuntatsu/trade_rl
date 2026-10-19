"""Deterministic fill charges derived from primitive rates and capacity."""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal, localcontext

from tools.market_account_oracle.numbers import Number, arithmetic_context, number


@dataclass(frozen=True, slots=True)
class FillEconomics:
    fee_rate: Number = 0
    row_fee_rate: Number = 0
    maker_fee_rate: Number = 0
    row_maker_fee_rate: Number = 0
    taker_fee_rate: Number = 0
    row_taker_fee_rate: Number = 0
    spread_rate: Number = 0
    row_spread_rate: Number = 0
    impact_rate: Number = 0
    cost_multiplier: Number = 1
    maker: bool = False
    slippage_std: Number = 0
    tail_slippage_probability: Number = 0

    def __post_init__(self) -> None:
        if type(self.maker) is not bool:
            raise ValueError("maker must be boolean")
        for field in fields(self):
            if field.name != "maker" and number(getattr(self, field.name)) < 0:
                raise ValueError("execution rates and multiplier must be non-negative")
        if number(self.slippage_std) or number(self.tail_slippage_probability):
            raise ValueError("stochastic slippage is unsupported without bound draws")


@dataclass(frozen=True, slots=True)
class FillCalculation:
    notional: Decimal
    participation: Decimal
    cost: Decimal


def calculate_fill(
    *,
    reported_quantity: Decimal,
    price: Decimal,
    contract_multiplier: Decimal,
    economics: FillEconomics,
    market_notional: Number,
    quantity_capacity: Number | None,
) -> FillCalculation:
    if type(economics) is not FillEconomics:
        raise ValueError("requires an exact deterministic fill declaration")
    economics.__post_init__()
    quote_capacity = number(market_notional)
    if quote_capacity <= 0:
        raise ValueError("market notional must be positive for an accepted fill")
    with localcontext(arithmetic_context()):
        notional = abs(reported_quantity) * price * contract_multiplier
        participation = notional / quote_capacity
        if quantity_capacity is not None:
            units = number(quantity_capacity)
            if units <= 0:
                raise ValueError("quantity capacity must be positive")
            participation = max(participation, abs(reported_quantity) / units)
        if not 0 < participation <= 1:
            raise ValueError("accepted fill exceeds its raw market capacity")
        venue = "maker" if economics.maker else "taker"
        fee = sum(
            number(getattr(economics, name))
            for name in (
                "fee_rate",
                "row_fee_rate",
                f"{venue}_fee_rate",
                f"row_{venue}_fee_rate",
            )
        )
        spread = number(economics.spread_rate) + number(economics.row_spread_rate)
        spread *= Decimal("0.5") if economics.maker else 1
        impact = number(economics.impact_rate) * participation.sqrt()
        cost = notional * number(economics.cost_multiplier) * (fee + spread + impact)
        return FillCalculation(notional, participation, cost)
