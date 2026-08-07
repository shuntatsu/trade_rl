"""Canonical funding settlement adapter for the pinned Python BacktestEngine gap.

NautilusTrader v1.230.0 exposes funding settlement in its Rust simulated exchange,
but the Python low-level BacktestEngine used by Trade RL does not dispatch
FundingRateUpdate/MarkPriceUpdate into that exchange. Until an adopted upstream
runtime closes that gap, Trade RL settles funding explicitly at this integration
boundary and records the result as canonical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from trade_rl.simulation.execution_parity import CanonicalExecutionRecord


@dataclass(frozen=True, slots=True)
class FundingSettlementInput:
    """Inputs visible at one explicit perpetual funding boundary."""

    instrument_id: str
    settlement_currency: str
    currency_precision: int
    signed_quantity: Decimal
    settlement_price: Decimal
    contract_multiplier: Decimal
    funding_rate: Decimal
    boundary_ns: int

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        if not self.settlement_currency:
            raise ValueError("settlement_currency must be non-empty")
        if (
            isinstance(self.currency_precision, bool)
            or not isinstance(self.currency_precision, int)
            or self.currency_precision < 0
            or self.currency_precision > 18
        ):
            raise ValueError("currency_precision must be an integer within [0, 18]")
        if not self.signed_quantity.is_finite():
            raise ValueError("signed_quantity must be finite")
        if not self.settlement_price.is_finite() or self.settlement_price <= 0:
            raise ValueError("settlement_price must be finite and positive")
        if not self.contract_multiplier.is_finite() or self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be finite and positive")
        if not self.funding_rate.is_finite():
            raise ValueError("funding_rate must be finite")
        if (
            isinstance(self.boundary_ns, bool)
            or not isinstance(self.boundary_ns, int)
            or self.boundary_ns < 0
        ):
            raise ValueError("boundary_ns must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CanonicalFundingSettlement:
    """Deterministic economic result of one funding boundary."""

    instrument_id: str
    settlement_currency: str
    boundary_ns: int
    funding_rate: Decimal
    settlement_price: Decimal
    signed_quantity: Decimal
    quantity_change: None
    amount: Decimal
    amount_minor: int


class CanonicalFundingLedger:
    """Settle each funding boundary exactly once and in timestamp order."""

    def __init__(self) -> None:
        self._settled_boundaries: list[int] = []

    @property
    def settled_boundaries(self) -> tuple[int, ...]:
        return tuple(self._settled_boundaries)

    def settle(self, value: FundingSettlementInput) -> CanonicalFundingSettlement:
        if value.boundary_ns in self._settled_boundaries:
            raise ValueError(f"funding boundary {value.boundary_ns} already settled")
        if (
            self._settled_boundaries
            and value.boundary_ns < self._settled_boundaries[-1]
        ):
            raise ValueError("funding boundaries must be strictly increasing")

        notional = (
            abs(value.signed_quantity)
            * value.settlement_price
            * value.contract_multiplier
        )
        side = Decimal("-1") if value.signed_quantity > 0 else Decimal("1")
        if value.signed_quantity == 0:
            raw_amount = Decimal("0")
        else:
            raw_amount = notional * value.funding_rate * side

        quantum = Decimal(1).scaleb(-value.currency_precision)
        if raw_amount < 0:
            # Conservative debit: round the debit magnitude away from zero.
            magnitude = (-raw_amount).quantize(quantum, rounding=ROUND_CEILING)
            amount = -magnitude
        else:
            # Conservative credit: never round a credit upward.
            amount = raw_amount.quantize(quantum, rounding=ROUND_FLOOR)

        scale = Decimal(10) ** value.currency_precision
        amount_minor = int(amount * scale)
        self._settled_boundaries.append(value.boundary_ns)
        return CanonicalFundingSettlement(
            instrument_id=value.instrument_id,
            settlement_currency=value.settlement_currency,
            boundary_ns=value.boundary_ns,
            funding_rate=value.funding_rate,
            settlement_price=value.settlement_price,
            signed_quantity=value.signed_quantity,
            quantity_change=None,
            amount=amount,
            amount_minor=amount_minor,
        )


def canonicalize_funding_settlement_record(
    settlement: CanonicalFundingSettlement,
    *,
    sequence: int,
    price_tick: Decimal,
    lot_size: Decimal,
    equity_before_minor: int,
) -> CanonicalExecutionRecord:
    """Project one settled funding boundary into the canonical execution trace."""

    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise ValueError("sequence must be a positive integer")
    if isinstance(equity_before_minor, bool) or not isinstance(equity_before_minor, int):
        raise ValueError("equity_before_minor must be an integer")

    price_ticks = _exact_grid_units(
        settlement.settlement_price,
        price_tick,
        value_name="price",
        increment_name="price_tick",
    )
    position_lots = _exact_grid_units(
        settlement.signed_quantity,
        lot_size,
        value_name="quantity",
        increment_name="lot_size",
    )
    return CanonicalExecutionRecord(
        sequence=sequence,
        event_type="funding",
        timestamp_ns=settlement.boundary_ns,
        price_ticks=price_ticks,
        quantity_lots=0,
        fee_minor=0,
        funding_minor=settlement.amount_minor,
        position_lots=position_lots,
        equity_minor=equity_before_minor + settlement.amount_minor,
        terminal_reason=None,
    )


def _exact_grid_units(
    value: Decimal,
    increment: Decimal,
    *,
    value_name: str,
    increment_name: str,
) -> int:
    if not isinstance(increment, Decimal) or not increment.is_finite() or increment <= 0:
        raise ValueError(f"{increment_name} must be a finite positive Decimal")
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{value_name} must be a finite Decimal")
    units = value / increment
    integral = units.to_integral_value()
    if units != integral:
        raise ValueError(f"{value_name} must align exactly to {increment_name}")
    return int(integral)


__all__ = [
    "CanonicalFundingLedger",
    "CanonicalFundingSettlement",
    "FundingSettlementInput",
    "canonicalize_funding_settlement_record",
]
