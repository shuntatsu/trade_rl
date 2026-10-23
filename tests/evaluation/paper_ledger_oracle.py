"""Small Decimal reference ledger for literal prospective-paper source events.

This module intentionally imports no production code. It checks the financial
ledger for quantities supplied by a test fixture; it is not a strategy oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum
from fractions import Fraction

ZERO = Decimal("0")
BASIS_POINTS = Decimal("10000")


class SourceSide(StrEnum):
    SPOT = "spot"
    PERPETUAL = "perpetual"


@dataclass(frozen=True)
class RawDepth:
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]


@dataclass(frozen=True)
class InstrumentSource:
    symbol: str
    side: SourceSide
    depth: RawDepth
    mark: Decimal | None
    lot_size: Decimal
    minimum_quantity: Decimal
    maximum_quantity: Decimal
    minimum_notional: Decimal
    fee_bps: Decimal
    maximum_notional: Decimal | None = None


@dataclass(frozen=True)
class FundingSettlement:
    symbol: str
    instrument_index: int
    at: datetime
    rate: Decimal
    mark: Decimal


@dataclass(frozen=True)
class SourceObservation:
    at: datetime
    instruments: tuple[InstrumentSource, ...]
    settlements: tuple[FundingSettlement, ...]
    adverse_bps: Decimal
    depth_fraction: Decimal


class PaperLedger:
    """Reference signed-cash ledger driven only by source literals and targets."""

    def __init__(self, *, initial_capital: Decimal) -> None:
        self.cash = initial_capital
        self.quantities: list[Decimal] = []
        self.marks: list[Decimal] = []
        self.last_instruments: tuple[InstrumentSource, ...] = ()
        self.total_cost = ZERO
        self.funding_pnl = ZERO
        self.turnover_total = ZERO
        self.fill_count = 0
        self.peak_value = initial_capital
        self.maximum_drawdown = ZERO
        self._settlements: dict[tuple[str, datetime], tuple[Decimal, Decimal]] = {}
        self._holdings: list[tuple[datetime, tuple[Decimal, ...]]] = []

    @property
    def equity(self) -> Decimal:
        return self.cash + sum(
            (
                quantity * mark
                for quantity, mark in zip(self.quantities, self.marks, strict=True)
            ),
            ZERO,
        )

    def lot_delta(self, targets: tuple[Decimal, ...]) -> tuple[int, ...]:
        if len(targets) != len(self.quantities) or len(self.last_instruments) != len(
            targets
        ):
            raise ValueError("target quantity vector must match observed instruments")
        result = []
        for target, quantity, source in zip(
            targets, self.quantities, self.last_instruments, strict=True
        ):
            lots = (target - quantity) / source.lot_size
            if lots != lots.to_integral_value():
                raise ValueError("target delta must be an exact number of source lots")
            result.append(int(lots))
        return tuple(result)

    def observe(self, source: SourceObservation) -> list[dict[str, object]]:
        if len(source.instruments) != 4:
            raise ValueError("the fixture must describe two spot/perpetual pairs")
        self.last_instruments = source.instruments
        if not self.quantities:
            self.quantities = [ZERO] * len(source.instruments)
        if len(self.quantities) != len(source.instruments):
            raise ValueError("source instrument roster changed")
        self.marks = [self._mark(instrument) for instrument in source.instruments]
        self._refresh_drawdown()

        payments: list[tuple[FundingSettlement, Decimal, Decimal]] = []
        for event in sorted(
            source.settlements, key=lambda item: (item.at, item.symbol)
        ):
            if event.at > source.at:
                raise ValueError("fixture cannot publish a future settlement")
            key = (event.symbol, event.at)
            previous = self._settlements.get(key)
            values = (event.rate, event.mark)
            if previous is not None:
                if previous != values:
                    raise ValueError("fixture revised an already settled event")
                continue
            if not 0 <= event.instrument_index < len(self.quantities):
                raise ValueError("funding instrument is outside the source roster")
            quantity = self._held_before(event.instrument_index, event.at)
            amount = -quantity * event.rate * event.mark
            self._settlements[key] = values
            payments.append((event, quantity, amount))

        by_time: dict[datetime, Decimal] = {}
        for event, _, amount in payments:
            by_time[event.at] = by_time.get(event.at, ZERO) + amount
        for at in sorted(by_time):
            amount = by_time[at]
            if amount:
                self.cash += amount
                self.funding_pnl += amount
                self._refresh_drawdown()

        return [
            {
                "symbol": event.symbol,
                "funding_at": event.at.isoformat(),
                "known_at": source.at.isoformat(),
                "quantity": str(Fraction(quantity)),
                "rate": str(Fraction(event.rate)),
                "mark": str(Fraction(event.mark)),
                "amount": float(amount),
            }
            for event, quantity, amount in payments
        ]

    def execute(
        self, source: SourceObservation, requested_lots: tuple[int, ...]
    ) -> list[dict[str, object]]:
        if len(requested_lots) != len(source.instruments):
            raise ValueError("requested lots must match the source roster")
        fills = []
        for index, (instrument, lots) in enumerate(
            zip(source.instruments, requested_lots, strict=True)
        ):
            if lots:
                fills.append(self._execute_one(source, index, instrument, lots))
        self._holdings.append((source.at, tuple(self.quantities)))
        return fills

    def account_state(self) -> dict[str, object]:
        collateral = self.cash + sum(
            (self.quantities[index] * self.marks[index] for index in (1, 3)),
            ZERO,
        )
        return {
            "cash": self.cash,
            "equity": self.equity,
            "quantities": list(self.quantities),
            "exact_quantities": [str(Fraction(value)) for value in self.quantities],
            "mark_prices": list(self.marks),
            "peak_value": self.peak_value,
            "maximum_drawdown": self.maximum_drawdown,
            "total_cost": self.total_cost,
            "funding_pnl": self.funding_pnl,
            "fill_count": self.fill_count,
            "turnover_total": self.turnover_total,
            "collateral": collateral,
        }

    def _held_before(self, index: int, settlement_at: datetime) -> Decimal:
        for filled_at, quantities in reversed(self._holdings):
            if filled_at < settlement_at:
                return quantities[index]
        return ZERO

    @staticmethod
    def _mark(instrument: InstrumentSource) -> Decimal:
        if instrument.side is SourceSide.SPOT:
            return (instrument.depth.bids[0][0] + instrument.depth.asks[0][0]) / 2
        if instrument.mark is None:
            raise ValueError("perpetual source must supply its mark price")
        return instrument.mark

    def _execute_one(
        self,
        source: SourceObservation,
        index: int,
        instrument: InstrumentSource,
        requested_lots: int,
    ) -> dict[str, object]:
        sign = 1 if requested_lots > 0 else -1
        levels = instrument.depth.asks if sign > 0 else instrument.depth.bids
        requested_quantity = Decimal(abs(requested_lots)) * instrument.lot_size
        adverse = Decimal(1) + Decimal(sign) * source.adverse_bps / BASIS_POINTS
        requested_notional = requested_quantity * levels[0][0] * adverse

        reason = None
        if requested_quantity < instrument.minimum_quantity:
            reason = "quantity_below_minimum"
        elif requested_quantity > instrument.maximum_quantity:
            reason = "quantity_above_maximum"
        elif requested_notional < instrument.minimum_notional:
            reason = "notional_below_minimum"
        elif (
            instrument.maximum_notional is not None
            and requested_notional > instrument.maximum_notional
        ):
            reason = "notional_above_maximum"
        if reason is not None:
            return self._empty_fill(index, requested_lots, reason)

        capacity = sum(
            (quantity * source.depth_fraction for _, quantity in levels), ZERO
        )
        capacity_lots = int(
            (capacity / instrument.lot_size).to_integral_value(rounding=ROUND_FLOOR)
        )
        filled_count = min(abs(requested_lots), capacity_lots)
        if not filled_count:
            return self._empty_fill(index, requested_lots, "insufficient_depth")

        filled_quantity = Decimal(filled_count) * instrument.lot_size
        remaining = filled_quantity
        weighted_notional = ZERO
        for price, displayed_quantity in levels:
            consumed = min(remaining, displayed_quantity * source.depth_fraction)
            weighted_notional += consumed * price * adverse
            remaining -= consumed
            if not remaining:
                break
        if remaining:
            raise AssertionError("reference capacity did not cover its fill")
        if (
            instrument.maximum_notional is not None
            and weighted_notional > instrument.maximum_notional
        ):
            return self._empty_fill(index, requested_lots, "notional_above_maximum")

        price = weighted_notional / filled_quantity
        signed_lots = sign * filled_count
        signed_quantity = Decimal(signed_lots) * instrument.lot_size
        notional = abs(signed_quantity) * price
        fee = notional * instrument.fee_bps / BASIS_POINTS
        equity_before = self.equity
        self.cash -= signed_quantity * price + fee
        self.quantities[index] += signed_quantity
        self.total_cost += fee
        self.turnover_total += notional / equity_before
        self.fill_count += 1
        self._refresh_drawdown()
        return {
            "symbol_index": index,
            "lot_count": requested_lots,
            "filled_lots": signed_lots,
            "unfilled_lots": requested_lots - signed_lots,
            "price": price,
            "fee": fee,
            "reason": None,
        }

    @staticmethod
    def _empty_fill(index: int, requested_lots: int, reason: str) -> dict[str, object]:
        return {
            "symbol_index": index,
            "lot_count": requested_lots,
            "filled_lots": 0,
            "unfilled_lots": requested_lots,
            "price": None,
            "fee": ZERO,
            "reason": reason,
        }

    def _refresh_drawdown(self) -> None:
        value = max(self.equity, ZERO)
        self.peak_value = max(self.peak_value, value)
        drawdown = Decimal(1) - value / max(self.peak_value, Decimal("1e-12"))
        self.maximum_drawdown = max(self.maximum_drawdown, drawdown)
