"""Canonicalization helpers for the maintained legacy stateful execution engine."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Iterable

from trade_rl.simulation.execution_canonicalization import CanonicalFillSignature
from trade_rl.simulation.execution_parity import CanonicalExecutionRecord
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
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


def canonicalize_legacy_funding_boundary_record(
    boundary: FundingBoundaryEvidence,
    *,
    sequence: int,
    price_tick: float,
    lot_size: float,
    currency_precision: int,
    equity_before_minor: int,
) -> CanonicalExecutionRecord:
    """Convert one maintained single-instrument funding boundary to trace identity."""

    if not isinstance(boundary, FundingBoundaryEvidence):
        raise ValueError("boundary must be FundingBoundaryEvidence")
    if len(boundary.funding_due) != 1 or not boundary.funding_due[0]:
        raise ValueError("legacy funding canonicalization requires one due instrument")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if (
        isinstance(currency_precision, bool)
        or not isinstance(currency_precision, int)
        or currency_precision < 0
        or currency_precision > 18
    ):
        raise ValueError("currency_precision must be an integer within [0, 18]")
    if isinstance(equity_before_minor, bool) or not isinstance(equity_before_minor, int):
        raise ValueError("equity_before_minor must be an integer")

    price_ticks = _to_grid_units(boundary.mark_prices[0], price_tick, "price")
    position_lots = _to_grid_units(boundary.signed_quantities[0], lot_size, "quantity")
    canonical_before = _minor_units(
        boundary.equity_before_funding,
        currency_precision=currency_precision,
    )
    if canonical_before != equity_before_minor:
        raise ValueError("equity_before_minor does not match funding boundary")

    funding_minor = _minor_units(
        boundary.funding_amount,
        currency_precision=currency_precision,
    )
    equity_minor = equity_before_minor + funding_minor
    canonical_after = _minor_units(
        boundary.equity_after_funding,
        currency_precision=currency_precision,
    )
    if equity_minor != canonical_after:
        raise ValueError("funding boundary canonical equity closure mismatch")

    return CanonicalExecutionRecord(
        sequence=sequence,
        event_type="funding",
        timestamp_ns=boundary.timestamp_ns,
        price_ticks=price_ticks,
        quantity_lots=0,
        fee_minor=0,
        funding_minor=funding_minor,
        position_lots=position_lots,
        equity_minor=equity_minor,
        terminal_reason=None,
    )


def _to_grid_units(value: float, increment: float, name: str) -> int:
    if not math.isfinite(increment) or increment <= 0.0:
        raise ValueError(f"{name} increment must be finite and positive")
    units = value / increment
    rounded = round(units)
    if not math.isclose(units, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} is not aligned to canonical increment")
    return int(rounded)


def _minor_units(value: float, *, currency_precision: int) -> int:
    quantum = Decimal(1).scaleb(-currency_precision)
    normalized = Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_EVEN)
    return int(normalized * (Decimal(10) ** currency_precision))


__all__ = [
    "canonicalize_legacy_fill_events",
    "canonicalize_legacy_funding_boundary_record",
]
