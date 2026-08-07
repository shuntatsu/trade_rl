"""Causal mark, index, and funding data projection for Nautilus backtests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    ProjectedMarketEvent,
)
from trade_rl.integrations.nautilus.runtime_identity import require_nautilus_runtime


@dataclass(frozen=True, slots=True)
class FundingPoint:
    """One observed perpetual funding rate with an explicit settlement boundary."""

    rate: Decimal
    observed_ns: int
    next_funding_ns: int
    interval_minutes: int

    def __post_init__(self) -> None:
        if not self.rate.is_finite():
            raise ValueError("funding rate must be finite")
        if (
            isinstance(self.observed_ns, bool)
            or not isinstance(self.observed_ns, int)
            or self.observed_ns < 0
        ):
            raise ValueError("observed_ns must be a non-negative integer")
        if (
            isinstance(self.next_funding_ns, bool)
            or not isinstance(self.next_funding_ns, int)
            or self.next_funding_ns <= self.observed_ns
        ):
            raise ValueError("next_funding_ns must be after observed_ns")
        if (
            isinstance(self.interval_minutes, bool)
            or not isinstance(self.interval_minutes, int)
            or self.interval_minutes <= 0
        ):
            raise ValueError("interval_minutes must be a positive integer")


def build_mark_price_update(event: ProjectedMarketEvent, *, instrument: Any) -> Any:
    """Convert only a causal MARK phase into a Nautilus mark-price update."""

    _validate_price_phase(event, expected=MarketPhase.MARK)
    require_nautilus_runtime()
    from nautilus_trader.model.data import MarkPriceUpdate
    from nautilus_trader.model.objects import Price

    return MarkPriceUpdate(
        instrument_id=instrument.id,
        value=Price.from_str(_price_text(event.price)),
        ts_event=event.timestamp_ns,
        ts_init=event.timestamp_ns,
    )


def build_index_price_update(event: ProjectedMarketEvent, *, instrument: Any) -> Any:
    """Convert only a causal INDEX phase into a Nautilus index-price update."""

    _validate_price_phase(event, expected=MarketPhase.INDEX)
    require_nautilus_runtime()
    from nautilus_trader.model.data import IndexPriceUpdate
    from nautilus_trader.model.objects import Price

    return IndexPriceUpdate(
        instrument_id=instrument.id,
        value=Price.from_str(_price_text(event.price)),
        ts_event=event.timestamp_ns,
        ts_init=event.timestamp_ns,
    )


def build_funding_rate_update(point: FundingPoint, *, instrument: Any) -> Any:
    """Build a funding update that carries an explicit future settlement boundary."""

    require_nautilus_runtime()
    from nautilus_trader.model.data import FundingRateUpdate

    return FundingRateUpdate(
        instrument_id=instrument.id,
        rate=point.rate,
        ts_event=point.observed_ns,
        ts_init=point.observed_ns,
        interval=point.interval_minutes,
        next_funding_ns=point.next_funding_ns,
    )


def _price_text(value: float | None) -> str:
    if value is None:
        raise ValueError("price is required")
    return format(value, ".16g")


def _validate_price_phase(
    event: ProjectedMarketEvent,
    *,
    expected: MarketPhase,
) -> None:
    if event.phase is not expected:
        raise ValueError(f"expected {expected.name} market phase")
    if event.price is None or not math.isfinite(event.price) or event.price <= 0.0:
        raise ValueError(f"{expected.name} market phase requires positive finite price")


__all__ = [
    "FundingPoint",
    "build_funding_rate_update",
    "build_index_price_update",
    "build_mark_price_update",
]
