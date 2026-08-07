"""Convert causal projected price events into Nautilus L1 quotes."""

from __future__ import annotations

import math
from typing import Any

from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    ProjectedMarketEvent,
)
from trade_rl.integrations.nautilus.runtime_identity import require_nautilus_runtime

_PRICE_PHASES = frozenset(
    {
        MarketPhase.OPEN_QUOTE,
        MarketPhase.HIGH,
        MarketPhase.LOW,
        MarketPhase.CLOSE_QUOTE,
    }
)


def build_quote_tick(
    event: ProjectedMarketEvent,
    *,
    instrument: Any,
    half_spread_ticks: int,
    displayed_size: float,
) -> Any:
    """Build a deterministic top-of-book quote around one projected midpoint."""

    require_nautilus_runtime()
    if event.phase not in _PRICE_PHASES or event.price is None:
        raise ValueError(
            "Nautilus quote projection requires a price-bearing market phase"
        )
    if (
        isinstance(half_spread_ticks, bool)
        or not isinstance(half_spread_ticks, int)
        or half_spread_ticks <= 0
    ):
        raise ValueError("half_spread_ticks must be a positive integer")
    if not math.isfinite(displayed_size) or displayed_size <= 0.0:
        raise ValueError("displayed_size must be finite and positive")

    from nautilus_trader.model.data import QuoteTick

    tick = instrument.price_increment.as_double()
    midpoint = float(event.price)
    bid = instrument.make_price(midpoint - half_spread_ticks * tick)
    ask = instrument.make_price(midpoint + half_spread_ticks * tick)
    size = instrument.make_qty(displayed_size)
    if bid >= ask:
        raise ValueError("projected quote must have positive spread")

    return QuoteTick(
        instrument_id=instrument.id,
        bid_price=bid,
        ask_price=ask,
        bid_size=size,
        ask_size=size,
        ts_event=event.timestamp_ns,
        ts_init=event.timestamp_ns,
    )


__all__ = ["build_quote_tick"]
