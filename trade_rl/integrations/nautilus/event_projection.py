"""Policy-independent causal market-event projection for Nautilus backtests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

TARGET_ACTIVATION_DELAY_NS = 1_000
_MINIMUM_BAR_SPAN_NS = TARGET_ACTIVATION_DELAY_NS * 4


class MarketPhase(str, Enum):
    OPEN_QUOTE = "open_quote"
    TARGET_ACTIVATION = "target_activation"
    HIGH = "high"
    LOW = "low"
    INDEX = "index"
    MARK = "mark"
    CLOSE_QUOTE = "close_quote"
    POLICY_DECISION = "policy_decision"


@dataclass(frozen=True, slots=True)
class SourceBar:
    """Closed source bar with explicit physical timestamps and valuation prices."""

    open_ns: int
    close_ns: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    mark_price: float
    index_price: float


@dataclass(frozen=True, slots=True)
class ProjectedMarketEvent:
    """One deterministic event emitted into the execution timeline."""

    phase: MarketPhase
    timestamp_ns: int
    price: float | None


def project_bar_events(
    bar: SourceBar,
    *,
    activate_queued_target: bool,
) -> tuple[ProjectedMarketEvent, ...]:
    """Project one bar without consulting policy, position, or order direction.

    The open quote is always first. A target decided after the previous close may
    activate only after that quote updates the execution book. High/low ordering is
    derived solely from the source OHLC shape, never from an order side.
    """

    _validate_bar(bar)
    span = bar.close_ns - bar.open_ns
    first_extreme_ns = bar.open_ns + span // 3
    second_extreme_ns = bar.open_ns + (2 * span) // 3

    low_distance = abs(bar.open_price - bar.low_price)
    high_distance = abs(bar.high_price - bar.open_price)
    if low_distance <= high_distance:
        extremes = (
            ProjectedMarketEvent(MarketPhase.LOW, first_extreme_ns, bar.low_price),
            ProjectedMarketEvent(MarketPhase.HIGH, second_extreme_ns, bar.high_price),
        )
    else:
        extremes = (
            ProjectedMarketEvent(MarketPhase.HIGH, first_extreme_ns, bar.high_price),
            ProjectedMarketEvent(MarketPhase.LOW, second_extreme_ns, bar.low_price),
        )

    events: list[ProjectedMarketEvent] = [
        ProjectedMarketEvent(MarketPhase.OPEN_QUOTE, bar.open_ns, bar.open_price),
    ]
    if activate_queued_target:
        events.append(
            ProjectedMarketEvent(
                MarketPhase.TARGET_ACTIVATION,
                bar.open_ns + TARGET_ACTIVATION_DELAY_NS,
                None,
            )
        )
    events.extend(extremes)
    events.extend(
        (
            ProjectedMarketEvent(MarketPhase.INDEX, bar.close_ns - 2, bar.index_price),
            ProjectedMarketEvent(MarketPhase.MARK, bar.close_ns - 1, bar.mark_price),
            ProjectedMarketEvent(
                MarketPhase.CLOSE_QUOTE,
                bar.close_ns,
                bar.close_price,
            ),
            ProjectedMarketEvent(
                MarketPhase.POLICY_DECISION,
                bar.close_ns + 1,
                None,
            ),
        )
    )
    _assert_strict_event_order(events)
    return tuple(events)


def _validate_bar(bar: SourceBar) -> None:
    if (
        isinstance(bar.open_ns, bool)
        or isinstance(bar.close_ns, bool)
        or not isinstance(bar.open_ns, int)
        or not isinstance(bar.close_ns, int)
        or bar.open_ns < 0
        or bar.close_ns - bar.open_ns < _MINIMUM_BAR_SPAN_NS
    ):
        raise ValueError("source bar timestamps must be ordered with sufficient span")

    prices = (
        bar.open_price,
        bar.high_price,
        bar.low_price,
        bar.close_price,
        bar.mark_price,
        bar.index_price,
    )
    if any(not math.isfinite(price) or price <= 0.0 for price in prices):
        raise ValueError("source bar prices must be finite and positive")
    if bar.high_price < max(bar.open_price, bar.close_price) or bar.low_price > min(
        bar.open_price,
        bar.close_price,
    ):
        raise ValueError("source bar OHLC bounds are inconsistent")
    if bar.low_price > bar.high_price:
        raise ValueError("source bar OHLC bounds are inconsistent")


def _assert_strict_event_order(events: list[ProjectedMarketEvent]) -> None:
    timestamps = [event.timestamp_ns for event in events]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise RuntimeError("projected market events must have strictly increasing timestamps")


__all__ = [
    "MarketPhase",
    "ProjectedMarketEvent",
    "SourceBar",
    "TARGET_ACTIVATION_DELAY_NS",
    "project_bar_events",
]
