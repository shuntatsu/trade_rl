"""Deterministic OHLC path selection and order trigger evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from trade_rl.simulation.orders import OrderStatus, OrderType, PendingOrder


class BarPathError(ValueError):
    """Raised when OHLC data or trigger state is invalid."""


class PathMode(StrEnum):
    OPTIMISTIC = "optimistic"
    NEUTRAL = "neutral"
    CONSERVATIVE = "conservative"


class TriggerSegment(StrEnum):
    OPEN = "open"
    FIRST_EXTREME = "first_extreme"
    SECOND_EXTREME = "second_extreme"
    CLOSE = "close"


_DEFAULT_VOLUME_FRACTIONS: dict[TriggerSegment, float] = {
    TriggerSegment.OPEN: 1.0,
    TriggerSegment.FIRST_EXTREME: 0.5,
    TriggerSegment.SECOND_EXTREME: 0.25,
    TriggerSegment.CLOSE: 0.0,
}


def volume_fraction_for_segment(segment: TriggerSegment) -> float:
    """Return the conservative usable-volume fraction for a trigger segment."""

    return _DEFAULT_VOLUME_FRACTIONS[segment]


@dataclass(frozen=True, slots=True)
class BarPath:
    mode: PathMode
    points: tuple[float, float, float, float]
    mixed_direction_fallback: bool = False

    def __post_init__(self) -> None:
        if len(self.points) != 4:
            raise BarPathError("bar path must contain open, two extremes, and close")
        if any(not math.isfinite(point) or point <= 0.0 for point in self.points):
            raise BarPathError("bar path points must be finite and positive")


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    executable: bool
    triggered: bool
    execution_price: float | None
    segment: TriggerSegment | None
    available_volume_fraction: float
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.executable and self.execution_price is None:
            raise BarPathError("executable trigger decisions require a price")
        if self.execution_price is not None and (
            not math.isfinite(self.execution_price) or self.execution_price <= 0.0
        ):
            raise BarPathError("execution price must be finite and positive")
        if not 0.0 <= self.available_volume_fraction <= 1.0:
            raise BarPathError("available volume fraction must be within [0, 1]")


def _neutral_points(
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> tuple[float, float, float, float]:
    if open_price - low <= high - open_price:
        return (open_price, low, high, close)
    return (open_price, high, low, close)


def select_bar_path(
    *,
    open_price: float,
    high: float,
    low: float,
    close: float,
    mode: PathMode,
    active_directions: frozenset[int],
) -> BarPath:
    """Select one deterministic path for every order on a symbol and bar."""

    values = (open_price, high, low, close)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise BarPathError("OHLC values must be finite and positive")
    if low > min(open_price, close) or high < max(open_price, close) or low > high:
        raise BarPathError("OHLC values are internally inconsistent")
    if not active_directions.issubset({-1, 1}):
        raise BarPathError("active directions must contain only -1 and 1")

    mixed = active_directions == frozenset({-1, 1})
    if mode is PathMode.NEUTRAL or mixed or not active_directions:
        return BarPath(
            mode=PathMode.NEUTRAL,
            points=_neutral_points(open_price, high, low, close),
            mixed_direction_fallback=mixed,
        )

    direction = next(iter(active_directions))
    if mode is PathMode.CONSERVATIVE:
        first, second = (high, low) if direction > 0 else (low, high)
    elif mode is PathMode.OPTIMISTIC:
        first, second = (low, high) if direction > 0 else (high, low)
    else:  # pragma: no cover - exhaustive StrEnum branch
        raise BarPathError(f"unsupported path mode: {mode}")
    return BarPath(mode=mode, points=(open_price, first, second, close))


def _segment_for_index(index: int) -> TriggerSegment:
    return (
        TriggerSegment.FIRST_EXTREME,
        TriggerSegment.SECOND_EXTREME,
        TriggerSegment.CLOSE,
    )[index - 1]


def _first_directional_touch(
    points: tuple[float, float, float, float],
    *,
    level: float,
    upward: bool,
) -> tuple[int, TriggerSegment] | None:
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        touched = start < level <= end if upward else start > level >= end
        if touched:
            return index, _segment_for_index(index)
    return None


def _decision_at_open(*, price: float, triggered: bool) -> TriggerDecision:
    return TriggerDecision(
        executable=True,
        triggered=triggered,
        execution_price=price,
        segment=TriggerSegment.OPEN,
        available_volume_fraction=volume_fraction_for_segment(TriggerSegment.OPEN),
    )


def _no_touch(*, triggered: bool = False) -> TriggerDecision:
    return TriggerDecision(
        executable=False,
        triggered=triggered,
        execution_price=None,
        segment=None,
        available_volume_fraction=0.0,
        reason="not_touched",
    )


def evaluate_trigger(order: PendingOrder, path: BarPath) -> TriggerDecision:
    """Evaluate market, limit, or stop-market eligibility against one bar path."""

    if order.terminal:
        raise BarPathError("terminal orders cannot be evaluated")
    if order.status not in {
        OrderStatus.ELIGIBLE,
        OrderStatus.TRIGGERED,
        OrderStatus.PARTIALLY_FILLED,
    }:
        raise BarPathError("order must be eligible before trigger evaluation")

    intent = order.intent
    open_price = path.points[0]
    buy = order.remaining_quantity > 0.0

    if intent.order_type is OrderType.MARKET:
        return _decision_at_open(price=open_price, triggered=False)

    if intent.order_type is OrderType.LIMIT:
        assert intent.limit_price is not None
        limit = intent.limit_price
        if (buy and open_price <= limit) or (not buy and open_price >= limit):
            return _decision_at_open(price=open_price, triggered=False)
        touch = _first_directional_touch(path.points, level=limit, upward=not buy)
        if touch is None:
            return _no_touch()
        _, segment = touch
        return TriggerDecision(
            executable=True,
            triggered=False,
            execution_price=limit,
            segment=segment,
            available_volume_fraction=volume_fraction_for_segment(segment),
        )

    if intent.order_type is not OrderType.STOP_MARKET:  # pragma: no cover
        raise BarPathError(f"unsupported order type: {intent.order_type}")
    assert intent.stop_price is not None
    stop = intent.stop_price

    if (
        order.status in {OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED}
        and order.trigger_index is not None
    ):
        return _decision_at_open(price=open_price, triggered=True)
    if (buy and open_price >= stop) or (not buy and open_price <= stop):
        return _decision_at_open(price=open_price, triggered=True)

    touch = _first_directional_touch(path.points, level=stop, upward=buy)
    if touch is None:
        return _no_touch()
    touch_index, segment = touch
    execution_price = stop
    if path.mode is PathMode.CONSERVATIVE:
        reachable = path.points[touch_index:]
        execution_price = max(stop, *reachable) if buy else min(stop, *reachable)
    return TriggerDecision(
        executable=True,
        triggered=True,
        execution_price=execution_price,
        segment=segment,
        available_volume_fraction=volume_fraction_for_segment(segment),
    )
