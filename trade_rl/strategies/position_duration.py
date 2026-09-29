"""Position duration state used by time-aware strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trade_rl.strategies.position_intent import PositionIntent


@dataclass(frozen=True, slots=True)
class MinimumHoldDecision:
    requested_intent: PositionIntent
    effective_intent: PositionIntent
    target_quantity_override: float | None
    suppressed: bool


def next_position_age_bars(
    age_bars: int,
    *,
    previous_quantity: float,
    filled_quantity: float,
) -> int:
    """Advance an age counter after one executed interval."""

    if isinstance(age_bars, bool) or not isinstance(age_bars, int) or age_bars < 0:
        raise ValueError("age_bars must be a non-negative integer")
    if not math.isfinite(previous_quantity) or not math.isfinite(filled_quantity):
        raise ValueError("position quantities must be finite")
    if filled_quantity == 0.0:
        return 0
    if previous_quantity == 0.0 or (previous_quantity > 0.0) != (filled_quantity > 0.0):
        return 1
    return age_bars + 1


def constrain_intent_for_minimum_hold(
    requested_intent: PositionIntent,
    *,
    current_quantity: float,
    position_age_bars: int,
    minimum_hold_bars: int,
) -> MinimumHoldDecision:
    """Suppress voluntary changes while an actual position is inside its hold."""

    if not isinstance(requested_intent, PositionIntent):
        raise TypeError("requested_intent must be a PositionIntent")
    if not math.isfinite(current_quantity):
        raise ValueError("current_quantity must be finite")
    if (
        isinstance(position_age_bars, bool)
        or not isinstance(position_age_bars, int)
        or position_age_bars < 0
    ):
        raise ValueError("position_age_bars must be a non-negative integer")
    if (
        isinstance(minimum_hold_bars, bool)
        or not isinstance(minimum_hold_bars, int)
        or minimum_hold_bars < 0
    ):
        raise ValueError("minimum_hold_bars must be a non-negative integer")

    if current_quantity != 0.0 and position_age_bars < minimum_hold_bars:
        held_intent = (
            PositionIntent.LONG if current_quantity > 0.0 else PositionIntent.SHORT
        )
        return MinimumHoldDecision(
            requested_intent=requested_intent,
            effective_intent=held_intent,
            target_quantity_override=current_quantity,
            suppressed=True,
        )
    return MinimumHoldDecision(
        requested_intent=requested_intent,
        effective_intent=requested_intent,
        target_quantity_override=None,
        suppressed=False,
    )


__all__ = [
    "MinimumHoldDecision",
    "constrain_intent_for_minimum_hold",
    "next_position_age_bars",
]
