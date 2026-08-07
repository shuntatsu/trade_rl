"""Pure, validated learning-rate schedules for RL algorithms."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

LearningRateScheduleKind = Literal["constant", "linear", "cosine"]


def build_learning_rate_schedule(
    *,
    initial_rate: float,
    final_ratio: float,
    kind: str,
) -> float | Callable[[float], float]:
    """Build an SB3-compatible schedule from progress remaining in ``[0, 1]``."""

    if not math.isfinite(initial_rate) or initial_rate <= 0.0:
        raise ValueError("initial_rate must be finite and positive")
    if not math.isfinite(final_ratio) or not 0.0 < final_ratio <= 1.0:
        raise ValueError("final_ratio must be within (0, 1]")
    if kind not in {"constant", "linear", "cosine"}:
        raise ValueError("kind must be constant, linear, or cosine")
    if kind == "constant":
        return float(initial_rate)

    def schedule(progress_remaining: float) -> float:
        if (
            not math.isfinite(progress_remaining)
            or not 0.0 <= progress_remaining <= 1.0
        ):
            raise ValueError("progress_remaining must be finite and within [0, 1]")
        if kind == "linear":
            multiplier = final_ratio + (1.0 - final_ratio) * progress_remaining
        else:
            completed = 1.0 - progress_remaining
            multiplier = final_ratio + (1.0 - final_ratio) * (
                0.5 * (1.0 + math.cos(math.pi * completed))
            )
        rate = initial_rate * multiplier
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("resolved learning rate must be finite and positive")
        return float(rate)

    return schedule


__all__ = ["LearningRateScheduleKind", "build_learning_rate_schedule"]
