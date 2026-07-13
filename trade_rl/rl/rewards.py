"""Reward functions for baseline-relative residual learning."""

from __future__ import annotations

import math


def relative_interval_reward(
    *,
    hybrid_log_return: float,
    shadow_log_return: float,
    scale: float,
    terminated: bool,
) -> float:
    """Reward one action with its full decision-interval excess log return."""

    for field, value in (
        ("hybrid_log_return", hybrid_log_return),
        ("shadow_log_return", shadow_log_return),
        ("scale", scale),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if terminated:
        return -abs(scale)
    reward = scale * (hybrid_log_return - shadow_log_return)
    if not math.isfinite(reward):
        raise ValueError("relative reward is non-finite")
    return reward
