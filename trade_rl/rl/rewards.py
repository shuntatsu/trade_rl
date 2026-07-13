"""Reward functions for baseline-relative residual learning."""

from __future__ import annotations

import math


def relative_interval_reward(
    *,
    hybrid_log_return: float,
    shadow_log_return: float,
    scale: float,
    hybrid_insolvent: bool,
    hybrid_insolvency_penalty: float,
) -> float:
    """Reward paired interval growth and penalize only controlled insolvency."""

    for field, value in (
        ("hybrid_log_return", hybrid_log_return),
        ("shadow_log_return", shadow_log_return),
        ("scale", scale),
        ("hybrid_insolvency_penalty", hybrid_insolvency_penalty),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if hybrid_insolvency_penalty < 0.0:
        raise ValueError("hybrid_insolvency_penalty must be non-negative")
    if not isinstance(hybrid_insolvent, bool):
        raise ValueError("hybrid_insolvent must be a boolean")

    reward = scale * (hybrid_log_return - shadow_log_return)
    if hybrid_insolvent:
        reward -= scale * hybrid_insolvency_penalty
    if not math.isfinite(reward):
        raise ValueError("relative reward is non-finite")
    return reward
