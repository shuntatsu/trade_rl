"""Reward functions for baseline-relative residual learning."""

from __future__ import annotations

import math


def relative_interval_reward(
    *,
    hybrid_log_return: float,
    shadow_log_return: float,
    scale: float,
    hybrid_terminated: bool,
    shadow_terminated: bool,
    hybrid_drawdown: float,
    shadow_drawdown: float,
    downside_penalty: float = 0.0,
    excess_drawdown_penalty: float = 0.0,
) -> float:
    """Reward one action with interval excess return and explicit risk terms."""

    for field_name, value in (
        ("hybrid_log_return", hybrid_log_return),
        ("shadow_log_return", shadow_log_return),
        ("scale", scale),
        ("hybrid_drawdown", hybrid_drawdown),
        ("shadow_drawdown", shadow_drawdown),
        ("downside_penalty", downside_penalty),
        ("excess_drawdown_penalty", excess_drawdown_penalty),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if not 0.0 <= hybrid_drawdown <= 1.0:
        raise ValueError("hybrid_drawdown must be within [0, 1]")
    if not 0.0 <= shadow_drawdown <= 1.0:
        raise ValueError("shadow_drawdown must be within [0, 1]")
    if downside_penalty < 0.0 or excess_drawdown_penalty < 0.0:
        raise ValueError("reward penalties must be non-negative")

    if hybrid_terminated and not shadow_terminated:
        return -abs(scale)
    if shadow_terminated and not hybrid_terminated:
        return abs(scale)
    if hybrid_terminated and shadow_terminated:
        return scale * (hybrid_log_return - shadow_log_return)

    downside = max(0.0, -hybrid_log_return)
    excess_drawdown = max(0.0, hybrid_drawdown - shadow_drawdown)
    reward = scale * (
        hybrid_log_return
        - shadow_log_return
        - downside_penalty * downside
        - excess_drawdown_penalty * excess_drawdown
    )
    if not math.isfinite(reward):
        raise ValueError("relative reward is non-finite")
    return reward
