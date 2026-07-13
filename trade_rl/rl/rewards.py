"""Hierarchical reward functions for absolute-growth residual learning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AbsoluteGrowthRewardConfig:
    """Typed configuration for absolute growth and light risk shaping."""

    scale: float = 100.0
    baseline_window_hours: float = 720.0
    baseline_minimum_history_hours: float = 168.0
    baseline_tolerance: float = 0.015
    baseline_penalty_weight: float = 0.10
    drawdown_penalty_weight: float = 0.05
    drawdown_free: float = 0.05
    drawdown_middle: float = 0.10
    drawdown_high: float = 0.15
    drawdown_stop: float = 0.20
    drawdown_slopes: tuple[float, float, float] = (1.0, 3.0, 8.0)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("scale", self.scale),
            ("baseline_window_hours", self.baseline_window_hours),
            ("baseline_minimum_history_hours", self.baseline_minimum_history_hours),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.baseline_minimum_history_hours > self.baseline_window_hours:
            raise ValueError(
                "baseline_minimum_history_hours cannot exceed baseline_window_hours"
            )
        for field_name, value in (
            ("baseline_tolerance", self.baseline_tolerance),
            ("baseline_penalty_weight", self.baseline_penalty_weight),
            ("drawdown_penalty_weight", self.drawdown_penalty_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not (
            0.0
            <= self.drawdown_free
            < self.drawdown_middle
            < self.drawdown_high
            < self.drawdown_stop
            <= 1.0
        ):
            raise ValueError(
                "drawdown thresholds must be strictly increasing in [0, 1]"
            )
        if len(self.drawdown_slopes) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.drawdown_slopes
        ):
            raise ValueError(
                "drawdown_slopes must contain three positive finite values"
            )
        if not (
            self.drawdown_slopes[0]
            <= self.drawdown_slopes[1]
            <= self.drawdown_slopes[2]
        ):
            raise ValueError("drawdown_slopes must be non-decreasing")


@dataclass(frozen=True, slots=True)
class RewardContext:
    """Observable reward state at one transition boundary."""

    rolling_hybrid_log_growth: float
    rolling_shadow_log_growth: float
    baseline_shortfall: float
    baseline_tolerance: float
    baseline_penalty: float
    hybrid_drawdown: float
    drawdown_severity: float
    history_bars: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("rolling_hybrid_log_growth", self.rolling_hybrid_log_growth),
            ("rolling_shadow_log_growth", self.rolling_shadow_log_growth),
            ("baseline_shortfall", self.baseline_shortfall),
            ("baseline_tolerance", self.baseline_tolerance),
            ("baseline_penalty", self.baseline_penalty),
            ("hybrid_drawdown", self.hybrid_drawdown),
            ("drawdown_severity", self.drawdown_severity),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.history_bars < 0:
            raise ValueError("history_bars must be non-negative")
        if not 0.0 <= self.hybrid_drawdown <= 1.0:
            raise ValueError("hybrid_drawdown must be within [0, 1]")
        if self.baseline_shortfall < 0.0:
            raise ValueError("baseline_shortfall must be non-negative")
        if self.baseline_tolerance < 0.0:
            raise ValueError("baseline_tolerance must be non-negative")
        if self.baseline_penalty < 0.0:
            raise ValueError("baseline_penalty must be non-negative")
        if self.drawdown_severity < 0.0:
            raise ValueError("drawdown_severity must be non-negative")

    @property
    def rolling_growth_gap(self) -> float:
        """Policy rolling growth minus the independent shadow baseline."""

        return self.rolling_hybrid_log_growth - self.rolling_shadow_log_growth


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """All unscaled and scaled terms used for one policy reward."""

    growth_raw: float
    baseline_penalty_delta: float
    baseline_penalty_weighted: float
    drawdown_penalty_delta: float
    drawdown_penalty_weighted: float
    total_raw: float
    total_scaled: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("growth_raw", self.growth_raw),
            ("baseline_penalty_delta", self.baseline_penalty_delta),
            ("baseline_penalty_weighted", self.baseline_penalty_weighted),
            ("drawdown_penalty_delta", self.drawdown_penalty_delta),
            ("drawdown_penalty_weighted", self.drawdown_penalty_weighted),
            ("total_raw", self.total_raw),
            ("total_scaled", self.total_scaled),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.baseline_penalty_delta < 0.0:
            raise ValueError("baseline_penalty_delta must be non-negative")
        if self.baseline_penalty_weighted < 0.0:
            raise ValueError("baseline_penalty_weighted must be non-negative")
        if self.drawdown_penalty_delta < 0.0:
            raise ValueError("drawdown_penalty_delta must be non-negative")
        if self.drawdown_penalty_weighted < 0.0:
            raise ValueError("drawdown_penalty_weighted must be non-negative")


def drawdown_severity(
    drawdown: float,
    config: AbsoluteGrowthRewardConfig,
) -> float:
    """Map drawdown to a continuous staged severity with increasing slopes."""

    if not math.isfinite(drawdown) or not 0.0 <= drawdown <= 1.0:
        raise ValueError("drawdown must be finite and within [0, 1]")
    free = config.drawdown_free
    middle = config.drawdown_middle
    high = config.drawdown_high
    first_slope, second_slope, third_slope = config.drawdown_slopes
    if drawdown <= free:
        return 0.0
    first = first_slope * (min(drawdown, middle) - free)
    if drawdown <= middle:
        return first
    second = second_slope * (min(drawdown, high) - middle)
    if drawdown <= high:
        return first + second
    third = third_slope * (drawdown - high)
    return first + second + third


def _rolling_log_growth(values: Sequence[float], history_bars: int) -> float:
    selected = values[-history_bars:] if history_bars else ()
    log_returns: list[float] = []
    for value in selected:
        resolved = float(value)
        if not math.isfinite(resolved) or resolved <= -1.0:
            raise ValueError("return history must be finite and greater than -1")
        log_returns.append(math.log1p(resolved))
    return math.fsum(log_returns)


def build_reward_context(
    *,
    hybrid_returns: Sequence[float],
    shadow_returns: Sequence[float],
    hybrid_drawdown: float,
    window_bars: int,
    minimum_history_bars: int,
    config: AbsoluteGrowthRewardConfig,
) -> RewardContext:
    """Build rolling baseline and drawdown state from causal book histories."""

    if (
        isinstance(window_bars, bool)
        or not isinstance(window_bars, int)
        or window_bars <= 0
    ):
        raise ValueError("window_bars must be a positive integer")
    if (
        isinstance(minimum_history_bars, bool)
        or not isinstance(minimum_history_bars, int)
        or minimum_history_bars <= 0
        or minimum_history_bars > window_bars
    ):
        raise ValueError(
            "minimum_history_bars must be a positive integer not exceeding window_bars"
        )
    if len(hybrid_returns) != len(shadow_returns):
        raise ValueError("hybrid and shadow return histories must have equal length")

    history_bars = min(len(hybrid_returns), window_bars)
    hybrid_growth = _rolling_log_growth(hybrid_returns, history_bars)
    shadow_growth = _rolling_log_growth(shadow_returns, history_bars)
    shortfall = max(0.0, shadow_growth - hybrid_growth)
    tolerance = config.baseline_tolerance * history_bars / window_bars
    penalty = (
        max(0.0, shortfall - tolerance) if history_bars >= minimum_history_bars else 0.0
    )
    return RewardContext(
        rolling_hybrid_log_growth=hybrid_growth,
        rolling_shadow_log_growth=shadow_growth,
        baseline_shortfall=shortfall,
        baseline_tolerance=tolerance,
        baseline_penalty=penalty,
        hybrid_drawdown=hybrid_drawdown,
        drawdown_severity=drawdown_severity(hybrid_drawdown, config),
        history_bars=history_bars,
    )


def absolute_growth_reward(
    *,
    hybrid_log_return: float,
    before: RewardContext,
    after: RewardContext,
    config: AbsoluteGrowthRewardConfig,
) -> RewardBreakdown:
    """Reward net absolute growth and only newly worsening shaping levels."""

    if not math.isfinite(hybrid_log_return):
        raise ValueError("hybrid_log_return must be finite")
    baseline_delta = max(0.0, after.baseline_penalty - before.baseline_penalty)
    drawdown_delta = max(0.0, after.drawdown_severity - before.drawdown_severity)
    baseline_weighted = config.baseline_penalty_weight * baseline_delta
    drawdown_weighted = config.drawdown_penalty_weight * drawdown_delta
    total_raw = hybrid_log_return - baseline_weighted - drawdown_weighted
    total_scaled = config.scale * total_raw
    if not math.isfinite(total_scaled):
        raise ValueError("absolute growth reward is non-finite")
    return RewardBreakdown(
        growth_raw=hybrid_log_return,
        baseline_penalty_delta=baseline_delta,
        baseline_penalty_weighted=baseline_weighted,
        drawdown_penalty_delta=drawdown_delta,
        drawdown_penalty_weighted=drawdown_weighted,
        total_raw=total_raw,
        total_scaled=total_scaled,
    )
