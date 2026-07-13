"""Structured rewards for absolute and baseline-relative portfolio learning."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, fields

REWARD_SCHEMA = "baseline_residual_reward_v3"


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Hierarchical reward weights with absolute log growth as the primary term."""

    scale: float = 100.0
    absolute_growth_weight: float = 1.0
    excess_growth_weight: float = 0.25
    incremental_drawdown_weight: float = 0.10
    drawdown_dead_zone: float = 0.0025
    baseline_underperformance_weight: float = 0.15
    baseline_window_hours: float = 168.0
    baseline_window_steps: int | None = None
    baseline_tolerance: float = 0.005
    baseline_progressive_power: float = 2.0
    projection_penalty_weight: float = 0.01
    terminal_equity_weight: float = 1.0
    margin_deficit_weight: float = 1.0
    equity_floor_fraction: float = 1e-9

    def __post_init__(self) -> None:
        for field_name, value in (
            ("scale", self.scale),
            ("absolute_growth_weight", self.absolute_growth_weight),
            ("excess_growth_weight", self.excess_growth_weight),
            ("incremental_drawdown_weight", self.incremental_drawdown_weight),
            ("drawdown_dead_zone", self.drawdown_dead_zone),
            (
                "baseline_underperformance_weight",
                self.baseline_underperformance_weight,
            ),
            ("baseline_window_hours", self.baseline_window_hours),
            ("baseline_tolerance", self.baseline_tolerance),
            ("baseline_progressive_power", self.baseline_progressive_power),
            ("projection_penalty_weight", self.projection_penalty_weight),
            ("terminal_equity_weight", self.terminal_equity_weight),
            ("margin_deficit_weight", self.margin_deficit_weight),
            ("equity_floor_fraction", self.equity_floor_fraction),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        non_negative = (
            self.absolute_growth_weight,
            self.excess_growth_weight,
            self.incremental_drawdown_weight,
            self.drawdown_dead_zone,
            self.baseline_underperformance_weight,
            self.baseline_tolerance,
            self.projection_penalty_weight,
            self.terminal_equity_weight,
            self.margin_deficit_weight,
        )
        if min(non_negative) < 0.0:
            raise ValueError("reward weights and tolerances must be non-negative")
        if self.baseline_progressive_power < 1.0:
            raise ValueError("baseline_progressive_power must be at least one")
        if self.baseline_window_hours <= 0.0:
            raise ValueError("baseline_window_hours must be positive")
        if self.baseline_window_steps is not None and (
            isinstance(self.baseline_window_steps, bool)
            or not isinstance(self.baseline_window_steps, int)
            or self.baseline_window_steps <= 0
        ):
            raise ValueError("baseline_window_steps must be a positive integer")
        if not 0.0 < self.equity_floor_fraction <= 1.0:
            raise ValueError("equity_floor_fraction must be within (0, 1]")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Auditable unscaled and scaled components for one decision interval."""

    absolute_log_growth: float
    excess_log_growth: float
    incremental_drawdown: float
    rolling_baseline_underperformance: float
    projection_distance: float
    terminal_equity_shortfall: float
    margin_deficit: float
    absolute_component: float
    excess_component: float
    drawdown_penalty: float
    baseline_penalty: float
    projection_penalty: float
    terminal_penalty: float
    margin_penalty: float
    unscaled_total: float
    scaled_total: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(float(getattr(self, item.name))) for item in fields(self)
        ):
            raise ValueError("reward breakdown values must be finite")


@dataclass(slots=True)
class RewardTracker:
    """State required by incremental drawdown and rolling baseline penalties."""

    config: RewardConfig = field(default_factory=RewardConfig)
    decision_hours: float = 4.0
    previous_hybrid_drawdown: float = 0.0
    previous_shadow_drawdown: float = 0.0
    _hybrid_returns: deque[float] = field(init=False, repr=False)
    _shadow_returns: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.decision_hours) or self.decision_hours <= 0.0:
            raise ValueError("decision_hours must be finite and positive")
        window_steps = self.config.baseline_window_steps
        if window_steps is None:
            window_steps = max(
                1,
                int(round(self.config.baseline_window_hours / self.decision_hours)),
            )
        self._hybrid_returns = deque(maxlen=window_steps)
        self._shadow_returns = deque(maxlen=window_steps)
        self.reset()

    @property
    def baseline_window_steps(self) -> int:
        maxlen = self._hybrid_returns.maxlen
        if maxlen is None:  # pragma: no cover - constructed with a fixed maxlen
            raise RuntimeError("reward window is not bounded")
        return maxlen

    def reset(
        self,
        *,
        hybrid_drawdown: float = 0.0,
        shadow_drawdown: float = 0.0,
    ) -> None:
        _validate_drawdown(hybrid_drawdown, field_name="hybrid_drawdown")
        _validate_drawdown(shadow_drawdown, field_name="shadow_drawdown")
        self.previous_hybrid_drawdown = hybrid_drawdown
        self.previous_shadow_drawdown = shadow_drawdown
        self._hybrid_returns.clear()
        self._shadow_returns.clear()

    def step(
        self,
        *,
        hybrid_log_return: float,
        shadow_log_return: float,
        hybrid_drawdown: float,
        shadow_drawdown: float,
        projection_distance: float = 0.0,
        hybrid_margin_deficit_fraction: float = 0.0,
        hybrid_equity_fraction: float = 1.0,
        shadow_equity_fraction: float = 1.0,
        hybrid_terminated: bool = False,
        shadow_terminated: bool = False,
    ) -> RewardBreakdown:
        for field_name, value in (
            ("hybrid_log_return", hybrid_log_return),
            ("shadow_log_return", shadow_log_return),
            ("projection_distance", projection_distance),
            ("hybrid_margin_deficit_fraction", hybrid_margin_deficit_fraction),
            ("hybrid_equity_fraction", hybrid_equity_fraction),
            ("shadow_equity_fraction", shadow_equity_fraction),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        _validate_drawdown(hybrid_drawdown, field_name="hybrid_drawdown")
        _validate_drawdown(shadow_drawdown, field_name="shadow_drawdown")
        if projection_distance < 0.0:
            raise ValueError("projection_distance must be non-negative")
        if hybrid_margin_deficit_fraction < 0.0:
            raise ValueError("hybrid_margin_deficit_fraction must be non-negative")
        if hybrid_equity_fraction < 0.0 or shadow_equity_fraction < 0.0:
            raise ValueError("equity fractions must be non-negative")

        self._hybrid_returns.append(float(hybrid_log_return))
        self._shadow_returns.append(float(shadow_log_return))
        absolute_log_growth = float(hybrid_log_return)
        excess_log_growth = float(hybrid_log_return - shadow_log_return)

        hybrid_increment = max(
            0.0,
            hybrid_drawdown - self.previous_hybrid_drawdown,
        )
        shadow_increment = max(
            0.0,
            shadow_drawdown - self.previous_shadow_drawdown,
        )
        incremental_drawdown = max(
            0.0,
            hybrid_increment - shadow_increment - self.config.drawdown_dead_zone,
        )
        self.previous_hybrid_drawdown = hybrid_drawdown
        self.previous_shadow_drawdown = shadow_drawdown

        rolling_underperformance = max(
            0.0,
            sum(self._shadow_returns)
            - sum(self._hybrid_returns)
            - self.config.baseline_tolerance,
        )
        progressive_hinge = _progressive_hinge(
            rolling_underperformance,
            tolerance=max(self.config.baseline_tolerance, 1e-12),
            power=self.config.baseline_progressive_power,
        )

        terminal_shortfall = 0.0
        if hybrid_terminated:
            safe_fraction = max(
                hybrid_equity_fraction,
                self.config.equity_floor_fraction,
            )
            terminal_shortfall = max(0.0, -math.log(safe_fraction))
        if shadow_terminated and not hybrid_terminated:
            # A shadow failure is useful evidence, but never a discontinuous jackpot.
            terminal_shortfall -= 0.25 * max(
                0.0,
                -math.log(
                    max(
                        shadow_equity_fraction,
                        self.config.equity_floor_fraction,
                    )
                ),
            )

        absolute_component = self.config.absolute_growth_weight * absolute_log_growth
        excess_component = self.config.excess_growth_weight * excess_log_growth
        drawdown_penalty = (
            self.config.incremental_drawdown_weight * incremental_drawdown
        )
        baseline_penalty = (
            self.config.baseline_underperformance_weight * progressive_hinge
        )
        projection_penalty = self.config.projection_penalty_weight * projection_distance
        terminal_penalty = self.config.terminal_equity_weight * terminal_shortfall
        margin_deficit = float(hybrid_margin_deficit_fraction)
        margin_penalty = self.config.margin_deficit_weight * margin_deficit
        unscaled_total = (
            absolute_component
            + excess_component
            - drawdown_penalty
            - baseline_penalty
            - projection_penalty
            - terminal_penalty
            - margin_penalty
        )
        scaled_total = self.config.scale * unscaled_total
        return RewardBreakdown(
            absolute_log_growth=absolute_log_growth,
            excess_log_growth=excess_log_growth,
            incremental_drawdown=incremental_drawdown,
            rolling_baseline_underperformance=rolling_underperformance,
            projection_distance=projection_distance,
            terminal_equity_shortfall=terminal_shortfall,
            margin_deficit=margin_deficit,
            absolute_component=absolute_component,
            excess_component=excess_component,
            drawdown_penalty=drawdown_penalty,
            baseline_penalty=baseline_penalty,
            projection_penalty=projection_penalty,
            terminal_penalty=terminal_penalty,
            margin_penalty=margin_penalty,
            unscaled_total=unscaled_total,
            scaled_total=scaled_total,
        )


def _validate_drawdown(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0, 1]")


def _progressive_hinge(value: float, *, tolerance: float, power: float) -> float:
    if value <= 0.0:
        return 0.0
    normalized = value / tolerance
    return tolerance * (normalized + normalized**power) / 2.0


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
    """Compatibility wrapper for the v2 stateless reward contract.

    New environments use :class:`RewardTracker`. This function intentionally keeps
    the historical formula for callers that have not migrated yet.
    """

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
    _validate_drawdown(hybrid_drawdown, field_name="hybrid_drawdown")
    _validate_drawdown(shadow_drawdown, field_name="shadow_drawdown")
    if downside_penalty < 0.0 or excess_drawdown_penalty < 0.0:
        raise ValueError("reward penalties must be non-negative")
    if hybrid_terminated and not shadow_terminated:
        return -abs(scale)
    if shadow_terminated and not hybrid_terminated:
        return abs(scale)
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
