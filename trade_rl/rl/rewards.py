"""Structured absolute-growth rewards with causal baseline and risk shaping."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, fields

REWARD_SCHEMA = "baseline_residual_reward_v4"


@dataclass(frozen=True, slots=True)
class AbsoluteGrowthRewardConfig:
    """Approved economic defaults exposed as a stable standalone contract."""

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
            (
                "baseline_minimum_history_hours",
                self.baseline_minimum_history_hours,
            ),
            ("baseline_tolerance", self.baseline_tolerance),
            ("baseline_penalty_weight", self.baseline_penalty_weight),
            ("drawdown_penalty_weight", self.drawdown_penalty_weight),
            ("drawdown_free", self.drawdown_free),
            ("drawdown_middle", self.drawdown_middle),
            ("drawdown_high", self.drawdown_high),
            ("drawdown_stop", self.drawdown_stop),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        if self.baseline_window_hours <= 0.0:
            raise ValueError("baseline_window_hours must be positive")
        if not 0.0 < self.baseline_minimum_history_hours <= self.baseline_window_hours:
            raise ValueError("baseline_minimum_history_hours must be within the window")
        if (
            min(
                self.baseline_tolerance,
                self.baseline_penalty_weight,
                self.drawdown_penalty_weight,
            )
            < 0.0
        ):
            raise ValueError("reward penalties and tolerance must be non-negative")
        if not (
            0.0
            <= self.drawdown_free
            < self.drawdown_middle
            < self.drawdown_high
            < self.drawdown_stop
            < 1.0
        ):
            raise ValueError(
                "drawdown stages must be strictly increasing within [0, 1)"
            )
        if len(self.drawdown_slopes) != 3:
            raise ValueError("drawdown_slopes must contain three values")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in self.drawdown_slopes
        ):
            raise ValueError("drawdown_slopes must be finite and positive")
        if not (
            self.drawdown_slopes[0]
            <= self.drawdown_slopes[1]
            <= self.drawdown_slopes[2]
        ):
            raise ValueError("drawdown_slopes must be non-decreasing")


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Hierarchical reward weights with absolute net log growth as primary."""

    scale: float = 100.0
    absolute_growth_weight: float = 1.0
    excess_growth_weight: float = 0.0
    incremental_drawdown_weight: float = 0.05
    drawdown_dead_zone: float = 0.0
    baseline_underperformance_weight: float = 0.10
    baseline_window_hours: float = 720.0
    baseline_minimum_history_hours: float = 168.0
    baseline_window_steps: int | None = None
    baseline_minimum_history_steps: int | None = None
    baseline_tolerance: float = 0.015
    baseline_progressive_power: float = 1.0
    projection_penalty_weight: float = 0.0
    terminal_equity_weight: float = 1.0
    margin_deficit_weight: float = 1.0
    equity_floor_fraction: float = 1e-9
    drawdown_free: float = 0.05
    drawdown_middle: float = 0.10
    drawdown_high: float = 0.15
    drawdown_stop: float = 0.20
    drawdown_slopes: tuple[float, float, float] = (1.0, 3.0, 8.0)

    @classmethod
    def from_absolute_growth(cls, config: AbsoluteGrowthRewardConfig) -> RewardConfig:
        return cls(
            scale=config.scale,
            incremental_drawdown_weight=config.drawdown_penalty_weight,
            baseline_underperformance_weight=config.baseline_penalty_weight,
            baseline_window_hours=config.baseline_window_hours,
            baseline_minimum_history_hours=config.baseline_minimum_history_hours,
            baseline_tolerance=config.baseline_tolerance,
            drawdown_free=config.drawdown_free,
            drawdown_middle=config.drawdown_middle,
            drawdown_high=config.drawdown_high,
            drawdown_stop=config.drawdown_stop,
            drawdown_slopes=config.drawdown_slopes,
        )

    def absolute_growth_contract(self) -> AbsoluteGrowthRewardConfig:
        return AbsoluteGrowthRewardConfig(
            scale=self.scale,
            baseline_window_hours=self.baseline_window_hours,
            baseline_minimum_history_hours=min(
                self.baseline_minimum_history_hours, self.baseline_window_hours
            ),
            baseline_tolerance=self.baseline_tolerance,
            baseline_penalty_weight=self.baseline_underperformance_weight,
            drawdown_penalty_weight=self.incremental_drawdown_weight,
            drawdown_free=self.drawdown_free,
            drawdown_middle=self.drawdown_middle,
            drawdown_high=self.drawdown_high,
            drawdown_stop=self.drawdown_stop,
            drawdown_slopes=self.drawdown_slopes,
        )

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
            (
                "baseline_minimum_history_hours",
                self.baseline_minimum_history_hours,
            ),
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
        if self.baseline_minimum_history_hours <= 0.0:
            raise ValueError("baseline_minimum_history_hours must be positive")
        for field_name, steps_value in (
            ("baseline_window_steps", self.baseline_window_steps),
            ("baseline_minimum_history_steps", self.baseline_minimum_history_steps),
        ):
            if steps_value is not None and (
                isinstance(steps_value, bool)
                or not isinstance(steps_value, int)
                or steps_value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            self.baseline_window_steps is not None
            and self.baseline_minimum_history_steps is not None
            and self.baseline_minimum_history_steps > self.baseline_window_steps
        ):
            raise ValueError("baseline minimum history cannot exceed the window")
        if not 0.0 < self.equity_floor_fraction <= 1.0:
            raise ValueError("equity_floor_fraction must be within (0, 1]")
        # Reuse the public contract validation for stage ordering and slopes.
        self.absolute_growth_contract()


@dataclass(frozen=True, slots=True)
class RewardContext:
    rolling_hybrid_log_growth: float
    rolling_shadow_log_growth: float
    baseline_shortfall: float
    baseline_tolerance: float
    baseline_penalty: float
    hybrid_drawdown: float
    drawdown_severity: float
    history_bars: int

    @property
    def rolling_growth_gap(self) -> float:
        return self.rolling_hybrid_log_growth - self.rolling_shadow_log_growth


@dataclass(frozen=True, slots=True)
class AbsoluteGrowthRewardBreakdown:
    growth_raw: float
    baseline_penalty_delta: float
    drawdown_penalty_delta: float
    baseline_penalty_weighted: float
    drawdown_penalty_weighted: float
    total_raw: float
    total_scaled: float


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


def drawdown_severity(
    drawdown: float,
    config: AbsoluteGrowthRewardConfig | RewardConfig,
) -> float:
    _validate_drawdown(drawdown, field_name="drawdown")
    contract = (
        config
        if isinstance(config, AbsoluteGrowthRewardConfig)
        else config.absolute_growth_contract()
    )
    free = contract.drawdown_free
    middle = contract.drawdown_middle
    high = contract.drawdown_high
    first, second, third = contract.drawdown_slopes
    if drawdown <= free:
        return 0.0
    if drawdown <= middle:
        return first * (drawdown - free)
    first_band = first * (middle - free)
    if drawdown <= high:
        return first_band + second * (drawdown - middle)
    second_band = second * (high - middle)
    return first_band + second_band + third * (drawdown - high)


def _log_growth(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        if not math.isfinite(value) or value <= -1.0:
            raise ValueError("returns must be finite and greater than -1")
        total += math.log1p(value)
    return total


def build_reward_context(
    *,
    hybrid_returns: Sequence[float],
    shadow_returns: Sequence[float],
    hybrid_drawdown: float,
    window_bars: int,
    minimum_history_bars: int,
    config: AbsoluteGrowthRewardConfig,
) -> RewardContext:
    if len(hybrid_returns) != len(shadow_returns):
        raise ValueError("hybrid and shadow histories must have equal length")
    if window_bars <= 0 or minimum_history_bars <= 0:
        raise ValueError("reward history sizes must be positive")
    if minimum_history_bars > window_bars:
        raise ValueError("minimum reward history cannot exceed the window")
    hybrid_window = tuple(hybrid_returns)[-window_bars:]
    shadow_window = tuple(shadow_returns)[-window_bars:]
    history = len(hybrid_window)
    hybrid_growth = _log_growth(hybrid_window)
    shadow_growth = _log_growth(shadow_window)
    shortfall = max(0.0, shadow_growth - hybrid_growth)
    effective_tolerance = config.baseline_tolerance * min(1.0, history / window_bars)
    penalty = (
        max(0.0, shortfall - effective_tolerance)
        if history >= minimum_history_bars
        else 0.0
    )
    return RewardContext(
        rolling_hybrid_log_growth=hybrid_growth,
        rolling_shadow_log_growth=shadow_growth,
        baseline_shortfall=shortfall,
        baseline_tolerance=effective_tolerance,
        baseline_penalty=penalty,
        hybrid_drawdown=hybrid_drawdown,
        drawdown_severity=drawdown_severity(hybrid_drawdown, config),
        history_bars=history,
    )


def absolute_growth_reward(
    *,
    hybrid_log_return: float,
    before: RewardContext,
    after: RewardContext,
    config: AbsoluteGrowthRewardConfig,
) -> AbsoluteGrowthRewardBreakdown:
    if not math.isfinite(hybrid_log_return):
        raise ValueError("hybrid_log_return must be finite")
    baseline_delta = max(0.0, after.baseline_penalty - before.baseline_penalty)
    drawdown_delta = max(0.0, after.drawdown_severity - before.drawdown_severity)
    baseline_weighted = config.baseline_penalty_weight * baseline_delta
    drawdown_weighted = config.drawdown_penalty_weight * drawdown_delta
    total_raw = hybrid_log_return - baseline_weighted - drawdown_weighted
    return AbsoluteGrowthRewardBreakdown(
        growth_raw=hybrid_log_return,
        baseline_penalty_delta=baseline_delta,
        drawdown_penalty_delta=drawdown_delta,
        baseline_penalty_weighted=baseline_weighted,
        drawdown_penalty_weighted=drawdown_weighted,
        total_raw=total_raw,
        total_scaled=config.scale * total_raw,
    )


@dataclass(slots=True)
class RewardTracker:
    """State for worsening-only drawdown and rolling baseline penalties."""

    config: RewardConfig = field(default_factory=RewardConfig)
    decision_hours: float = 4.0
    previous_hybrid_drawdown: float = 0.0
    previous_shadow_drawdown: float = 0.0
    previous_baseline_hinge: float = 0.0
    _hybrid_returns: deque[float] = field(init=False, repr=False)
    _shadow_returns: deque[float] = field(init=False, repr=False)
    _minimum_history_steps: int = field(init=False, repr=False)
    last_context_before: RewardContext = field(init=False)
    last_context_after: RewardContext = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.decision_hours) or self.decision_hours <= 0.0:
            raise ValueError("decision_hours must be finite and positive")
        window_steps = self.config.baseline_window_steps
        if window_steps is None:
            window_steps = max(
                1,
                int(round(self.config.baseline_window_hours / self.decision_hours)),
            )
        minimum_steps = self.config.baseline_minimum_history_steps
        if minimum_steps is None:
            minimum_steps = max(
                1,
                int(
                    round(
                        self.config.baseline_minimum_history_hours / self.decision_hours
                    )
                ),
            )
        minimum_steps = min(minimum_steps, window_steps)
        self._minimum_history_steps = minimum_steps
        self._hybrid_returns = deque(maxlen=window_steps)
        self._shadow_returns = deque(maxlen=window_steps)
        self.reset()

    @property
    def baseline_window_steps(self) -> int:
        maxlen = self._hybrid_returns.maxlen
        if maxlen is None:  # pragma: no cover
            raise RuntimeError("reward window is not bounded")
        return maxlen

    @property
    def baseline_minimum_history_steps(self) -> int:
        return self._minimum_history_steps

    def _current_context(self, hybrid_drawdown: float) -> RewardContext:
        history = len(self._hybrid_returns)
        hybrid_growth = sum(self._hybrid_returns)
        shadow_growth = sum(self._shadow_returns)
        shortfall = max(0.0, shadow_growth - hybrid_growth)
        tolerance = self.config.baseline_tolerance * min(
            1.0, history / self.baseline_window_steps
        )
        penalty = 0.0
        if history >= self.baseline_minimum_history_steps:
            penalty = _progressive_hinge(
                max(0.0, shortfall - tolerance),
                tolerance=max(tolerance, 1e-12),
                power=self.config.baseline_progressive_power,
            )
        return RewardContext(
            rolling_hybrid_log_growth=hybrid_growth,
            rolling_shadow_log_growth=shadow_growth,
            baseline_shortfall=shortfall,
            baseline_tolerance=tolerance,
            baseline_penalty=penalty,
            hybrid_drawdown=hybrid_drawdown,
            drawdown_severity=drawdown_severity(hybrid_drawdown, self.config),
            history_bars=history,
        )

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
        self.previous_baseline_hinge = 0.0
        self._hybrid_returns.clear()
        self._shadow_returns.clear()
        context = self._current_context(hybrid_drawdown)
        self.last_context_before = context
        self.last_context_after = context

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
        del shadow_terminated
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

        self.last_context_before = self._current_context(self.previous_hybrid_drawdown)
        self._hybrid_returns.append(float(hybrid_log_return))
        self._shadow_returns.append(float(shadow_log_return))
        absolute_log_growth = float(hybrid_log_return)
        excess_log_growth = float(hybrid_log_return - shadow_log_return)

        previous_severity = drawdown_severity(
            self.previous_hybrid_drawdown, self.config
        )
        current_severity = drawdown_severity(hybrid_drawdown, self.config)
        incremental_drawdown = max(
            0.0,
            current_severity - previous_severity - self.config.drawdown_dead_zone,
        )
        self.previous_hybrid_drawdown = hybrid_drawdown
        self.previous_shadow_drawdown = shadow_drawdown

        history = len(self._hybrid_returns)
        underperformance = max(
            0.0,
            sum(self._shadow_returns) - sum(self._hybrid_returns),
        )
        effective_tolerance = self.config.baseline_tolerance * min(
            1.0, history / self.baseline_window_steps
        )
        current_hinge = 0.0
        if history >= self.baseline_minimum_history_steps:
            linear_hinge = max(0.0, underperformance - effective_tolerance)
            current_hinge = _progressive_hinge(
                linear_hinge,
                tolerance=max(effective_tolerance, 1e-12),
                power=self.config.baseline_progressive_power,
            )
        hinge_delta = max(0.0, current_hinge - self.previous_baseline_hinge)
        self.previous_baseline_hinge = current_hinge
        self.last_context_after = self._current_context(hybrid_drawdown)

        terminal_shortfall = 0.0
        if hybrid_terminated:
            safe_fraction = max(
                hybrid_equity_fraction,
                self.config.equity_floor_fraction,
            )
            terminal_shortfall = max(0.0, -math.log(safe_fraction))

        absolute_component = self.config.absolute_growth_weight * absolute_log_growth
        excess_component = self.config.excess_growth_weight * excess_log_growth
        drawdown_penalty = (
            self.config.incremental_drawdown_weight * incremental_drawdown
        )
        baseline_penalty = self.config.baseline_underperformance_weight * hinge_delta
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
            rolling_baseline_underperformance=underperformance,
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
    if power == 1.0:
        return value
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
    """Compatibility wrapper for the historical relative reward contract."""

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
