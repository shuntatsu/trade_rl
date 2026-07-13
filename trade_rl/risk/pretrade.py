"""Pure pre-trade portfolio constraints applied before market execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PreTradeRiskConfig:
    max_gross: float = 1.0
    max_abs_weight: float = 0.40
    max_turnover: float = 1.0
    drawdown_start: float = 0.10
    drawdown_stop: float = 0.20

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_gross", self.max_gross),
            ("max_abs_weight", self.max_abs_weight),
            ("max_turnover", self.max_turnover),
            ("drawdown_start", self.drawdown_start),
            ("drawdown_stop", self.drawdown_stop),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if not 0.0 < self.max_gross <= 1.0:
            raise ValueError("max_gross must be within (0, 1]")
        if not 0.0 < self.max_abs_weight <= self.max_gross:
            raise ValueError("max_abs_weight must be within (0, max_gross]")
        if not 0.0 <= self.max_turnover <= 2.0:
            raise ValueError("max_turnover must be within [0, 2]")
        if not 0.0 <= self.drawdown_start <= 1.0:
            raise ValueError("drawdown_start must be within [0, 1]")
        if not 0.0 <= self.drawdown_stop <= 1.0:
            raise ValueError("drawdown_stop must be within [0, 1]")
        if self.drawdown_stop < self.drawdown_start:
            raise ValueError("drawdown_stop must be at least drawdown_start")


@dataclass(frozen=True, slots=True)
class RiskConstrainedTarget:
    weights: np.ndarray
    requested_turnover: float
    constrained_turnover: float
    was_constrained: bool
    reasons: tuple[str, ...]
    risk_scale: float


class PreTradeRisk:
    """Apply deterministic limits without importing execution or policy code."""

    def __init__(self, config: PreTradeRiskConfig | None = None) -> None:
        self.config = config or PreTradeRiskConfig()

    def risk_scale(self, drawdown: float) -> float:
        if not math.isfinite(drawdown) or not 0.0 <= drawdown <= 1.0:
            raise ValueError("drawdown must be finite and within [0, 1]")
        start = self.config.drawdown_start
        stop = self.config.drawdown_stop
        if drawdown <= start:
            return 1.0
        if drawdown >= stop or stop == start:
            return 0.0
        return 1.0 - (drawdown - start) / (stop - start)

    def constrain(
        self,
        target: np.ndarray,
        *,
        current: np.ndarray,
        drawdown: float,
    ) -> RiskConstrainedTarget:
        requested = np.asarray(target, dtype=np.float64).reshape(-1)
        existing = np.asarray(current, dtype=np.float64).reshape(-1)
        if requested.size == 0 or requested.shape != existing.shape:
            raise ValueError("target and current weights must have the same shape")
        if not np.isfinite(requested).all() or not np.isfinite(existing).all():
            raise ValueError("target and current weights must be finite")
        scale = self.risk_scale(drawdown)

        weights = requested.copy()
        reasons: list[str] = []

        clipped = np.clip(
            weights,
            -self.config.max_abs_weight,
            self.config.max_abs_weight,
        )
        if not np.array_equal(clipped, weights):
            weights = clipped
            reasons.append("max_abs_weight")

        gross = float(np.abs(weights).sum())
        if gross > self.config.max_gross:
            weights *= self.config.max_gross / gross
            reasons.append("max_gross")

        if scale < 1.0:
            weights *= scale
            reasons.append("drawdown_deleveraging")

        requested_turnover = float(np.abs(requested - existing).sum())
        constrained_turnover = float(np.abs(weights - existing).sum())
        if constrained_turnover > self.config.max_turnover:
            if self.config.max_turnover == 0.0:
                weights = existing.copy()
            else:
                weights = (
                    existing
                    + (weights - existing)
                    * self.config.max_turnover
                    / constrained_turnover
                )
            reasons.append("max_turnover")
            constrained_turnover = float(np.abs(weights - existing).sum())

        return RiskConstrainedTarget(
            weights=weights,
            requested_turnover=requested_turnover,
            constrained_turnover=constrained_turnover,
            was_constrained=bool(reasons),
            reasons=tuple(reasons),
            risk_scale=scale,
        )
