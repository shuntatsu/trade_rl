"""Pure pre-trade portfolio constraints applied before market execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class PreTradeRiskConfig:
    max_gross: float = 1.0
    max_abs_weight: float = 0.40
    max_turnover: float = 1.0
    drawdown_start: float = 0.10
    drawdown_stop: float = 0.20
    emergency_turnover_override: bool = True
    fail_closed_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_gross", self.max_gross),
            ("max_abs_weight", self.max_abs_weight),
            ("max_turnover", self.max_turnover),
            ("drawdown_start", self.drawdown_start),
            ("drawdown_stop", self.drawdown_stop),
            ("fail_closed_tolerance", self.fail_closed_tolerance),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if not 0.0 < self.max_gross <= 10.0:
            raise ValueError("max_gross must be within (0, 10]")
        if not 0.0 < self.max_abs_weight <= self.max_gross:
            raise ValueError("max_abs_weight must be within (0, max_gross]")
        if not 0.0 <= self.max_turnover <= 2.0 * self.max_gross:
            raise ValueError("max_turnover must be within [0, 2 * max_gross]")
        if not 0.0 <= self.drawdown_start <= 1.0:
            raise ValueError("drawdown_start must be within [0, 1]")
        if not 0.0 <= self.drawdown_stop <= 1.0:
            raise ValueError("drawdown_stop must be within [0, 1]")
        if self.drawdown_stop < self.drawdown_start:
            raise ValueError("drawdown_stop must be at least drawdown_start")
        if self.fail_closed_tolerance < 0.0:
            raise ValueError("fail_closed_tolerance must be non-negative")
        if not isinstance(self.emergency_turnover_override, bool):
            raise ValueError("emergency_turnover_override must be a boolean")


@dataclass(frozen=True, slots=True)
class RiskConstrainedTarget:
    weights: np.ndarray
    requested_turnover: float
    constrained_turnover: float
    was_constrained: bool
    reasons: tuple[str, ...]
    risk_scale: float
    projection_l1: float = 0.0
    turnover_overridden: bool = False


class PreTradeRisk:
    """Apply soft trading limits followed by non-negotiable hard limits."""

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

    def _apply_hard_limits(
        self,
        value: np.ndarray,
        *,
        risk_scale: float,
        reasons: list[str],
    ) -> np.ndarray:
        weights = np.asarray(value, dtype=np.float64).reshape(-1).copy()
        clipped = np.clip(
            weights,
            -self.config.max_abs_weight,
            self.config.max_abs_weight,
        )
        if not np.allclose(clipped, weights, atol=0.0, rtol=0.0):
            weights = clipped
            reasons.append("max_abs_weight")
        gross = float(np.abs(weights).sum())
        if gross > self.config.max_gross:
            weights *= self.config.max_gross / gross
            reasons.append("max_gross")
        if risk_scale < 1.0:
            weights *= risk_scale
            reasons.append("drawdown_deleveraging")
        return weights

    def _validate_final(self, weights: np.ndarray, *, risk_scale: float) -> None:
        tolerance = self.config.fail_closed_tolerance
        if not np.isfinite(weights).all():
            raise RuntimeError("risk projection produced non-finite weights")
        if np.max(np.abs(weights), initial=0.0) > (
            self.config.max_abs_weight * risk_scale + tolerance
        ):
            raise RuntimeError("risk projection violated max_abs_weight")
        if float(np.abs(weights).sum()) > (
            self.config.max_gross * risk_scale + tolerance
        ):
            raise RuntimeError("risk projection violated max_gross")
        if risk_scale == 0.0 and np.any(np.abs(weights) > tolerance):
            raise RuntimeError("emergency drawdown stop did not flatten target")

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
        reasons: list[str] = []
        requested_turnover = float(np.abs(requested - existing).sum())

        # Turnover is a soft operational limit. It is applied before hard risk limits
        # so that an already-invalid current portfolio cannot block deleveraging.
        weights = requested.copy()
        if requested_turnover > self.config.max_turnover:
            if self.config.max_turnover == 0.0:
                weights = existing.copy()
            else:
                weights = existing + (requested - existing) * (
                    self.config.max_turnover / requested_turnover
                )
            reasons.append("max_turnover")

        before_hard = weights.copy()
        weights = self._apply_hard_limits(
            weights,
            risk_scale=scale,
            reasons=reasons,
        )
        hard_changed = not np.allclose(
            before_hard,
            weights,
            atol=_TOLERANCE,
            rtol=0.0,
        )
        turnover_overridden = False
        constrained_turnover = float(np.abs(weights - existing).sum())
        if hard_changed and constrained_turnover > self.config.max_turnover + _TOLERANCE:
            if self.config.emergency_turnover_override:
                turnover_overridden = True
                reasons.append("hard_risk_turnover_override")
            else:
                raise RuntimeError(
                    "hard risk target requires turnover above configured maximum"
                )

        self._validate_final(weights, risk_scale=scale)
        projection_l1 = float(np.abs(requested - weights).sum())
        return RiskConstrainedTarget(
            weights=weights,
            requested_turnover=requested_turnover,
            constrained_turnover=constrained_turnover,
            was_constrained=bool(reasons),
            reasons=tuple(dict.fromkeys(reasons)),
            risk_scale=scale,
            projection_l1=projection_l1,
            turnover_overridden=turnover_overridden,
        )
