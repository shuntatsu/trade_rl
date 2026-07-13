"""Two-dimensional baseline-anchored residual action composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.strategies.trend import TrendTargets

ACTION_SCHEMA = "baseline_residual_v1"


def _normalize_gross(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if not np.isfinite(vector).all():
        raise ValueError("proposal weights must be finite")
    gross = float(np.abs(vector).sum())
    if gross > 1.0:
        vector /= gross
    return vector


@dataclass(frozen=True, slots=True)
class ResidualAction:
    """Trend-horizon interpolation and signed alpha budget."""

    trend_mix: float
    alpha_budget: float

    def __post_init__(self) -> None:
        for field, value in (
            ("trend_mix", self.trend_mix),
            ("alpha_budget", self.alpha_budget),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{field} must be finite")
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{field} must be within [-1, 1]")

    @classmethod
    def from_array(cls, value: np.ndarray) -> ResidualAction:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (2,):
            raise ValueError("residual action requires exactly two values")
        if not np.isfinite(vector).all():
            raise ValueError("residual action values must be finite")
        clipped = np.clip(vector, -1.0, 1.0)
        return cls(
            trend_mix=float(clipped[0]),
            alpha_budget=float(clipped[1]),
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.trend_mix, self.alpha_budget], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ResidualComposition:
    action: ResidualAction
    baseline: np.ndarray
    trend_component: np.ndarray
    alpha_component: np.ndarray
    proposal: np.ndarray


class BaselineResidualComposer:
    """Compose a proposal while preserving zero-action baseline identity."""

    def compose(
        self,
        action: ResidualAction,
        trends: TrendTargets,
        alpha: np.ndarray,
        *,
        alpha_enabled: bool,
    ) -> ResidualComposition:
        alpha_vector = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if alpha_vector.shape != trends.base.shape:
            raise ValueError("alpha vector shape must match trend targets")
        if not np.isfinite(alpha_vector).all():
            raise ValueError("alpha vector must be finite")
        alpha_vector = _normalize_gross(alpha_vector)

        if action.trend_mix >= 0.0:
            trend = trends.base + action.trend_mix * (trends.fast - trends.base)
        else:
            trend = trends.base + (-action.trend_mix) * (trends.slow - trends.base)

        alpha_component = (
            action.alpha_budget * alpha_vector
            if alpha_enabled
            else np.zeros_like(trends.base)
        )
        proposal = _normalize_gross(trend + alpha_component)
        return ResidualComposition(
            action=action,
            baseline=trends.base.copy(),
            trend_component=trend,
            alpha_component=alpha_component,
            proposal=proposal,
        )
