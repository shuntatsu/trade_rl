from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mars_lite.trading.trend_family import TrendTargets


@dataclass(frozen=True)
class CompositionResult:
    proposal: np.ndarray
    trend_weights: np.ndarray
    trend_mix: float
    alpha_budget: float


class BaselineResidualComposer:
    """Map a two-dimensional residual action onto executable proposal weights."""

    def __init__(self, alpha_budget_max: float = 0.30, max_gross: float = 1.0):
        if not np.isfinite(alpha_budget_max) or not 0.0 <= alpha_budget_max <= 1.0:
            raise ValueError("alpha_budget_max must be finite and within [0, 1]")
        if not np.isfinite(max_gross) or max_gross <= 0.0:
            raise ValueError("max_gross must be finite and positive")
        self.alpha_budget_max = float(alpha_budget_max)
        self.max_gross = float(max_gross)

    def compose(
        self,
        action: np.ndarray,
        trends: TrendTargets,
        alpha: np.ndarray,
        *,
        alpha_enabled: bool = True,
    ) -> CompositionResult:
        residual_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if residual_action.shape != (2,):
            raise ValueError(f"action shape must be (2,), got {residual_action.shape}")
        if not np.all(np.isfinite(residual_action)):
            raise ValueError("action must be finite")
        residual_action = np.clip(residual_action, -1.0, 1.0)

        fast = np.asarray(trends.fast, dtype=np.float64)
        base = np.asarray(trends.base, dtype=np.float64)
        slow = np.asarray(trends.slow, dtype=np.float64)
        alpha_weights = np.asarray(alpha, dtype=np.float64)
        if any(x.shape != base.shape for x in (fast, slow, alpha_weights)):
            raise ValueError("trend and alpha shapes must match")
        if not all(np.all(np.isfinite(x)) for x in (fast, base, slow, alpha_weights)):
            raise ValueError("trend and alpha weights must be finite")

        trend_mix = float(residual_action[0])
        if trend_mix >= 0.0:
            trend_weights = (1.0 - trend_mix) * base + trend_mix * fast
        else:
            magnitude = abs(trend_mix)
            trend_weights = (1.0 - magnitude) * base + magnitude * slow

        alpha_budget = (
            self.alpha_budget_max * float(residual_action[1]) if alpha_enabled else 0.0
        )
        proposal = (
            1.0 - abs(alpha_budget)
        ) * trend_weights + alpha_budget * alpha_weights
        proposal = self._project_gross(proposal)
        return CompositionResult(
            proposal=proposal,
            trend_weights=trend_weights,
            trend_mix=trend_mix,
            alpha_budget=alpha_budget,
        )

    def _project_gross(self, weights: np.ndarray) -> np.ndarray:
        gross = float(np.abs(weights).sum())
        return weights * (self.max_gross / gross) if gross > self.max_gross else weights
