from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HTFConstraintResult:
    weights: np.ndarray
    zeroed_fraction: float
    neutral_scaled_fraction: float


class HTFProposalConstraint:
    """Apply HTF direction/neutral rules to a desired proposal, not inventory."""

    def __init__(self, threshold: float = 0.3, neutral_scale: float = 0.5):
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be finite and non-negative")
        if not np.isfinite(neutral_scale) or not 0.0 <= neutral_scale <= 1.0:
            raise ValueError("neutral_scale must be finite and within [0, 1]")
        self.threshold = float(threshold)
        self.neutral_scale = float(neutral_scale)

    def apply(
        self, proposal: np.ndarray, htf_trend: np.ndarray
    ) -> HTFConstraintResult:
        weights = np.asarray(proposal, dtype=np.float64).reshape(-1)
        trend = np.asarray(htf_trend, dtype=np.float64).reshape(-1)
        if weights.shape != trend.shape:
            raise ValueError("proposal and htf_trend shapes must match")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(trend)):
            raise ValueError("proposal and htf_trend must be finite")

        constrained = weights.copy()
        zeroed = 0
        neutral = 0
        for idx, (weight, htf_value) in enumerate(zip(weights, trend)):
            if htf_value > self.threshold and weight < 0.0:
                constrained[idx] = 0.0
                zeroed += 1
            elif htf_value < -self.threshold and weight > 0.0:
                constrained[idx] = 0.0
                zeroed += 1
            elif abs(htf_value) <= self.threshold:
                constrained[idx] = weight * self.neutral_scale
                neutral += 1

        count = max(len(constrained), 1)
        return HTFConstraintResult(
            weights=constrained,
            zeroed_fraction=zeroed / count,
            neutral_scaled_fraction=neutral / count,
        )
