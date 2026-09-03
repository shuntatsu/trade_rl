"""Strict scalar target-exposure action contract for Universal Trade RL U1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NormalizedTargetExposureAction:
    """One normalized policy action and its static requested portfolio weight."""

    normalized: float
    policy_requested_weight: float


def parse_normalized_target_exposure(
    value: np.ndarray,
    *,
    policy_weight_scale: float,
) -> NormalizedTargetExposureAction:
    """Parse one strict policy action without clipping or applying Risk logic."""

    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (1,) or not np.isfinite(vector).all():
        raise ValueError("Universal Trade RL action must be one finite scalar")

    normalized = float(vector[0])
    if not -1.0 <= normalized <= 1.0:
        raise ValueError("Universal Trade RL action must be within [-1, 1]")

    if (
        isinstance(policy_weight_scale, bool)
        or not math.isfinite(policy_weight_scale)
        or not 0.0 < policy_weight_scale <= 1.0
    ):
        raise ValueError("policy_weight_scale must be finite and within (0, 1]")

    return NormalizedTargetExposureAction(
        normalized=normalized,
        policy_requested_weight=normalized * policy_weight_scale,
    )


__all__ = [
    "NormalizedTargetExposureAction",
    "parse_normalized_target_exposure",
]
