"""Metrics that distinguish deterministic composition, exploration and fills."""

from __future__ import annotations

import numpy as np


def _vector(value: object, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    return array


def hierarchical_action_stage_metrics(
    *,
    deterministic_composed: object,
    sampled_policy_action: object,
    submitted_target: object,
    effective_filled_weights: object,
) -> dict[str, float]:
    deterministic = _vector(deterministic_composed, field="deterministic_composed")
    sampled = _vector(sampled_policy_action, field="sampled_policy_action")
    submitted = _vector(submitted_target, field="submitted_target")
    effective = _vector(effective_filled_weights, field="effective_filled_weights")
    if len({len(deterministic), len(sampled), len(submitted), len(effective)}) != 1:
        raise ValueError("hierarchical action stages must have equal dimensions")
    return {
        "exploration_l1": float(np.sum(np.abs(sampled - deterministic))),
        "submission_l1": float(np.sum(np.abs(submitted - sampled))),
        "effective_action_l1": float(np.sum(np.abs(effective - sampled))),
    }


__all__ = ["hierarchical_action_stage_metrics"]
