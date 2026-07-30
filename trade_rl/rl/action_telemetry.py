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
    current_weights: object,
    deterministic_composed: object,
    sampled_policy_action: object,
    submitted_target: object | None = None,
    effective_filled_weights: object | None = None,
) -> dict[str, float]:
    """Return causal L1 distances across hierarchical action stages.

    ``deterministic_composed`` is the Gate-interpolated policy mean before
    Gaussian exploration. ``sampled_policy_action`` is the actual stochastic
    action supplied to the environment. Submission and fill stages are
    optional so pre-execution diagnostics remain available when an environment
    does not expose downstream vectors.
    """

    current = _vector(current_weights, field="current_weights")
    deterministic = _vector(deterministic_composed, field="deterministic_composed")
    sampled = _vector(sampled_policy_action, field="sampled_policy_action")
    vectors = (current, deterministic, sampled)
    if len({len(vector) for vector in vectors}) != 1:
        raise ValueError("hierarchical action stages must have equal dimensions")

    metrics = {
        "deterministic_change_l1": float(np.sum(np.abs(deterministic - current))),
        "exploration_l1": float(np.sum(np.abs(sampled - deterministic))),
        "sampled_change_l1": float(np.sum(np.abs(sampled - current))),
    }
    if submitted_target is not None:
        submitted = _vector(submitted_target, field="submitted_target")
        if len(submitted) != len(sampled):
            raise ValueError("hierarchical action stages must have equal dimensions")
        metrics["submission_l1"] = float(np.sum(np.abs(submitted - sampled)))
    if effective_filled_weights is not None:
        effective = _vector(
            effective_filled_weights,
            field="effective_filled_weights",
        )
        if len(effective) != len(sampled):
            raise ValueError("hierarchical action stages must have equal dimensions")
        metrics["effective_action_l1"] = float(np.sum(np.abs(effective - sampled)))
    return metrics


__all__ = ["hierarchical_action_stage_metrics"]
