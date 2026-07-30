"""Causal quality gate for direct target-weight behavior cloning."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from trade_rl.learning.evaluation import (
    BehaviorCloningGateEvaluation,
    BehaviorCloningGateGroup,
    BehaviorCloningGateMetric,
    BehaviorCloningGateThresholds,
    evaluate_behavior_cloning_gates,
)


def _relative_improvement(initial_mse: object, final_mse: object) -> float | None:
    values = (initial_mse, final_mse)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        return None
    initial = float(initial_mse)
    final = float(final_mse)
    return (initial - final) / max(initial, float(np.finfo(np.float64).eps))


def evaluate_direct_behavior_cloning_gates(
    *,
    initial_mse: object,
    final_mse: object,
    teacher_change_support: int,
    holdout: Any,
    thresholds: BehaviorCloningGateThresholds,
) -> BehaviorCloningGateEvaluation:
    """Require direct-head reconstruction plus the canonical causal holdout gate."""

    if (
        isinstance(teacher_change_support, bool)
        or not isinstance(teacher_change_support, int)
        or teacher_change_support < 0
    ):
        raise ValueError("teacher_change_support must be a non-negative integer")
    improvement = _relative_improvement(initial_mse, final_mse)
    minimum_support = thresholds.minimum_teacher_positive_support
    if teacher_change_support < minimum_support:
        status = "insufficient_support"
        reason = (
            "action_mse_relative_improvement has support "
            f"{teacher_change_support}; minimum required support is {minimum_support}"
        )
    elif improvement is None:
        status = "insufficient_support"
        reason = "action_mse_relative_improvement is unavailable"
    elif improvement >= thresholds.minimum_composed_loss_relative_improvement:
        status = "passed"
        reason = "action_mse_relative_improvement passed"
    else:
        status = "failed"
        reason = "action-MSE improvement is below the required threshold"
    teacher_metric = BehaviorCloningGateMetric(
        name="action_mse_relative_improvement",
        status=status,
        observed=improvement,
        comparison=">=",
        threshold=thresholds.minimum_composed_loss_relative_improvement,
        support=teacher_change_support,
        minimum_support=minimum_support,
        reason=reason,
    )
    synthetic_metrics = SimpleNamespace(
        positive_support=teacher_change_support,
        active_target_rmse=0.0,
        activity_ratio=1.0,
        gate_precision=1.0,
        predicted_positive_support=teacher_change_support,
        gate_recall=1.0,
        constant_action_collapse=False,
        all_hold_collapse=False,
        all_trade_collapse=False,
    )
    canonical = evaluate_behavior_cloning_gates(
        initial_composed_loss=float(initial_mse)
        if isinstance(initial_mse, int | float) and not isinstance(initial_mse, bool)
        else None,
        final_composed_loss=float(final_mse)
        if isinstance(final_mse, int | float) and not isinstance(final_mse, bool)
        else None,
        reconstruction_metrics=synthetic_metrics,
        holdout=holdout,
        thresholds=thresholds,
    )
    return BehaviorCloningGateEvaluation(
        teacher_reconstruction_gate=BehaviorCloningGateGroup(
            name="teacher_reconstruction_gate",
            metrics=(teacher_metric,),
        ),
        causal_non_collapse_gate=canonical.causal_non_collapse_gate,
    )


__all__ = ["evaluate_direct_behavior_cloning_gates"]
