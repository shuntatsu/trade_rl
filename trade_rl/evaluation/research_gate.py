"""Fail-closed profitability gate for sealed walk-forward research evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

ObservedValue: TypeAlias = float | None

_THRESHOLDS = {
    "selected_mean_return_exclusive_minimum": 0.0,
    "baseline_uplift_minimum": 0.0,
    "maximum_independently_reset_fold_drawdown": 0.20,
}


@dataclass(frozen=True, slots=True)
class ResearchReturnGate:
    """Machine-readable profitability decision and its complete evidence."""

    thresholds: dict[str, float]
    observed: dict[str, ObservedValue]
    conditions: dict[str, bool]
    passed: bool
    evidence_errors: tuple[str, ...]


def _finite_number(value: object, *, field_name: str) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{field_name} must be a finite number"
    resolved = float(value)
    if not math.isfinite(resolved):
        return None, f"{field_name} must be a finite number"
    return resolved, None


def evaluate_research_return_gate(
    *,
    selected_mean_return: object,
    baseline_mean_return: object,
    maximum_fold_drawdown: object,
) -> ResearchReturnGate:
    """Evaluate the final research return thresholds without raising on evidence."""

    selected, selected_error = _finite_number(
        selected_mean_return,
        field_name="selected_mean_return",
    )
    baseline, baseline_error = _finite_number(
        baseline_mean_return,
        field_name="baseline_mean_return",
    )
    drawdown, drawdown_error = _finite_number(
        maximum_fold_drawdown,
        field_name="maximum_fold_drawdown",
    )
    if drawdown is not None and not 0.0 <= drawdown <= 1.0:
        drawdown = None
        drawdown_error = "maximum_fold_drawdown must be between 0 and 1"
    errors = tuple(
        error
        for error in (selected_error, baseline_error, drawdown_error)
        if error is not None
    )
    uplift = None if selected is None or baseline is None else selected - baseline
    evidence_valid = not errors
    selected_positive = selected is not None and selected > 0.0
    uplift_nonnegative = uplift is not None and uplift >= 0.0
    drawdown_within_limit = (
        drawdown is not None
        and drawdown >= 0.0
        and drawdown <= _THRESHOLDS["maximum_independently_reset_fold_drawdown"]
    )
    conditions = {
        "selected_mean_return_positive": selected_positive,
        "baseline_uplift_nonnegative": uplift_nonnegative,
        "maximum_fold_drawdown_within_limit": drawdown_within_limit,
        "evidence_valid": evidence_valid,
    }
    return ResearchReturnGate(
        thresholds=dict(_THRESHOLDS),
        observed={
            "selected_mean_return": selected,
            "baseline_mean_return": baseline,
            "baseline_uplift": uplift,
            "maximum_independently_reset_fold_drawdown": drawdown,
        },
        conditions=conditions,
        passed=all(conditions.values()),
        evidence_errors=errors,
    )
