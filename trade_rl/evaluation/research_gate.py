"""Fail-closed profitability gate for sealed walk-forward research evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

from trade_rl.domain.common import require_sha256

PolicyIdentity: TypeAlias = str | None
ObservedValue: TypeAlias = float | tuple[PolicyIdentity, ...] | None

_REQUIRED_SEALED_FOLD_COUNT = 2

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


def _finite_number(
    value: object, *, field_name: str
) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{field_name} must be a finite number"
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError):
        return None, f"{field_name} must be a finite number"
    if not math.isfinite(resolved):
        return None, f"{field_name} must be a finite number"
    return resolved, None


def _total_return(
    value: object,
    *,
    field_name: str,
) -> tuple[float | None, str | None]:
    resolved, error = _finite_number(value, field_name=field_name)
    if resolved is not None and resolved < -1.0:
        return None, f"{field_name} must be greater than or equal to -1"
    return resolved, error


def _selected_policy_identities(
    value: object,
) -> tuple[tuple[PolicyIdentity, ...] | None, tuple[str, ...]]:
    if not isinstance(value, (list, tuple)):
        return None, ("selected_policy_digests must be a sequence",)
    identities: list[PolicyIdentity] = []
    canonical_identities: list[str] = []
    errors: list[str] = []
    if len(value) != _REQUIRED_SEALED_FOLD_COUNT:
        errors.append(
            "selected_policy_digests must contain exactly "
            f"{_REQUIRED_SEALED_FOLD_COUNT} identities"
        )
    for index, identity in enumerate(value):
        if not isinstance(identity, str) or not identity:
            identities.append(None)
            errors.append(
                f"selected_policy_digests[{index}] must be a non-empty string"
            )
        else:
            identities.append(identity)
            try:
                canonical_identities.append(
                    require_sha256(
                        identity,
                        field=f"selected_policy_digests[{index}]",
                    )
                )
            except ValueError as exc:
                errors.append(str(exc))
    if len(canonical_identities) == _REQUIRED_SEALED_FOLD_COUNT and len(
        set(canonical_identities)
    ) != len(canonical_identities):
        errors.append("selected_policy_digests must contain unique identities")
    return tuple(identities), tuple(errors)


def evaluate_research_return_gate(
    *,
    selected_mean_return: object,
    baseline_mean_return: object,
    maximum_fold_drawdown: object,
    selected_policy_digests: object,
) -> ResearchReturnGate:
    """Evaluate the final research return thresholds without raising on evidence."""

    selected, selected_error = _total_return(
        selected_mean_return,
        field_name="selected_mean_return",
    )
    baseline, baseline_error = _total_return(
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
    uplift = None if selected is None or baseline is None else selected - baseline
    uplift_error = None
    if uplift is not None and not math.isfinite(uplift):
        uplift = None
        uplift_error = "baseline_uplift must be a finite number"
    policy_identities, policy_identity_errors = _selected_policy_identities(
        selected_policy_digests
    )
    errors = (
        tuple(
            error
            for error in (
                selected_error,
                baseline_error,
                drawdown_error,
                uplift_error,
            )
            if error is not None
        )
        + policy_identity_errors
    )
    evidence_valid = not errors
    selected_positive = selected is not None and selected > 0.0
    uplift_nonnegative = uplift is not None and uplift >= 0.0
    drawdown_within_limit = (
        drawdown is not None
        and drawdown >= 0.0
        and drawdown <= _THRESHOLDS["maximum_independently_reset_fold_drawdown"]
    )
    rl_policy_selected_all_folds = (
        policy_identities is not None and not policy_identity_errors
    )
    conditions = {
        "selected_mean_return_positive": selected_positive,
        "baseline_uplift_nonnegative": uplift_nonnegative,
        "maximum_fold_drawdown_within_limit": drawdown_within_limit,
        "rl_policy_selected_all_folds": rl_policy_selected_all_folds,
        "evidence_valid": evidence_valid,
    }
    return ResearchReturnGate(
        thresholds=dict(_THRESHOLDS),
        observed={
            "selected_mean_return": selected,
            "baseline_mean_return": baseline,
            "baseline_uplift": uplift,
            "maximum_independently_reset_fold_drawdown": drawdown,
            "selected_policy_digests": policy_identities,
        },
        conditions=conditions,
        passed=all(conditions.values()),
        evidence_errors=errors,
    )
