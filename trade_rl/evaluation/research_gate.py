"""Fail-closed profitability gates for sealed walk-forward evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from trade_rl.domain.common import require_sha256

PolicyIdentity: TypeAlias = str | None
ObservedValue: TypeAlias = object

_BASE_THRESHOLDS = {
    "selected_mean_return_exclusive_minimum": 0.0,
    "baseline_uplift_minimum": 0.0,
    "maximum_independently_reset_fold_drawdown": 0.20,
    "maximum_turnover_per_day": 1.0,
    "maximum_cost_fraction": 0.03,
}


@dataclass(frozen=True, slots=True)
class ResearchEvidenceRequirements:
    """Materiality requirements applied only to maintained research runs."""

    required_fold_count: int = 2
    minimum_oos_days: float = 0.0
    require_positive_bootstrap_lower_bound: bool = False
    require_confirmation: bool = False
    minimum_confirmation_days: float = 0.0
    minimum_baseline_uplift: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_fold_count, bool)
            or not isinstance(self.required_fold_count, int)
            or self.required_fold_count <= 0
        ):
            raise ValueError("required_fold_count must be a positive integer")
        for name, value in (
            ("minimum_oos_days", self.minimum_oos_days),
            ("minimum_confirmation_days", self.minimum_confirmation_days),
            ("minimum_baseline_uplift", self.minimum_baseline_uplift),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            (
                "require_positive_bootstrap_lower_bound",
                self.require_positive_bootstrap_lower_bound,
            ),
            ("require_confirmation", self.require_confirmation),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class ResearchReturnGate:
    thresholds: dict[str, float]
    observed: dict[str, ObservedValue]
    conditions: dict[str, bool]
    passed: bool
    evidence_errors: tuple[str, ...]


def block_bootstrap_mean_lower_bound(
    daily_returns: object,
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    block_size: int = 5,
    seed: int = 0,
) -> float:
    """Deterministic circular block-bootstrap lower bound on mean daily growth."""

    values = np.asarray(daily_returns, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all() or np.any(values < -1.0):
        raise ValueError("daily_returns must contain at least two finite returns")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be within (0.5, 1)")
    for name, value in (("samples", samples), ("block_size", block_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    log_returns = np.log1p(values)
    rng = np.random.default_rng(seed)
    block_count = math.ceil(values.size / block_size)
    means = np.empty(samples, dtype=np.float64)
    offsets = np.arange(block_size, dtype=np.int64)
    for sample_index in range(samples):
        starts = rng.integers(0, values.size, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % values.size).reshape(-1)
        means[sample_index] = float(np.mean(log_returns[indices[: values.size]]))
    lower_log = float(np.quantile(means, 1.0 - confidence))
    return float(np.expm1(lower_log))


def paired_block_bootstrap_excess_lower_bound(
    selected_daily_returns: object,
    baseline_daily_returns: object,
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    block_size: int = 5,
    seed: int = 0,
) -> float:
    """Lower confidence bound for paired daily log-return excess."""

    selected = np.asarray(selected_daily_returns, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline_daily_returns, dtype=np.float64).reshape(-1)
    if selected.shape != baseline.shape or selected.size < 2:
        raise ValueError(
            "selected and baseline daily returns must have the same length of at least two"
        )
    if (
        not np.isfinite(selected).all()
        or not np.isfinite(baseline).all()
        or np.any(selected < -1.0)
        or np.any(baseline < -1.0)
    ):
        raise ValueError("paired daily returns must be finite and at least -1")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be within (0.5, 1)")
    for name, value in (("samples", samples), ("block_size", block_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    excess = np.log1p(selected) - np.log1p(baseline)
    if np.all(excess == 0.0):
        return 0.0
    rng = np.random.default_rng(seed)
    block_count = math.ceil(excess.size / block_size)
    means = np.empty(samples, dtype=np.float64)
    offsets = np.arange(block_size, dtype=np.int64)
    for sample_index in range(samples):
        starts = rng.integers(0, excess.size, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % excess.size).reshape(-1)
        means[sample_index] = float(np.mean(excess[indices[: excess.size]]))
    return float(np.expm1(np.quantile(means, 1.0 - confidence)))


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


def _positive_integer(
    value: object, *, field_name: str
) -> tuple[int | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None, f"{field_name} must be a positive integer"
    return value, None


def _total_return(value: object, *, field_name: str) -> tuple[float | None, str | None]:
    resolved, error = _finite_number(value, field_name=field_name)
    if resolved is not None and resolved < -1.0:
        return None, f"{field_name} must be greater than or equal to -1"
    return resolved, error


def _selected_policy_identities(
    value: object,
    *,
    expected_count: int,
) -> tuple[tuple[PolicyIdentity, ...] | None, tuple[str, ...]]:
    if not isinstance(value, (list, tuple)):
        return None, ("selected_policy_digests must be a sequence",)
    identities: list[PolicyIdentity] = []
    canonical_identities: list[str] = []
    errors: list[str] = []
    if len(value) != expected_count:
        errors.append(
            f"selected_policy_digests must contain exactly {expected_count} identities"
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
                    require_sha256(identity, field=f"selected_policy_digests[{index}]")
                )
            except ValueError as exc:
                errors.append(str(exc))
    if len(canonical_identities) == expected_count and len(
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
    maximum_turnover_per_day: object = 0.0,
    maximum_cost_fraction: object = 0.0,
    selection_stability_passed: object = True,
    sealed_fold_count: object = None,
    oos_days: object = None,
    bootstrap_lower_bound: object = None,
    confirmation_passed: object = None,
    confirmation_days: object = None,
    requirements: ResearchEvidenceRequirements | None = None,
) -> ResearchReturnGate:
    """Evaluate base or material research thresholds without raising on evidence."""

    strict = requirements is not None
    resolved_requirements = requirements or ResearchEvidenceRequirements()
    fold_count: int | None = None
    fold_count_error: str | None = None
    if strict:
        fold_count, fold_count_error = _positive_integer(
            sealed_fold_count, field_name="sealed_fold_count"
        )
    expected_count = (
        fold_count
        if strict and fold_count is not None
        else resolved_requirements.required_fold_count
    )
    selected, selected_error = _total_return(
        selected_mean_return, field_name="selected_mean_return"
    )
    baseline, baseline_error = _total_return(
        baseline_mean_return, field_name="baseline_mean_return"
    )
    drawdown, drawdown_error = _finite_number(
        maximum_fold_drawdown, field_name="maximum_fold_drawdown"
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
        selected_policy_digests,
        expected_count=expected_count,
    )
    turnover, turnover_error = _finite_number(
        maximum_turnover_per_day, field_name="maximum_turnover_per_day"
    )
    if turnover is not None and turnover < 0.0:
        turnover = None
        turnover_error = "maximum_turnover_per_day must be non-negative"
    cost_fraction, cost_error = _finite_number(
        maximum_cost_fraction, field_name="maximum_cost_fraction"
    )
    if cost_fraction is not None and cost_fraction < 0.0:
        cost_fraction = None
        cost_error = "maximum_cost_fraction must be non-negative"
    stability_error = (
        None
        if isinstance(selection_stability_passed, bool)
        else "selection_stability_passed must be a boolean"
    )

    strict_errors: list[str] = []
    oos: float | None = None
    bootstrap: float | None = None
    confirmation_duration: float | None = None
    if strict:
        if fold_count_error is not None:
            strict_errors.append(fold_count_error)
        oos, error = _finite_number(oos_days, field_name="oos_days")
        if error is not None:
            strict_errors.append(error)
        elif oos is not None and oos < 0.0:
            oos = None
            strict_errors.append("oos_days must be non-negative")
        if resolved_requirements.require_positive_bootstrap_lower_bound:
            bootstrap, error = _finite_number(
                bootstrap_lower_bound, field_name="bootstrap_lower_bound"
            )
            if error is not None:
                strict_errors.append(error)
        if resolved_requirements.require_confirmation:
            if not isinstance(confirmation_passed, bool):
                strict_errors.append("confirmation_passed must be a boolean")
            confirmation_duration, error = _finite_number(
                confirmation_days, field_name="confirmation_days"
            )
            if error is not None:
                strict_errors.append(error)
            elif confirmation_duration is not None and confirmation_duration < 0.0:
                confirmation_duration = None
                strict_errors.append("confirmation_days must be non-negative")

    errors = (
        tuple(
            error
            for error in (
                selected_error,
                baseline_error,
                drawdown_error,
                uplift_error,
                turnover_error,
                cost_error,
                stability_error,
            )
            if error is not None
        )
        + policy_identity_errors
        + tuple(strict_errors)
    )
    evidence_valid = not errors
    conditions = {
        "selected_mean_return_positive": selected is not None and selected > 0.0,
        "baseline_uplift_nonnegative": uplift is not None and uplift >= 0.0,
        "maximum_fold_drawdown_within_limit": (
            drawdown is not None
            and 0.0
            <= drawdown
            <= _BASE_THRESHOLDS["maximum_independently_reset_fold_drawdown"]
        ),
        "rl_policy_selected_all_folds": (
            policy_identities is not None and not policy_identity_errors
        ),
        "turnover_within_limit": (
            turnover is not None
            and turnover <= _BASE_THRESHOLDS["maximum_turnover_per_day"]
        ),
        "cost_fraction_within_limit": (
            cost_fraction is not None
            and cost_fraction <= _BASE_THRESHOLDS["maximum_cost_fraction"]
        ),
        "selection_stability_passed": selection_stability_passed is True,
        "evidence_valid": evidence_valid,
    }
    thresholds = dict(_BASE_THRESHOLDS)
    observed: dict[str, ObservedValue] = {
        "selected_mean_return": selected,
        "baseline_mean_return": baseline,
        "baseline_uplift": uplift,
        "maximum_independently_reset_fold_drawdown": drawdown,
        "selected_policy_digests": policy_identities,
        "maximum_turnover_per_day": turnover,
        "maximum_cost_fraction": cost_fraction,
        "selection_stability_passed": (
            selection_stability_passed
            if isinstance(selection_stability_passed, bool)
            else None
        ),
    }
    if strict:
        thresholds.update(
            {
                "minimum_sealed_fold_count": float(
                    resolved_requirements.required_fold_count
                ),
                "minimum_oos_days": resolved_requirements.minimum_oos_days,
            }
        )
        conditions["minimum_fold_count_met"] = (
            fold_count is not None
            and fold_count >= resolved_requirements.required_fold_count
        )
        conditions["minimum_oos_days_met"] = (
            oos is not None and oos >= resolved_requirements.minimum_oos_days
        )
        thresholds["minimum_material_baseline_uplift"] = (
            resolved_requirements.minimum_baseline_uplift
        )
        conditions["material_baseline_uplift_met"] = (
            uplift is not None
            and uplift >= resolved_requirements.minimum_baseline_uplift
        )
        observed["sealed_fold_count"] = fold_count
        observed["oos_days"] = oos
        if resolved_requirements.require_positive_bootstrap_lower_bound:
            thresholds["bootstrap_lower_bound_exclusive_minimum"] = 0.0
            conditions["bootstrap_lower_bound_positive"] = (
                bootstrap is not None and bootstrap > 0.0
            )
            observed["bootstrap_lower_bound"] = bootstrap
        if resolved_requirements.require_confirmation:
            thresholds["minimum_confirmation_days"] = (
                resolved_requirements.minimum_confirmation_days
            )
            conditions["fresh_confirmation_passed"] = confirmation_passed is True
            conditions["minimum_confirmation_days_met"] = (
                confirmation_duration is not None
                and confirmation_duration
                >= resolved_requirements.minimum_confirmation_days
            )
            observed["confirmation_passed"] = (
                confirmation_passed if isinstance(confirmation_passed, bool) else None
            )
            observed["confirmation_days"] = confirmation_duration
    return ResearchReturnGate(
        thresholds=thresholds,
        observed=observed,
        conditions=conditions,
        passed=all(conditions.values()),
        evidence_errors=errors,
    )


__all__ = [
    "ResearchEvidenceRequirements",
    "ResearchReturnGate",
    "block_bootstrap_mean_lower_bound",
    "paired_block_bootstrap_excess_lower_bound",
    "evaluate_research_return_gate",
]
