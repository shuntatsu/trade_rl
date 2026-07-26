"""Deterministic raw-unit diagnostics for Lagrangian PPO stability."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trade_rl.rl.lagrangian import DualUpdateReport

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class VectorStatistics:
    """Finite descriptive statistics for one raw diagnostic vector."""

    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    l2_norm: float

    def payload(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "l2_norm": self.l2_norm,
        }


@dataclass(frozen=True, slots=True)
class ConstraintCorrelationDiagnostics:
    """Raw actor-penalty and observational correlation diagnostics."""

    cost_names: tuple[str, ...]
    reward_advantages: np.ndarray
    raw_cost_advantages: np.ndarray
    penalty_contributions: np.ndarray
    aggregate_penalty: np.ndarray
    raw_cost_covariance: np.ndarray
    raw_cost_correlation: np.ndarray
    normalized_cost_advantage_correlation: np.ndarray | None
    raw_reward_advantage_statistics: VectorStatistics
    raw_cost_advantage_statistics: tuple[VectorStatistics, ...]
    raw_effective_penalty_statistics: tuple[VectorStatistics, ...]
    penalty_to_reward_l2_ratio: float


@dataclass(frozen=True, slots=True)
class ConstraintStabilityDiagnostics:
    """Per-constraint dual-boundary and residual history diagnostics."""

    name: str
    rollout_count: int
    saturation_fraction: float
    lower_bound_fraction: float
    longest_saturation_run: int
    update_sign_change_frequency: float
    multiplier_variance: float
    residual_variance: float
    violation_area: float
    longest_satisfaction_run: int
    post_satisfaction_overconstraint_count: int

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "rollout_count": self.rollout_count,
            "saturation_fraction": self.saturation_fraction,
            "lower_bound_fraction": self.lower_bound_fraction,
            "longest_saturation_run": self.longest_saturation_run,
            "update_sign_change_frequency": self.update_sign_change_frequency,
            "multiplier_variance": self.multiplier_variance,
            "residual_variance": self.residual_variance,
            "violation_area": self.violation_area,
            "longest_satisfaction_run": self.longest_satisfaction_run,
            "post_satisfaction_overconstraint_count": (
                self.post_satisfaction_overconstraint_count
            ),
        }


@dataclass(frozen=True, slots=True)
class DualStabilityDiagnostics:
    """Canonical-order stability diagnostics for all constraints."""

    cost_names: tuple[str, ...]
    constraints: tuple[ConstraintStabilityDiagnostics, ...]

    def payload(self) -> dict[str, object]:
        return {
            "cost_names": list(self.cost_names),
            "constraints": [constraint.payload() for constraint in self.constraints],
        }


def _validated_cost_names(cost_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(cost_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("cost_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("cost_names must not contain duplicates")
    return names


def _finite_array(
    value: np.ndarray,
    *,
    field_name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{field_name} must be a non-empty {ndim}D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain finite values")
    return array


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


def _vector_statistics(value: np.ndarray) -> VectorStatistics:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("diagnostic vector must be non-empty and finite")
    return VectorStatistics(
        mean=float(np.mean(vector)),
        standard_deviation=float(np.std(vector)),
        minimum=float(np.min(vector)),
        maximum=float(np.max(vector)),
        l2_norm=float(np.linalg.norm(vector)),
    )


def _covariance_and_correlation(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(value, dtype=np.float64)
    sample_count, column_count = matrix.shape
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    denominator = sample_count - 1
    if denominator <= 0:
        covariance = np.zeros((column_count, column_count), dtype=np.float64)
    else:
        covariance = centered.T @ centered / denominator
    covariance = np.asarray(covariance, dtype=np.float64)
    variances = np.maximum(np.diag(covariance), 0.0)
    standard_deviations = np.sqrt(variances)
    scale = np.outer(standard_deviations, standard_deviations)
    correlation = np.zeros_like(covariance)
    non_constant = scale > _EPSILON
    correlation[non_constant] = covariance[non_constant] / scale[non_constant]
    correlation = np.clip(correlation, -1.0, 1.0)
    return _readonly(covariance), _readonly(correlation)


def build_constraint_correlation_diagnostics(
    *,
    cost_names: Sequence[str],
    raw_costs: np.ndarray,
    raw_cost_advantages: np.ndarray,
    normalized_cost_advantages: np.ndarray | None,
    multipliers: np.ndarray,
    reward_advantages: np.ndarray,
) -> ConstraintCorrelationDiagnostics:
    """Build raw effective-penalty diagnostics without changing optimization."""

    names = _validated_cost_names(cost_names)
    costs = _finite_array(raw_costs, field_name="raw_costs", ndim=2)
    cost_advantages = _finite_array(
        raw_cost_advantages,
        field_name="raw_cost_advantages",
        ndim=2,
    )
    if costs.shape != cost_advantages.shape:
        raise ValueError("raw_costs and raw_cost_advantages must have the same shape")
    if costs.shape[1] != len(names):
        raise ValueError("cost diagnostic columns must match cost_names")

    reward = _finite_array(
        reward_advantages,
        field_name="reward_advantages",
        ndim=1,
    )
    if reward.shape[0] != costs.shape[0]:
        raise ValueError("reward_advantages must align with cost transitions")

    multiplier_vector = _finite_array(
        multipliers,
        field_name="multipliers",
        ndim=1,
    )
    if multiplier_vector.shape != (len(names),):
        raise ValueError("multipliers must match cost_names")
    if np.any(multiplier_vector < 0.0):
        raise ValueError("multipliers must be non-negative")

    normalized_correlation: np.ndarray | None = None
    if normalized_cost_advantages is not None:
        normalized = _finite_array(
            normalized_cost_advantages,
            field_name="normalized_cost_advantages",
            ndim=2,
        )
        if normalized.shape != cost_advantages.shape:
            raise ValueError(
                "normalized_cost_advantages must align with raw cost advantages"
            )
        _, normalized_correlation = _covariance_and_correlation(normalized)

    penalty_contributions = cost_advantages * multiplier_vector[None, :]
    aggregate_penalty = np.sum(penalty_contributions, axis=1)
    reward_l2 = float(np.linalg.norm(reward))
    penalty_l2 = float(np.linalg.norm(aggregate_penalty))
    ratio = penalty_l2 / max(reward_l2, _EPSILON)
    if not math.isfinite(ratio):
        raise ValueError("penalty_to_reward_l2_ratio became non-finite")

    raw_cost_covariance, raw_cost_correlation = _covariance_and_correlation(costs)
    return ConstraintCorrelationDiagnostics(
        cost_names=names,
        reward_advantages=_readonly(reward),
        raw_cost_advantages=_readonly(cost_advantages),
        penalty_contributions=_readonly(penalty_contributions),
        aggregate_penalty=_readonly(aggregate_penalty),
        raw_cost_covariance=raw_cost_covariance,
        raw_cost_correlation=raw_cost_correlation,
        normalized_cost_advantage_correlation=normalized_correlation,
        raw_reward_advantage_statistics=_vector_statistics(reward),
        raw_cost_advantage_statistics=tuple(
            _vector_statistics(cost_advantages[:, index]) for index in range(len(names))
        ),
        raw_effective_penalty_statistics=tuple(
            _vector_statistics(penalty_contributions[:, index])
            for index in range(len(names))
        ),
        penalty_to_reward_l2_ratio=ratio,
    )


def _longest_true_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _sign_change_frequency(values: Sequence[float]) -> float:
    signs = [int(np.sign(value)) for value in values if abs(value) > _EPSILON]
    if len(signs) < 2:
        return 0.0
    changes = sum(left != right for left, right in zip(signs, signs[1:]))
    return changes / (len(signs) - 1)


def build_dual_stability_diagnostics(
    *,
    cost_names: Sequence[str],
    report_history: Sequence[Mapping[str, DualUpdateReport]],
) -> DualStabilityDiagnostics:
    """Summarize dual history while counting only upper-cap saturation."""

    names = _validated_cost_names(cost_names)
    history = tuple(report_history)
    if not history:
        raise ValueError("report_history must not be empty")
    for reports in history:
        if not isinstance(reports, Mapping) or tuple(reports) != names:
            raise ValueError("report history constraint names do not match cost_names")
        for name in names:
            report = reports[name]
            if not isinstance(report, DualUpdateReport) or report.name != name:
                raise ValueError(
                    "report history constraint names do not match cost_names"
                )

    constraints: list[ConstraintStabilityDiagnostics] = []
    for name in names:
        constraint_reports = tuple(entry[name] for entry in history)
        upper_cap = tuple(report.at_upper_cap for report in constraint_reports)
        lower_bound = tuple(report.at_lower_bound for report in constraint_reports)
        multipliers = np.asarray(
            [report.multiplier_after for report in constraint_reports],
            dtype=np.float64,
        )
        if not np.isfinite(multipliers).all():
            raise ValueError("multiplier history must be finite")
        residuals = tuple(
            float(report.constraint_residual)
            for report in constraint_reports
            if report.constraint_residual is not None
        )
        if any(not math.isfinite(value) for value in residuals):
            raise ValueError("constraint residual history must be finite")
        deltas = tuple(
            report.multiplier_after - report.multiplier_before
            for report in constraint_reports
            if report.updated
        )
        satisfaction = tuple(value <= 0.0 for value in residuals)
        first_satisfied = next(
            (index for index, satisfied in enumerate(satisfaction) if satisfied),
            None,
        )
        post_satisfaction_overconstraint_count = (
            0
            if first_satisfied is None
            else sum(value > 0.0 for value in residuals[first_satisfied + 1 :])
        )
        constraints.append(
            ConstraintStabilityDiagnostics(
                name=name,
                rollout_count=len(constraint_reports),
                saturation_fraction=sum(upper_cap) / len(constraint_reports),
                lower_bound_fraction=sum(lower_bound) / len(constraint_reports),
                longest_saturation_run=_longest_true_run(upper_cap),
                update_sign_change_frequency=_sign_change_frequency(deltas),
                multiplier_variance=float(np.var(multipliers)),
                residual_variance=(
                    0.0
                    if not residuals
                    else float(np.var(np.asarray(residuals, dtype=np.float64)))
                ),
                violation_area=sum(max(value, 0.0) for value in residuals),
                longest_satisfaction_run=_longest_true_run(satisfaction),
                post_satisfaction_overconstraint_count=(
                    post_satisfaction_overconstraint_count
                ),
            )
        )
    return DualStabilityDiagnostics(
        cost_names=names,
        constraints=tuple(constraints),
    )


__all__ = [
    "ConstraintCorrelationDiagnostics",
    "ConstraintStabilityDiagnostics",
    "DualStabilityDiagnostics",
    "VectorStatistics",
    "build_constraint_correlation_diagnostics",
    "build_dual_stability_diagnostics",
]
