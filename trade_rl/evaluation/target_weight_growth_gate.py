"""Deterministic production gates for target-weight growth experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

from trade_rl.artifacts.hashing import content_digest

_TOLERANCE = 1e-12


def _finite(value: float, *, field: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _non_negative_integer(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _non_empty(value: str, *, field: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{field} must be non-empty")
    return resolved


@dataclass(frozen=True, slots=True)
class SoftConstraintEstimate:
    """One fold-seed estimate evaluated against a stable operational budget."""

    name: str
    observed_value: float
    budget: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, field="name"))
        observed = _finite(self.observed_value, field="observed_value")
        budget = _finite(self.budget, field="budget")
        if observed < 0.0 or budget < 0.0:
            raise ValueError("soft constraint values must be non-negative")
        object.__setattr__(self, "observed_value", observed)
        object.__setattr__(self, "budget", budget)

    def digest_payload(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "name": self.name,
            "observed_value": self.observed_value,
        }


@dataclass(frozen=True, slots=True)
class GrowthEvaluationCell:
    """One sealed fold-seed-scenario economic evaluation cell."""

    fold_index: int
    seed: int
    scenario: str
    selected_net_log_growth: float
    baseline_net_log_growth: float
    forced_liquidation_count: int = 0
    margin_deficit_count: int = 0
    insolvency_count: int = 0
    hard_safety_violation_count: int = 0
    soft_constraints: tuple[SoftConstraintEstimate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fold_index",
            _non_negative_integer(self.fold_index, field="fold_index"),
        )
        object.__setattr__(self, "seed", _non_negative_integer(self.seed, field="seed"))
        object.__setattr__(
            self,
            "scenario",
            _non_empty(self.scenario, field="scenario"),
        )
        object.__setattr__(
            self,
            "selected_net_log_growth",
            _finite(
                self.selected_net_log_growth,
                field="selected_net_log_growth",
            ),
        )
        object.__setattr__(
            self,
            "baseline_net_log_growth",
            _finite(
                self.baseline_net_log_growth,
                field="baseline_net_log_growth",
            ),
        )
        for field_name in (
            "forced_liquidation_count",
            "margin_deficit_count",
            "insolvency_count",
            "hard_safety_violation_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative_integer(getattr(self, field_name), field=field_name),
            )
        constraints = tuple(
            sorted(tuple(self.soft_constraints), key=lambda item: item.name)
        )
        names = tuple(item.name for item in constraints)
        if len(set(names)) != len(names):
            raise ValueError("soft constraint names must be unique")
        object.__setattr__(self, "soft_constraints", constraints)

    @property
    def paired_baseline_difference(self) -> float:
        return self.selected_net_log_growth - self.baseline_net_log_growth

    @property
    def catastrophic_event_count(self) -> int:
        return (
            self.forced_liquidation_count
            + self.margin_deficit_count
            + self.insolvency_count
            + self.hard_safety_violation_count
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "baseline_net_log_growth": self.baseline_net_log_growth,
            "fold_index": self.fold_index,
            "forced_liquidation_count": self.forced_liquidation_count,
            "hard_safety_violation_count": self.hard_safety_violation_count,
            "insolvency_count": self.insolvency_count,
            "margin_deficit_count": self.margin_deficit_count,
            "scenario": self.scenario,
            "seed": self.seed,
            "selected_net_log_growth": self.selected_net_log_growth,
            "soft_constraints": tuple(
                item.digest_payload() for item in self.soft_constraints
            ),
        }


@dataclass(frozen=True, slots=True)
class GrowthProductionGateConfig:
    """Stable statistical and operational thresholds for production admission."""

    required_fold_count: int = 6
    required_seeds: tuple[int, ...] = (0, 1, 2)
    required_positive_fold_count: int = 4
    required_positive_seed_count: int = 2
    nominal_scenario: str = "nominal"
    cost_2x_scenario: str = "joint_2x"
    cost_3x_scenario: str = "joint_3x"
    confidence: float = 0.95
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_fold_count, bool)
            or not isinstance(self.required_fold_count, int)
            or self.required_fold_count <= 0
        ):
            raise ValueError("required_fold_count must be a positive integer")
        seeds = tuple(self.required_seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("required_seeds must be non-empty and unique")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        ):
            raise ValueError("required_seeds must contain non-negative integers")
        object.__setattr__(self, "required_seeds", seeds)
        for field_name, value, upper in (
            (
                "required_positive_fold_count",
                self.required_positive_fold_count,
                self.required_fold_count,
            ),
            (
                "required_positive_seed_count",
                self.required_positive_seed_count,
                len(seeds),
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= upper
            ):
                raise ValueError(f"{field_name} must be within [0, {upper}]")
        scenarios = (
            _non_empty(self.nominal_scenario, field="nominal_scenario"),
            _non_empty(self.cost_2x_scenario, field="cost_2x_scenario"),
            _non_empty(self.cost_3x_scenario, field="cost_3x_scenario"),
        )
        if len(set(scenarios)) != len(scenarios):
            raise ValueError("required scenarios must be unique")
        object.__setattr__(self, "nominal_scenario", scenarios[0])
        object.__setattr__(self, "cost_2x_scenario", scenarios[1])
        object.__setattr__(self, "cost_3x_scenario", scenarios[2])
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be within (0.5, 1)")
        if (
            isinstance(self.bootstrap_samples, bool)
            or not isinstance(self.bootstrap_samples, int)
            or self.bootstrap_samples <= 0
        ):
            raise ValueError("bootstrap_samples must be a positive integer")
        _non_negative_integer(self.bootstrap_seed, field="bootstrap_seed")

    @property
    def required_scenarios(self) -> tuple[str, str, str]:
        return (
            self.nominal_scenario,
            self.cost_2x_scenario,
            self.cost_3x_scenario,
        )


@dataclass(frozen=True, slots=True)
class SoftConstraintGateSummary:
    name: str
    budget: float
    maximum_fold_estimate: float
    pooled_upper_bound: float
    passed: bool

    def digest_payload(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "maximum_fold_estimate": self.maximum_fold_estimate,
            "name": self.name,
            "passed": self.passed,
            "pooled_upper_bound": self.pooled_upper_bound,
        }


@dataclass(frozen=True, slots=True)
class GrowthProductionGateDecision:
    passed: bool
    reasons: tuple[str, ...]
    nominal_cell_count: int
    nominal_growth_median: float | None
    nominal_paired_median: float | None
    nominal_paired_lower_bound: float | None
    positive_fold_count: int
    nonnegative_seed_count: int
    positive_seed_count: int
    cost_2x_paired_median: float | None
    cost_3x_growth_median: float | None
    catastrophic_event_count: int
    identity_verified: bool
    soft_constraints: tuple[SoftConstraintGateSummary, ...]
    evidence_digest: str
    schema_version: str = "target_weight_growth_production_gate_v1"

    def digest_payload(self) -> dict[str, object]:
        return {
            "catastrophic_event_count": self.catastrophic_event_count,
            "cost_2x_paired_median": self.cost_2x_paired_median,
            "cost_3x_growth_median": self.cost_3x_growth_median,
            "evidence_digest": self.evidence_digest,
            "identity_verified": self.identity_verified,
            "nominal_cell_count": self.nominal_cell_count,
            "nominal_growth_median": self.nominal_growth_median,
            "nominal_paired_lower_bound": self.nominal_paired_lower_bound,
            "nominal_paired_median": self.nominal_paired_median,
            "nonnegative_seed_count": self.nonnegative_seed_count,
            "passed": self.passed,
            "positive_fold_count": self.positive_fold_count,
            "positive_seed_count": self.positive_seed_count,
            "reasons": self.reasons,
            "schema_version": self.schema_version,
            "soft_constraints": tuple(
                item.digest_payload() for item in self.soft_constraints
            ),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


@dataclass(frozen=True, slots=True)
class GrowthProfileComparisonCell:
    fold_index: int
    seed: int
    lagrangian_minus_ppo_net_log_growth: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fold_index",
            _non_negative_integer(self.fold_index, field="fold_index"),
        )
        object.__setattr__(self, "seed", _non_negative_integer(self.seed, field="seed"))
        object.__setattr__(
            self,
            "lagrangian_minus_ppo_net_log_growth",
            _finite(
                self.lagrangian_minus_ppo_net_log_growth,
                field="lagrangian_minus_ppo_net_log_growth",
            ),
        )


@dataclass(frozen=True, slots=True)
class GrowthProfileSelectionDecision:
    selected_profile: str | None
    reason: str
    lagrangian_minus_ppo_lower_bound: float | None
    lagrangian_minus_ppo_upper_bound: float | None
    comparison_evidence_digest: str | None
    schema_version: str = "target_weight_growth_profile_selection_v1"

    def digest_payload(self) -> dict[str, object]:
        return {
            "comparison_evidence_digest": self.comparison_evidence_digest,
            "lagrangian_minus_ppo_lower_bound": (self.lagrangian_minus_ppo_lower_bound),
            "lagrangian_minus_ppo_upper_bound": (self.lagrangian_minus_ppo_upper_bound),
            "reason": self.reason,
            "schema_version": self.schema_version,
            "selected_profile": self.selected_profile,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


def _group_values_by_fold(
    cells: tuple[GrowthEvaluationCell, ...],
    value_getter: object,
) -> dict[int, tuple[float, ...]]:
    getter = value_getter
    grouped: dict[int, list[float]] = {}
    for cell in cells:
        value = getter(cell)  # type: ignore[operator]
        grouped.setdefault(cell.fold_index, []).append(float(value))
    return {fold: tuple(values) for fold, values in grouped.items()}


def _cluster_bootstrap_bounds(
    values_by_fold: dict[int, tuple[float, ...]],
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    folds = tuple(sorted(values_by_fold))
    if not folds or any(not values_by_fold[fold] for fold in folds):
        raise ValueError("cluster bootstrap requires non-empty fold values")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_folds = rng.choice(folds, size=len(folds), replace=True)
        sample_values = tuple(
            value for fold in sampled_folds for value in values_by_fold[int(fold)]
        )
        estimates[sample_index] = float(np.mean(sample_values))
    two_sided_alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, two_sided_alpha)),
        float(np.quantile(estimates, 1.0 - two_sided_alpha)),
        float(np.quantile(estimates, confidence)),
    )


def _median_or_none(values: tuple[float, ...]) -> float | None:
    return None if not values else float(median(values))


def _support_reasons(
    cells: tuple[GrowthEvaluationCell, ...],
    config: GrowthProductionGateConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    keys = tuple((cell.fold_index, cell.seed, cell.scenario) for cell in cells)
    if len(set(keys)) != len(keys):
        reasons.append("duplicate_evaluation_cell")
    folds = tuple(sorted({cell.fold_index for cell in cells}))
    if len(folds) != config.required_fold_count:
        reasons.append(
            f"fold_count_mismatch:observed={len(folds)}:required={config.required_fold_count}"
        )
    required_seed_set = set(config.required_seeds)
    for scenario in config.required_scenarios:
        scenario_cells = tuple(cell for cell in cells if cell.scenario == scenario)
        for fold in folds:
            observed = {cell.seed for cell in scenario_cells if cell.fold_index == fold}
            if observed != required_seed_set:
                reasons.append(
                    "support_mismatch:"
                    f"scenario={scenario}:fold={fold}:"
                    f"observed={tuple(sorted(observed))}:"
                    f"required={config.required_seeds}"
                )
    return tuple(sorted(set(reasons)))


def _soft_constraint_summaries(
    nominal: tuple[GrowthEvaluationCell, ...],
    *,
    config: GrowthProductionGateConfig,
) -> tuple[tuple[SoftConstraintGateSummary, ...], tuple[str, ...]]:
    reasons: list[str] = []
    if not nominal:
        return (), ("soft_constraint_evidence_missing",)
    expected_names = tuple(item.name for item in nominal[0].soft_constraints)
    if not expected_names:
        return (), ("soft_constraint_evidence_missing",)
    budgets = {item.name: item.budget for item in nominal[0].soft_constraints}
    for cell in nominal:
        names = tuple(item.name for item in cell.soft_constraints)
        if names != expected_names:
            reasons.append("soft_constraint_schema_mismatch")
            continue
        for item in cell.soft_constraints:
            if not math.isclose(
                item.budget,
                budgets[item.name],
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            ):
                reasons.append(f"soft_constraint_budget_mismatch:{item.name}")
    summaries: list[SoftConstraintGateSummary] = []
    if reasons:
        return (), tuple(sorted(set(reasons)))
    for name in expected_names:
        observations = {
            cell.fold_index: next(
                item.observed_value
                for item in cell.soft_constraints
                if item.name == name
            )
            for cell in nominal
        }
        by_fold: dict[int, tuple[float, ...]] = {}
        for fold in sorted({cell.fold_index for cell in nominal}):
            by_fold[fold] = tuple(
                next(
                    item.observed_value
                    for item in cell.soft_constraints
                    if item.name == name
                )
                for cell in nominal
                if cell.fold_index == fold
            )
        maximum_fold = max(float(np.mean(values)) for values in by_fold.values())
        _, _, upper = _cluster_bootstrap_bounds(
            by_fold,
            confidence=config.confidence,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed,
        )
        budget = budgets[name]
        passed = maximum_fold <= budget + _TOLERANCE and upper <= budget + _TOLERANCE
        summaries.append(
            SoftConstraintGateSummary(
                name=name,
                budget=budget,
                maximum_fold_estimate=maximum_fold,
                pooled_upper_bound=upper,
                passed=passed,
            )
        )
        if not passed:
            reasons.append(f"soft_constraint_budget_failed:{name}")
        del observations
    return tuple(summaries), tuple(sorted(set(reasons)))


def evaluate_target_weight_growth_gate(
    cells: tuple[GrowthEvaluationCell, ...],
    *,
    identity_verified: bool,
    config: GrowthProductionGateConfig | None = None,
) -> GrowthProductionGateDecision:
    """Evaluate the approved six-fold, three-seed production contract."""

    resolved_config = config or GrowthProductionGateConfig()
    normalized = tuple(
        sorted(
            tuple(cells),
            key=lambda item: (item.scenario, item.fold_index, item.seed),
        )
    )
    reasons = list(_support_reasons(normalized, resolved_config))
    if not isinstance(identity_verified, bool) or not identity_verified:
        reasons.append("identity_not_verified")

    nominal = tuple(
        cell for cell in normalized if cell.scenario == resolved_config.nominal_scenario
    )
    cost_2x = tuple(
        cell for cell in normalized if cell.scenario == resolved_config.cost_2x_scenario
    )
    cost_3x = tuple(
        cell for cell in normalized if cell.scenario == resolved_config.cost_3x_scenario
    )

    nominal_growth = tuple(cell.selected_net_log_growth for cell in nominal)
    nominal_paired = tuple(cell.paired_baseline_difference for cell in nominal)
    cost_2x_paired = tuple(cell.paired_baseline_difference for cell in cost_2x)
    cost_3x_growth = tuple(cell.selected_net_log_growth for cell in cost_3x)

    nominal_growth_median = _median_or_none(nominal_growth)
    nominal_paired_median = _median_or_none(nominal_paired)
    cost_2x_paired_median = _median_or_none(cost_2x_paired)
    cost_3x_growth_median = _median_or_none(cost_3x_growth)

    nominal_paired_lower_bound: float | None = None
    if nominal:
        lower, _, _ = _cluster_bootstrap_bounds(
            _group_values_by_fold(
                nominal,
                lambda cell: cell.paired_baseline_difference,
            ),
            confidence=resolved_config.confidence,
            samples=resolved_config.bootstrap_samples,
            seed=resolved_config.bootstrap_seed,
        )
        nominal_paired_lower_bound = lower

    if nominal_growth_median is None or nominal_growth_median <= 0.0:
        reasons.append("nominal_growth_median_not_positive")
    if nominal_paired_median is None or nominal_paired_median <= 0.0:
        reasons.append("nominal_paired_median_not_positive")
    if nominal_paired_lower_bound is None or nominal_paired_lower_bound <= 0.0:
        reasons.append("nominal_paired_lower_bound_not_positive")
    if cost_2x_paired_median is None or cost_2x_paired_median <= 0.0:
        reasons.append("cost_2x_paired_median_not_positive")
    if cost_3x_growth_median is None or cost_3x_growth_median < 0.0:
        reasons.append("cost_3x_growth_median_negative")

    fold_medians = tuple(
        float(
            median(
                cell.paired_baseline_difference
                for cell in nominal
                if cell.fold_index == fold
            )
        )
        for fold in sorted({cell.fold_index for cell in nominal})
        if any(cell.fold_index == fold for cell in nominal)
    )
    positive_fold_count = sum(value > 0.0 for value in fold_medians)
    if positive_fold_count < resolved_config.required_positive_fold_count:
        reasons.append("insufficient_positive_folds")

    seed_medians = tuple(
        float(
            median(
                cell.selected_net_log_growth for cell in nominal if cell.seed == seed
            )
        )
        for seed in resolved_config.required_seeds
        if any(cell.seed == seed for cell in nominal)
    )
    nonnegative_seed_count = sum(value >= 0.0 for value in seed_medians)
    positive_seed_count = sum(value > 0.0 for value in seed_medians)
    if nonnegative_seed_count != len(resolved_config.required_seeds):
        reasons.append("seed_median_negative")
    if positive_seed_count < resolved_config.required_positive_seed_count:
        reasons.append("insufficient_positive_seeds")

    catastrophic_event_count = sum(cell.catastrophic_event_count for cell in normalized)
    if catastrophic_event_count:
        reasons.append("catastrophic_event_detected")

    soft_constraints, soft_reasons = _soft_constraint_summaries(
        nominal,
        config=resolved_config,
    )
    reasons.extend(soft_reasons)
    canonical_reasons = tuple(sorted(set(reasons)))
    evidence_digest = content_digest(
        {
            "cells": tuple(cell.digest_payload() for cell in normalized),
            "config": {
                "bootstrap_samples": resolved_config.bootstrap_samples,
                "bootstrap_seed": resolved_config.bootstrap_seed,
                "confidence": resolved_config.confidence,
                "cost_2x_scenario": resolved_config.cost_2x_scenario,
                "cost_3x_scenario": resolved_config.cost_3x_scenario,
                "nominal_scenario": resolved_config.nominal_scenario,
                "required_fold_count": resolved_config.required_fold_count,
                "required_positive_fold_count": (
                    resolved_config.required_positive_fold_count
                ),
                "required_positive_seed_count": (
                    resolved_config.required_positive_seed_count
                ),
                "required_seeds": resolved_config.required_seeds,
            },
            "identity_verified": identity_verified,
        }
    )
    return GrowthProductionGateDecision(
        passed=not canonical_reasons,
        reasons=canonical_reasons,
        nominal_cell_count=len(nominal),
        nominal_growth_median=nominal_growth_median,
        nominal_paired_median=nominal_paired_median,
        nominal_paired_lower_bound=nominal_paired_lower_bound,
        positive_fold_count=positive_fold_count,
        nonnegative_seed_count=nonnegative_seed_count,
        positive_seed_count=positive_seed_count,
        cost_2x_paired_median=cost_2x_paired_median,
        cost_3x_growth_median=cost_3x_growth_median,
        catastrophic_event_count=catastrophic_event_count,
        identity_verified=identity_verified,
        soft_constraints=soft_constraints,
        evidence_digest=evidence_digest,
    )


def _comparison_support_reasons(
    comparisons: tuple[GrowthProfileComparisonCell, ...],
    config: GrowthProductionGateConfig,
) -> tuple[str, ...]:
    keys = tuple((cell.fold_index, cell.seed) for cell in comparisons)
    reasons: list[str] = []
    if len(set(keys)) != len(keys):
        reasons.append("duplicate_comparison_cell")
    folds = tuple(sorted({cell.fold_index for cell in comparisons}))
    if len(folds) != config.required_fold_count:
        reasons.append("comparison_fold_count_mismatch")
    for fold in folds:
        seeds = {cell.seed for cell in comparisons if cell.fold_index == fold}
        if seeds != set(config.required_seeds):
            reasons.append(f"comparison_support_mismatch:fold={fold}")
    return tuple(sorted(set(reasons)))


def select_target_weight_growth_profile(
    *,
    ppo: GrowthProductionGateDecision,
    lagrangian: GrowthProductionGateDecision,
    comparisons: tuple[GrowthProfileComparisonCell, ...],
    config: GrowthProductionGateConfig | None = None,
) -> GrowthProfileSelectionDecision:
    """Select Lagrangian only when it has statistically positive growth uplift."""

    resolved_config = config or GrowthProductionGateConfig()
    if ppo.passed and not lagrangian.passed:
        return GrowthProfileSelectionDecision(
            selected_profile="g1_ppo",
            reason="only_ppo_passed_production_gate",
            lagrangian_minus_ppo_lower_bound=None,
            lagrangian_minus_ppo_upper_bound=None,
            comparison_evidence_digest=None,
        )
    if lagrangian.passed and not ppo.passed:
        return GrowthProfileSelectionDecision(
            selected_profile="g1_lagrangian",
            reason="only_lagrangian_passed_production_gate",
            lagrangian_minus_ppo_lower_bound=None,
            lagrangian_minus_ppo_upper_bound=None,
            comparison_evidence_digest=None,
        )
    if not ppo.passed and not lagrangian.passed:
        return GrowthProfileSelectionDecision(
            selected_profile=None,
            reason="no_profile_passed_production_gate",
            lagrangian_minus_ppo_lower_bound=None,
            lagrangian_minus_ppo_upper_bound=None,
            comparison_evidence_digest=None,
        )

    normalized = tuple(
        sorted(tuple(comparisons), key=lambda item: (item.fold_index, item.seed))
    )
    support_reasons = _comparison_support_reasons(normalized, resolved_config)
    evidence_digest = content_digest(
        {
            "cells": tuple(
                {
                    "fold_index": cell.fold_index,
                    "lagrangian_minus_ppo_net_log_growth": (
                        cell.lagrangian_minus_ppo_net_log_growth
                    ),
                    "seed": cell.seed,
                }
                for cell in normalized
            ),
            "config": {
                "bootstrap_samples": resolved_config.bootstrap_samples,
                "bootstrap_seed": resolved_config.bootstrap_seed,
                "confidence": resolved_config.confidence,
                "required_fold_count": resolved_config.required_fold_count,
                "required_seeds": resolved_config.required_seeds,
            },
            "lagrangian_gate_digest": lagrangian.digest,
            "ppo_gate_digest": ppo.digest,
        }
    )
    if support_reasons:
        return GrowthProfileSelectionDecision(
            selected_profile=None,
            reason="comparison_evidence_invalid",
            lagrangian_minus_ppo_lower_bound=None,
            lagrangian_minus_ppo_upper_bound=None,
            comparison_evidence_digest=evidence_digest,
        )

    values_by_fold: dict[int, tuple[float, ...]] = {
        fold: tuple(
            cell.lagrangian_minus_ppo_net_log_growth
            for cell in normalized
            if cell.fold_index == fold
        )
        for fold in sorted({cell.fold_index for cell in normalized})
    }
    lower, upper, _ = _cluster_bootstrap_bounds(
        values_by_fold,
        confidence=resolved_config.confidence,
        samples=resolved_config.bootstrap_samples,
        seed=resolved_config.bootstrap_seed,
    )
    if lower > 0.0:
        selected = "g1_lagrangian"
        reason = "lagrangian_growth_significantly_positive"
    else:
        selected = "g1_ppo"
        reason = "growth_difference_not_significantly_positive"
    return GrowthProfileSelectionDecision(
        selected_profile=selected,
        reason=reason,
        lagrangian_minus_ppo_lower_bound=lower,
        lagrangian_minus_ppo_upper_bound=upper,
        comparison_evidence_digest=evidence_digest,
    )


__all__ = [
    "GrowthEvaluationCell",
    "GrowthProductionGateConfig",
    "GrowthProductionGateDecision",
    "GrowthProfileComparisonCell",
    "GrowthProfileSelectionDecision",
    "SoftConstraintEstimate",
    "SoftConstraintGateSummary",
    "evaluate_target_weight_growth_gate",
    "select_target_weight_growth_profile",
]
