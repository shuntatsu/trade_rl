"""Fold-local and aggregate C3 evidence reports."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.bootstrap import moving_block_mean_test
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    CausalScenarioQueryComparison,
    PerfectInformationComparisonStatus,
)

C3_FOLD_REPORT_SCHEMA: Final = "causal_scenario_c3_fold_report_v1"
C3_AGGREGATE_REPORT_SCHEMA: Final = "causal_scenario_c3_aggregate_report_v1"


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _readonly_vector(name: str, value: object) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).copy(order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array[array == 0.0] = 0.0
    array.setflags(write=False)
    return array


def _readonly_day_indices(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError("day_indices must be integers")
    array = np.asarray(raw, dtype=np.int64).copy(order="C")
    if array.ndim != 1 or array.size == 0:
        raise ValueError("day_indices must be a non-empty vector")
    if np.any(array < 0) or np.any(np.diff(array) <= 0):
        raise ValueError("day_indices must be strictly increasing and non-negative")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CausalScenarioFoldReport:
    fold_id: str
    selection_days: int
    effective_days: int
    day_indices: np.ndarray
    comparisons: tuple[CausalScenarioQueryComparison, ...]
    uplift: np.ndarray
    spearman: np.ndarray
    regret_margin: np.ndarray
    scenario_oracle_max_drawdown: float
    trend_max_drawdown: float
    required_adverse_passed: bool
    perfect_information_valid: bool
    failure_reasons: tuple[str, ...]
    schema_version: str = C3_FOLD_REPORT_SCHEMA

    def __post_init__(self) -> None:
        fold_id = self.fold_id.strip()
        if not fold_id:
            raise ValueError("fold_id must be non-empty")
        object.__setattr__(self, "fold_id", fold_id)
        selection_days = _positive_int("selection_days", self.selection_days)
        effective_days = _positive_int("effective_days", self.effective_days)
        if effective_days > selection_days:
            raise ValueError("effective_days must not exceed selection_days")
        object.__setattr__(self, "selection_days", selection_days)
        object.__setattr__(self, "effective_days", effective_days)
        comparisons = tuple(self.comparisons)
        if not comparisons or any(
            not isinstance(item, CausalScenarioQueryComparison) for item in comparisons
        ):
            raise ValueError("comparisons must contain C3 query comparisons")
        if len({item.decision_digest for item in comparisons}) != len(comparisons):
            raise ValueError("comparison decision digests must be unique within a fold")
        object.__setattr__(self, "comparisons", comparisons)
        days = _readonly_day_indices(self.day_indices)
        uplift = _readonly_vector("uplift", self.uplift)
        spearman = _readonly_vector("spearman", self.spearman)
        regret_margin = _readonly_vector("regret_margin", self.regret_margin)
        if not (
            len(days)
            == effective_days
            == len(uplift)
            == len(spearman)
            == len(regret_margin)
        ):
            raise ValueError("daily fold metric vectors must match effective_days")
        if tuple(sorted({item.day_index for item in comparisons})) != tuple(days):
            raise ValueError("day_indices do not match query comparisons")
        object.__setattr__(self, "day_indices", days)
        object.__setattr__(self, "uplift", uplift)
        object.__setattr__(self, "spearman", spearman)
        object.__setattr__(self, "regret_margin", regret_margin)
        for field in ("scenario_oracle_max_drawdown", "trend_max_drawdown"):
            value = _finite_float(field, getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, value)
        if not isinstance(self.required_adverse_passed, bool):
            raise ValueError("required_adverse_passed must be boolean")
        if not isinstance(self.perfect_information_valid, bool):
            raise ValueError("perfect_information_valid must be boolean")
        reasons = tuple(reason.strip() for reason in self.failure_reasons)
        if any(not reason for reason in reasons) or len(set(reasons)) != len(reasons):
            raise ValueError("failure_reasons must be unique and non-empty")
        object.__setattr__(self, "failure_reasons", reasons)
        if self.schema_version != C3_FOLD_REPORT_SCHEMA:
            raise ValueError("unsupported C3 fold report schema")

    @property
    def mean_uplift(self) -> float:
        return float(self.uplift.mean())

    @property
    def mean_spearman(self) -> float:
        return float(self.spearman.mean())

    @property
    def mean_regret_margin(self) -> float:
        return float(self.regret_margin.mean())

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "comparison_digests": tuple(item.digest for item in self.comparisons),
                "day_indices": self.day_indices.tolist(),
                "effective_days": self.effective_days,
                "failure_reasons": self.failure_reasons,
                "fold_id": self.fold_id,
                "perfect_information_valid": self.perfect_information_valid,
                "regret_margin": self.regret_margin.tolist(),
                "required_adverse_passed": self.required_adverse_passed,
                "scenario_oracle_max_drawdown": self.scenario_oracle_max_drawdown,
                "schema_version": self.schema_version,
                "selection_days": self.selection_days,
                "spearman": self.spearman.tolist(),
                "trend_max_drawdown": self.trend_max_drawdown,
                "uplift": self.uplift.tolist(),
            }
        )


@dataclass(frozen=True, slots=True)
class CausalScenarioAggregateReport:
    folds: tuple[CausalScenarioFoldReport, ...]
    total_selection_days: int
    total_effective_days: int
    positive_uplift_folds: int
    mean_uplift: float
    uplift_lower_ci: float
    uplift_upper_ci: float
    uplift_p_value: float
    mean_spearman: float
    spearman_lower_ci: float
    spearman_upper_ci: float
    mean_regret_margin: float
    regret_margin_lower_ci: float
    regret_margin_upper_ci: float
    worst_scenario_oracle_drawdown: float
    worst_trend_drawdown: float
    all_required_adverse_passed: bool
    all_perfect_information_valid: bool
    failure_reasons: tuple[str, ...]
    bootstrap_resamples: int
    bootstrap_block_days: int
    schema_version: str = C3_AGGREGATE_REPORT_SCHEMA

    def __post_init__(self) -> None:
        folds = tuple(self.folds)
        if not folds or any(
            not isinstance(item, CausalScenarioFoldReport) for item in folds
        ):
            raise ValueError("folds must contain C3 fold reports")
        if len({item.fold_id for item in folds}) != len(folds):
            raise ValueError("fold IDs must be unique")
        object.__setattr__(self, "folds", folds)
        object.__setattr__(
            self,
            "total_selection_days",
            _positive_int("total_selection_days", self.total_selection_days),
        )
        object.__setattr__(
            self,
            "total_effective_days",
            _positive_int("total_effective_days", self.total_effective_days),
        )
        if self.total_effective_days > self.total_selection_days:
            raise ValueError("total_effective_days exceeds total_selection_days")
        positive = _non_negative_int(
            "positive_uplift_folds", self.positive_uplift_folds
        )
        if positive > len(folds):
            raise ValueError("positive_uplift_folds exceeds fold count")
        object.__setattr__(self, "positive_uplift_folds", positive)
        object.__setattr__(
            self,
            "bootstrap_resamples",
            _positive_int("bootstrap_resamples", self.bootstrap_resamples),
        )
        object.__setattr__(
            self,
            "bootstrap_block_days",
            _positive_int("bootstrap_block_days", self.bootstrap_block_days),
        )
        for field in (
            "mean_uplift",
            "uplift_lower_ci",
            "uplift_upper_ci",
            "uplift_p_value",
            "mean_spearman",
            "spearman_lower_ci",
            "spearman_upper_ci",
            "mean_regret_margin",
            "regret_margin_lower_ci",
            "regret_margin_upper_ci",
            "worst_scenario_oracle_drawdown",
            "worst_trend_drawdown",
        ):
            object.__setattr__(self, field, _finite_float(field, getattr(self, field)))
        if not 0.0 <= self.uplift_p_value <= 1.0:
            raise ValueError("uplift_p_value must be in [0, 1]")
        for field in ("worst_scenario_oracle_drawdown", "worst_trend_drawdown"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
        if not isinstance(self.all_required_adverse_passed, bool):
            raise ValueError("all_required_adverse_passed must be boolean")
        if not isinstance(self.all_perfect_information_valid, bool):
            raise ValueError("all_perfect_information_valid must be boolean")
        reasons = tuple(reason.strip() for reason in self.failure_reasons)
        if any(not reason for reason in reasons) or len(set(reasons)) != len(reasons):
            raise ValueError("failure_reasons must be unique and non-empty")
        object.__setattr__(self, "failure_reasons", reasons)
        if self.schema_version != C3_AGGREGATE_REPORT_SCHEMA:
            raise ValueError("unsupported C3 aggregate report schema")

    @property
    def fold_count(self) -> int:
        return len(self.folds)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "all_perfect_information_valid": self.all_perfect_information_valid,
                "all_required_adverse_passed": self.all_required_adverse_passed,
                "bootstrap_block_days": self.bootstrap_block_days,
                "bootstrap_resamples": self.bootstrap_resamples,
                "failure_reasons": self.failure_reasons,
                "fold_digests": tuple(item.digest for item in self.folds),
                "mean_regret_margin": self.mean_regret_margin,
                "mean_spearman": self.mean_spearman,
                "mean_uplift": self.mean_uplift,
                "positive_uplift_folds": self.positive_uplift_folds,
                "regret_margin_lower_ci": self.regret_margin_lower_ci,
                "regret_margin_upper_ci": self.regret_margin_upper_ci,
                "schema_version": self.schema_version,
                "spearman_lower_ci": self.spearman_lower_ci,
                "spearman_upper_ci": self.spearman_upper_ci,
                "total_effective_days": self.total_effective_days,
                "total_selection_days": self.total_selection_days,
                "uplift_lower_ci": self.uplift_lower_ci,
                "uplift_p_value": self.uplift_p_value,
                "uplift_upper_ci": self.uplift_upper_ci,
                "worst_scenario_oracle_drawdown": self.worst_scenario_oracle_drawdown,
                "worst_trend_drawdown": self.worst_trend_drawdown,
            }
        )


def build_c3_fold_report(
    *,
    fold_id: str,
    selection_days: int,
    comparisons: tuple[CausalScenarioQueryComparison, ...],
    required_adverse_passed: bool,
    failure_reasons: tuple[str, ...] = (),
) -> CausalScenarioFoldReport:
    items = tuple(comparisons)
    if not items:
        raise ValueError("comparisons must not be empty")
    grouped: dict[int, list[CausalScenarioQueryComparison]] = defaultdict(list)
    for item in items:
        grouped[item.day_index].append(item)
    day_indices = np.asarray(sorted(grouped), dtype=np.int64)
    uplift = np.asarray(
        [
            sum(
                item.scenario_oracle.gross_log_return - item.trend.gross_log_return
                for item in grouped[int(day)]
            )
            for day in day_indices
        ],
        dtype=np.float64,
    )
    spearman = np.asarray(
        [
            float(
                np.mean(
                    [item.predicted_realized_spearman for item in grouped[int(day)]]
                )
            )
            for day in day_indices
        ],
        dtype=np.float64,
    )
    regret_margin = np.asarray(
        [
            float(
                np.mean(
                    [
                        item.random_realized_regret - item.selected_realized_regret
                        for item in grouped[int(day)]
                    ]
                )
            )
            for day in day_indices
        ],
        dtype=np.float64,
    )
    perfect_valid = all(
        item.perfect_information.status is PerfectInformationComparisonStatus.COMPARABLE
        and item.perfect_information.bound_log_return is not None
        and item.perfect_information.causal_log_return is not None
        and item.perfect_information.bound_log_return
        >= item.perfect_information.causal_log_return - 1e-12
        for item in items
    )
    return CausalScenarioFoldReport(
        fold_id=fold_id,
        selection_days=selection_days,
        effective_days=len(day_indices),
        day_indices=day_indices,
        comparisons=items,
        uplift=uplift,
        spearman=spearman,
        regret_margin=regret_margin,
        scenario_oracle_max_drawdown=max(
            item.scenario_oracle.max_drawdown for item in items
        ),
        trend_max_drawdown=max(item.trend.max_drawdown for item in items),
        required_adverse_passed=required_adverse_passed,
        perfect_information_valid=perfect_valid,
        failure_reasons=failure_reasons,
    )


def _bootstrap(
    values: np.ndarray,
    *,
    resamples: int,
    block_days: int,
    seed_payload: dict[str, object],
) -> tuple[float, float, float]:
    seed = int(content_digest(seed_payload)[:8], 16)
    result = moving_block_mean_test(
        tuple(float(value) for value in values),
        n_bootstrap=resamples,
        seed=seed,
        block_size=block_days,
    )
    return result.lower_ci, result.upper_ci, result.p_value


def build_c3_aggregate_report(
    folds: tuple[CausalScenarioFoldReport, ...],
    *,
    bootstrap_resamples: int = 1_000,
    bootstrap_block_days: int = 7,
) -> CausalScenarioAggregateReport:
    items = tuple(folds)
    if not items:
        raise ValueError("folds must not be empty")
    resamples = _positive_int("bootstrap_resamples", bootstrap_resamples)
    block_days = _positive_int("bootstrap_block_days", bootstrap_block_days)
    uplift = np.concatenate([item.uplift for item in items])
    spearman = np.concatenate([item.spearman for item in items])
    regret_margin = np.concatenate([item.regret_margin for item in items])
    fold_digests = tuple(item.digest for item in items)
    common_seed = {
        "bootstrap_block_days": block_days,
        "fold_digests": fold_digests,
        "schema_version": C3_AGGREGATE_REPORT_SCHEMA,
    }
    uplift_lower, uplift_upper, uplift_p = _bootstrap(
        uplift,
        resamples=resamples,
        block_days=block_days,
        seed_payload={**common_seed, "metric": "uplift"},
    )
    spearman_lower, spearman_upper, _ = _bootstrap(
        spearman,
        resamples=resamples,
        block_days=block_days,
        seed_payload={**common_seed, "metric": "spearman"},
    )
    regret_lower, regret_upper, _ = _bootstrap(
        regret_margin,
        resamples=resamples,
        block_days=block_days,
        seed_payload={**common_seed, "metric": "regret_margin"},
    )
    failures = tuple(
        f"{item.fold_id}:{reason}" for item in items for reason in item.failure_reasons
    )
    return CausalScenarioAggregateReport(
        folds=items,
        total_selection_days=sum(item.selection_days for item in items),
        total_effective_days=sum(item.effective_days for item in items),
        positive_uplift_folds=sum(item.mean_uplift > 0.0 for item in items),
        mean_uplift=float(uplift.mean()),
        uplift_lower_ci=uplift_lower,
        uplift_upper_ci=uplift_upper,
        uplift_p_value=uplift_p,
        mean_spearman=float(spearman.mean()),
        spearman_lower_ci=spearman_lower,
        spearman_upper_ci=spearman_upper,
        mean_regret_margin=float(regret_margin.mean()),
        regret_margin_lower_ci=regret_lower,
        regret_margin_upper_ci=regret_upper,
        worst_scenario_oracle_drawdown=max(
            item.scenario_oracle_max_drawdown for item in items
        ),
        worst_trend_drawdown=max(item.trend_max_drawdown for item in items),
        all_required_adverse_passed=all(item.required_adverse_passed for item in items),
        all_perfect_information_valid=all(
            item.perfect_information_valid for item in items
        ),
        failure_reasons=failures,
        bootstrap_resamples=resamples,
        bootstrap_block_days=block_days,
    )
