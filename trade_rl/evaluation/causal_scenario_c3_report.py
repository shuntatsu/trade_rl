"""Fold-local and aggregate C3 evidence reports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.bootstrap import moving_block_mean_test
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    CausalScenarioQueryComparison,
    PerfectInformationComparisonStatus,
    RealizedPolicyOutcome,
)

C3_FOLD_REPORT_SCHEMA: Final = "causal_scenario_c3_fold_report_v1"
C3_AGGREGATE_REPORT_SCHEMA: Final = "causal_scenario_c3_aggregate_report_v1"
C3_CALIBRATION_BUCKET_SCHEMA: Final = "causal_scenario_c3_calibration_bucket_v1"
C3_EXECUTION_SUMMARY_SCHEMA: Final = "causal_scenario_c3_execution_summary_v1"


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
class C3CalibrationBucket:
    bucket_index: int
    sample_count: int
    minimum_score: float
    maximum_score: float
    predicted_mean_advantage: float
    predicted_loss_cvar: float
    realized_mean_advantage: float
    realized_downside_mean: float
    schema_version: str = C3_CALIBRATION_BUCKET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bucket_index", _non_negative_int("bucket_index", self.bucket_index)
        )
        object.__setattr__(
            self, "sample_count", _positive_int("sample_count", self.sample_count)
        )
        for field in (
            "minimum_score",
            "maximum_score",
            "predicted_mean_advantage",
            "predicted_loss_cvar",
            "realized_mean_advantage",
            "realized_downside_mean",
        ):
            object.__setattr__(self, field, _finite_float(field, getattr(self, field)))
        if self.minimum_score > self.maximum_score:
            raise ValueError("calibration bucket score range is invalid")
        if self.predicted_loss_cvar < 0.0:
            raise ValueError("predicted_loss_cvar must be non-negative")
        if self.realized_downside_mean > 0.0:
            raise ValueError("realized_downside_mean must be non-positive")
        if self.schema_version != C3_CALIBRATION_BUCKET_SCHEMA:
            raise ValueError("unsupported C3 calibration bucket schema")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "bucket_index": self.bucket_index,
                "maximum_score": self.maximum_score,
                "minimum_score": self.minimum_score,
                "predicted_loss_cvar": self.predicted_loss_cvar,
                "predicted_mean_advantage": self.predicted_mean_advantage,
                "realized_downside_mean": self.realized_downside_mean,
                "realized_mean_advantage": self.realized_mean_advantage,
                "sample_count": self.sample_count,
                "schema_version": self.schema_version,
            }
        )


@dataclass(frozen=True, slots=True)
class C3PolicyExecutionSummary:
    execution_scenario: str
    policy_kind: str
    observation_count: int
    mean_gross_log_return: float
    mean_filled_turnover: float
    mean_fees: float
    mean_spread_cost: float
    mean_impact_cost: float
    mean_funding_paid: float
    mean_borrow_paid: float
    mean_total_economic_cost: float
    mean_fill_ratio: float
    total_fill_count: int
    total_pending_order_events: int
    total_cancel_replace_events: int
    maximum_drawdown: float
    termination_distribution: tuple[tuple[str, int], ...]
    schema_version: str = C3_EXECUTION_SUMMARY_SCHEMA

    def __post_init__(self) -> None:
        scenario = self.execution_scenario.strip()
        policy = self.policy_kind.strip()
        if not scenario or not policy:
            raise ValueError("execution summary scenario and policy must be non-empty")
        object.__setattr__(self, "execution_scenario", scenario)
        object.__setattr__(self, "policy_kind", policy)
        object.__setattr__(
            self,
            "observation_count",
            _positive_int("observation_count", self.observation_count),
        )
        for field in (
            "mean_gross_log_return",
            "mean_filled_turnover",
            "mean_fees",
            "mean_spread_cost",
            "mean_impact_cost",
            "mean_funding_paid",
            "mean_borrow_paid",
            "mean_total_economic_cost",
            "mean_fill_ratio",
            "maximum_drawdown",
        ):
            object.__setattr__(self, field, _finite_float(field, getattr(self, field)))
        for field in (
            "mean_filled_turnover",
            "mean_fees",
            "mean_spread_cost",
            "mean_impact_cost",
            "mean_funding_paid",
            "mean_borrow_paid",
            "mean_total_economic_cost",
        ):
            if getattr(self, field) < 0.0:
                raise ValueError(f"{field} must be non-negative")
        if not 0.0 <= self.mean_fill_ratio <= 1.0:
            raise ValueError("mean_fill_ratio must be in [0, 1]")
        if not 0.0 <= self.maximum_drawdown <= 1.0:
            raise ValueError("maximum_drawdown must be in [0, 1]")
        for field in (
            "total_fill_count",
            "total_pending_order_events",
            "total_cancel_replace_events",
        ):
            object.__setattr__(
                self,
                field,
                _non_negative_int(field, getattr(self, field)),
            )
        distribution = tuple(
            (reason.strip(), _positive_int("termination_count", count))
            for reason, count in self.termination_distribution
        )
        if any(not reason for reason, _ in distribution):
            raise ValueError("termination reasons must be non-empty")
        if tuple(sorted(distribution)) != distribution:
            raise ValueError("termination_distribution must be sorted")
        if len({reason for reason, _ in distribution}) != len(distribution):
            raise ValueError("termination reasons must be unique")
        if sum(count for _, count in distribution) != self.observation_count:
            raise ValueError("termination distribution count mismatch")
        object.__setattr__(self, "termination_distribution", distribution)
        if self.schema_version != C3_EXECUTION_SUMMARY_SCHEMA:
            raise ValueError("unsupported C3 execution summary schema")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "execution_scenario": self.execution_scenario,
                "maximum_drawdown": self.maximum_drawdown,
                "mean_borrow_paid": self.mean_borrow_paid,
                "mean_fees": self.mean_fees,
                "mean_fill_ratio": self.mean_fill_ratio,
                "mean_filled_turnover": self.mean_filled_turnover,
                "mean_funding_paid": self.mean_funding_paid,
                "mean_gross_log_return": self.mean_gross_log_return,
                "mean_impact_cost": self.mean_impact_cost,
                "mean_spread_cost": self.mean_spread_cost,
                "mean_total_economic_cost": self.mean_total_economic_cost,
                "observation_count": self.observation_count,
                "policy_kind": self.policy_kind,
                "schema_version": self.schema_version,
                "termination_distribution": self.termination_distribution,
                "total_cancel_replace_events": self.total_cancel_replace_events,
                "total_fill_count": self.total_fill_count,
                "total_pending_order_events": self.total_pending_order_events,
            }
        )


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
    required_adverse_evidence_digest: str
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
        comparison_keys = tuple(
            (item.decision_digest, item.execution_scenario) for item in comparisons
        )
        if len(set(comparison_keys)) != len(comparisons):
            raise ValueError("comparison decision and scenario pairs must be unique")
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
        nominal_days = tuple(
            sorted(
                {
                    item.day_index
                    for item in comparisons
                    if item.execution_scenario == "nominal"
                }
            )
        )
        if nominal_days != tuple(days):
            raise ValueError("day_indices do not match nominal query comparisons")
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
        object.__setattr__(
            self,
            "required_adverse_evidence_digest",
            require_sha256(
                self.required_adverse_evidence_digest,
                field="required_adverse_evidence_digest",
            ),
        )
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
                "required_adverse_evidence_digest": (
                    self.required_adverse_evidence_digest
                ),
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
    calibration_buckets: tuple[C3CalibrationBucket, ...]
    neighbor_distance_p50: float
    neighbor_distance_p90: float
    neighbor_distance_p99: float
    unique_anchor_count: int
    anchor_max_share: float
    effective_anchor_count: float
    historical_coverage_fraction: float
    execution_summaries: tuple[C3PolicyExecutionSummary, ...]
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
        for field in ("total_selection_days", "total_effective_days"):
            object.__setattr__(self, field, _positive_int(field, getattr(self, field)))
        if self.total_effective_days > self.total_selection_days:
            raise ValueError("total_effective_days exceeds total_selection_days")
        positive = _non_negative_int(
            "positive_uplift_folds", self.positive_uplift_folds
        )
        if positive > len(folds):
            raise ValueError("positive_uplift_folds exceeds fold count")
        object.__setattr__(self, "positive_uplift_folds", positive)
        for field in ("bootstrap_resamples", "bootstrap_block_days"):
            object.__setattr__(self, field, _positive_int(field, getattr(self, field)))
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
            "neighbor_distance_p50",
            "neighbor_distance_p90",
            "neighbor_distance_p99",
            "anchor_max_share",
            "effective_anchor_count",
            "historical_coverage_fraction",
        ):
            object.__setattr__(self, field, _finite_float(field, getattr(self, field)))
        if not 0.0 <= self.uplift_p_value <= 1.0:
            raise ValueError("uplift_p_value must be in [0, 1]")
        for field in ("worst_scenario_oracle_drawdown", "worst_trend_drawdown"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
        if not (
            0.0
            <= self.neighbor_distance_p50
            <= self.neighbor_distance_p90
            <= self.neighbor_distance_p99
        ):
            raise ValueError("neighbor distance quantiles are invalid")
        object.__setattr__(
            self,
            "unique_anchor_count",
            _positive_int("unique_anchor_count", self.unique_anchor_count),
        )
        if not 0.0 < self.anchor_max_share <= 1.0:
            raise ValueError("anchor_max_share must be in (0, 1]")
        if not 0.0 < self.effective_anchor_count <= self.unique_anchor_count + 1e-12:
            raise ValueError("effective_anchor_count is invalid")
        if not 0.0 < self.historical_coverage_fraction <= 1.0:
            raise ValueError("historical_coverage_fraction must be in (0, 1]")
        buckets = tuple(self.calibration_buckets)
        if not buckets or any(
            not isinstance(item, C3CalibrationBucket) for item in buckets
        ):
            raise ValueError("calibration_buckets must contain C3 buckets")
        if tuple(item.bucket_index for item in buckets) != tuple(range(len(buckets))):
            raise ValueError("calibration bucket indices must be contiguous")
        object.__setattr__(self, "calibration_buckets", buckets)
        summaries = tuple(self.execution_summaries)
        if not summaries or any(
            not isinstance(item, C3PolicyExecutionSummary) for item in summaries
        ):
            raise ValueError("execution_summaries must contain C3 summaries")
        keys = tuple((item.execution_scenario, item.policy_kind) for item in summaries)
        if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
            raise ValueError("execution summary keys must be sorted and unique")
        object.__setattr__(self, "execution_summaries", summaries)
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
    def execution_scenario_names(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.execution_scenario for item in self.execution_summaries})
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "all_perfect_information_valid": self.all_perfect_information_valid,
                "all_required_adverse_passed": self.all_required_adverse_passed,
                "anchor_max_share": self.anchor_max_share,
                "bootstrap_block_days": self.bootstrap_block_days,
                "bootstrap_resamples": self.bootstrap_resamples,
                "calibration_bucket_digests": tuple(
                    item.digest for item in self.calibration_buckets
                ),
                "effective_anchor_count": self.effective_anchor_count,
                "execution_summary_digests": tuple(
                    item.digest for item in self.execution_summaries
                ),
                "failure_reasons": self.failure_reasons,
                "fold_digests": tuple(item.digest for item in self.folds),
                "historical_coverage_fraction": self.historical_coverage_fraction,
                "mean_regret_margin": self.mean_regret_margin,
                "mean_spearman": self.mean_spearman,
                "mean_uplift": self.mean_uplift,
                "neighbor_distance_p50": self.neighbor_distance_p50,
                "neighbor_distance_p90": self.neighbor_distance_p90,
                "neighbor_distance_p99": self.neighbor_distance_p99,
                "positive_uplift_folds": self.positive_uplift_folds,
                "regret_margin_lower_ci": self.regret_margin_lower_ci,
                "regret_margin_upper_ci": self.regret_margin_upper_ci,
                "schema_version": self.schema_version,
                "spearman_lower_ci": self.spearman_lower_ci,
                "spearman_upper_ci": self.spearman_upper_ci,
                "total_effective_days": self.total_effective_days,
                "total_selection_days": self.total_selection_days,
                "unique_anchor_count": self.unique_anchor_count,
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
    required_adverse_evidence_digest: str,
    failure_reasons: tuple[str, ...] = (),
) -> CausalScenarioFoldReport:
    items = tuple(comparisons)
    if not items:
        raise ValueError("comparisons must not be empty")
    nominal_items = tuple(
        item for item in items if item.execution_scenario == "nominal"
    )
    if not nominal_items:
        raise ValueError("each fold requires nominal C3 comparisons")
    grouped: dict[int, list[CausalScenarioQueryComparison]] = defaultdict(list)
    for item in nominal_items:
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
        and item.perfect_information.compatibility_evidence_digest is not None
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
            item.scenario_oracle.max_drawdown for item in nominal_items
        ),
        trend_max_drawdown=max(item.trend.max_drawdown for item in nominal_items),
        required_adverse_passed=required_adverse_passed,
        required_adverse_evidence_digest=required_adverse_evidence_digest,
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


def _calibration_buckets(
    comparisons: tuple[CausalScenarioQueryComparison, ...],
    *,
    bucket_count: int = 5,
) -> tuple[C3CalibrationBucket, ...]:
    scores = np.concatenate([item.predicted_score for item in comparisons])
    predicted_mean = np.concatenate(
        [item.predicted_mean_advantage for item in comparisons]
    )
    predicted_cvar = np.concatenate([item.predicted_loss_cvar for item in comparisons])
    realized = np.concatenate(
        [item.realized_candidate_advantages for item in comparisons]
    )
    order = np.argsort(scores, kind="mergesort")
    groups = tuple(
        group
        for group in np.array_split(order, min(bucket_count, order.size))
        if group.size
    )
    return tuple(
        C3CalibrationBucket(
            bucket_index=index,
            sample_count=int(group.size),
            minimum_score=float(scores[group].min()),
            maximum_score=float(scores[group].max()),
            predicted_mean_advantage=float(predicted_mean[group].mean()),
            predicted_loss_cvar=float(predicted_cvar[group].mean()),
            realized_mean_advantage=float(realized[group].mean()),
            realized_downside_mean=float(np.minimum(realized[group], 0.0).mean()),
        )
        for index, group in enumerate(groups)
    )


def _policy_observations(
    comparison: CausalScenarioQueryComparison,
) -> tuple[tuple[str, RealizedPolicyOutcome], ...]:
    return (
        ("trend", comparison.trend),
        ("scenario_oracle", comparison.scenario_oracle),
        ("ppo_mean", comparison.ppo_mean),
        *(("random_candidate", item) for item in comparison.random_candidate_outcomes),
    )


def _execution_summaries(
    comparisons: tuple[CausalScenarioQueryComparison, ...],
) -> tuple[C3PolicyExecutionSummary, ...]:
    grouped: dict[tuple[str, str], list[RealizedPolicyOutcome]] = defaultdict(list)
    for comparison in comparisons:
        for policy_kind, outcome in _policy_observations(comparison):
            grouped[(comparison.execution_scenario, policy_kind)].append(outcome)
    summaries = []
    for (scenario, policy), outcomes in sorted(grouped.items()):
        terminations = Counter(item.termination_reason for item in outcomes)
        summaries.append(
            C3PolicyExecutionSummary(
                execution_scenario=scenario,
                policy_kind=policy,
                observation_count=len(outcomes),
                mean_gross_log_return=float(
                    np.mean([item.gross_log_return for item in outcomes])
                ),
                mean_filled_turnover=float(
                    np.mean([item.filled_turnover for item in outcomes])
                ),
                mean_fees=float(np.mean([item.fees for item in outcomes])),
                mean_spread_cost=float(
                    np.mean([item.spread_cost for item in outcomes])
                ),
                mean_impact_cost=float(
                    np.mean([item.impact_cost for item in outcomes])
                ),
                mean_funding_paid=float(
                    np.mean([item.funding_paid for item in outcomes])
                ),
                mean_borrow_paid=float(
                    np.mean([item.borrow_paid for item in outcomes])
                ),
                mean_total_economic_cost=float(
                    np.mean([item.total_economic_cost for item in outcomes])
                ),
                mean_fill_ratio=float(np.mean([item.fill_ratio for item in outcomes])),
                total_fill_count=sum(item.fill_count for item in outcomes),
                total_pending_order_events=sum(
                    item.pending_order_events for item in outcomes
                ),
                total_cancel_replace_events=sum(
                    item.cancel_replace_events for item in outcomes
                ),
                maximum_drawdown=max(item.max_drawdown for item in outcomes),
                termination_distribution=tuple(sorted(terminations.items())),
            )
        )
    return tuple(summaries)


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
    comparisons = tuple(item for fold in items for item in fold.comparisons)
    nominal_comparisons = tuple(
        item for item in comparisons if item.execution_scenario == "nominal"
    )
    if not nominal_comparisons:
        raise ValueError("aggregate C3 report requires nominal comparisons")
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
    distances = np.concatenate(
        [item.scenario_distances for item in nominal_comparisons]
    )
    anchors = np.concatenate(
        [item.scenario_anchor_indices for item in nominal_comparisons]
    )
    unique_anchors, anchor_counts = np.unique(anchors, return_counts=True)
    total_anchor_samples = int(anchor_counts.sum())
    anchor_shares = anchor_counts.astype(np.float64) / total_anchor_samples
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
        calibration_buckets=_calibration_buckets(nominal_comparisons),
        neighbor_distance_p50=float(np.quantile(distances, 0.50)),
        neighbor_distance_p90=float(np.quantile(distances, 0.90)),
        neighbor_distance_p99=float(np.quantile(distances, 0.99)),
        unique_anchor_count=int(unique_anchors.size),
        anchor_max_share=float(anchor_shares.max()),
        effective_anchor_count=float(1.0 / np.square(anchor_shares).sum()),
        historical_coverage_fraction=float(unique_anchors.size / total_anchor_samples),
        execution_summaries=_execution_summaries(comparisons),
        all_required_adverse_passed=all(item.required_adverse_passed for item in items),
        all_perfect_information_valid=all(
            item.perfect_information_valid for item in items
        ),
        failure_reasons=failures,
        bootstrap_resamples=resamples,
        bootstrap_block_days=block_days,
    )
