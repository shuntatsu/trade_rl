"""Strict reporting, gate, and artifact boundary for causal-scenario C3."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

C3_AGGREGATE_SUMMARY_SCHEMA: Final = "causal_scenario_c3_aggregate_summary_v1"
C3_PHASE_A_GATE_CONFIG_SCHEMA: Final = "causal_scenario_c3_phase_a_gate_config_v1"
C3_PHASE_A_GATE_SCHEMA: Final = "causal_scenario_c3_phase_a_gate_v1"
C3_REPORT_ARTIFACT_SCHEMA: Final = "causal_scenario_c3_report_artifact_v1"
C3_GATE_ARTIFACT_SCHEMA: Final = "causal_scenario_c3_gate_artifact_v1"
PRODUCTION_STATUS: Final = "NO-GO"

_REPORT_FILES: Final = frozenset({"manifest.json", "summary.json", "report.md"})
_GATE_FILES: Final = frozenset({"manifest.json", "gate.json"})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return dict(value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise ValueError(f"{field} must be a list")


def _require_fields(
    payload: Mapping[str, object], expected: set[str], *, field: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{field} field closure mismatch")


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _probability(value: object, *, field: str) -> float:
    result = _number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _digest(value: object, *, field: str) -> str:
    return require_sha256(_string(value, field=field), field=field)


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    result = tuple(
        _string(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{field} values must be unique")
    return result


def _check_interval(lower: float, mean: float, upper: float, *, field: str) -> None:
    if not lower <= mean <= upper:
        raise ValueError(f"{field} confidence interval is invalid")


@dataclass(frozen=True, slots=True)
class C3FoldSummary:
    fold_id: str
    selection_days: int
    effective_days: int
    mean_uplift: float
    mean_spearman: float
    mean_regret_margin: float
    scenario_oracle_max_drawdown: float
    trend_max_drawdown: float
    required_adverse_passed: bool
    perfect_information_valid: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold_id", _string(self.fold_id, field="fold_id"))
        selection_days = _integer(
            self.selection_days, field="selection_days", minimum=1
        )
        effective_days = _integer(
            self.effective_days, field="effective_days", minimum=1
        )
        if effective_days > selection_days:
            raise ValueError("effective_days must not exceed selection_days")
        object.__setattr__(self, "selection_days", selection_days)
        object.__setattr__(self, "effective_days", effective_days)
        for name in ("mean_uplift", "mean_spearman", "mean_regret_margin"):
            object.__setattr__(self, name, _number(getattr(self, name), field=name))
        for name in ("scenario_oracle_max_drawdown", "trend_max_drawdown"):
            object.__setattr__(
                self, name, _probability(getattr(self, name), field=name)
            )
        if not isinstance(self.required_adverse_passed, bool):
            raise ValueError("required_adverse_passed must be boolean")
        if not isinstance(self.perfect_information_valid, bool):
            raise ValueError("perfect_information_valid must be boolean")
        reasons = tuple(reason.strip() for reason in self.failure_reasons)
        if any(not reason for reason in reasons) or len(set(reasons)) != len(reasons):
            raise ValueError("failure_reasons must be unique and non-empty")
        object.__setattr__(self, "failure_reasons", reasons)

    def to_payload(self) -> dict[str, object]:
        return {
            "effective_days": self.effective_days,
            "failure_reasons": list(self.failure_reasons),
            "fold_id": self.fold_id,
            "mean_regret_margin": self.mean_regret_margin,
            "mean_spearman": self.mean_spearman,
            "mean_uplift": self.mean_uplift,
            "perfect_information_valid": self.perfect_information_valid,
            "required_adverse_passed": self.required_adverse_passed,
            "scenario_oracle_max_drawdown": self.scenario_oracle_max_drawdown,
            "selection_days": self.selection_days,
            "trend_max_drawdown": self.trend_max_drawdown,
        }


@dataclass(frozen=True, slots=True)
class C3CalibrationBucketSummary:
    bucket_index: int
    sample_count: int
    minimum_score: float
    maximum_score: float
    predicted_mean_advantage: float
    predicted_loss_cvar: float
    realized_mean_advantage: float
    realized_downside_mean: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bucket_index", _integer(self.bucket_index, field="bucket_index")
        )
        object.__setattr__(
            self,
            "sample_count",
            _integer(self.sample_count, field="sample_count", minimum=1),
        )
        for name in (
            "minimum_score",
            "maximum_score",
            "predicted_mean_advantage",
            "predicted_loss_cvar",
            "realized_mean_advantage",
            "realized_downside_mean",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), field=name))
        if self.minimum_score > self.maximum_score:
            raise ValueError("calibration bucket score range is invalid")
        if self.predicted_loss_cvar < 0.0:
            raise ValueError("predicted_loss_cvar must be non-negative")
        if self.realized_downside_mean > 0.0:
            raise ValueError("realized_downside_mean must be non-positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "bucket_index": self.bucket_index,
            "maximum_score": self.maximum_score,
            "minimum_score": self.minimum_score,
            "predicted_loss_cvar": self.predicted_loss_cvar,
            "predicted_mean_advantage": self.predicted_mean_advantage,
            "realized_downside_mean": self.realized_downside_mean,
            "realized_mean_advantage": self.realized_mean_advantage,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class C3ExecutionSummary:
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_scenario",
            _string(self.execution_scenario, field="execution_scenario"),
        )
        object.__setattr__(
            self, "policy_kind", _string(self.policy_kind, field="policy_kind")
        )
        object.__setattr__(
            self,
            "observation_count",
            _integer(self.observation_count, field="observation_count", minimum=1),
        )
        for name in (
            "mean_gross_log_return",
            "mean_filled_turnover",
            "mean_fees",
            "mean_spread_cost",
            "mean_impact_cost",
            "mean_funding_paid",
            "mean_borrow_paid",
            "mean_total_economic_cost",
        ):
            value = _number(getattr(self, name), field=name)
            if name != "mean_gross_log_return" and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        expected_cost = (
            self.mean_fees
            + self.mean_spread_cost
            + self.mean_impact_cost
            + self.mean_funding_paid
            + self.mean_borrow_paid
        )
        if not math.isclose(
            self.mean_total_economic_cost,
            expected_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("mean_total_economic_cost does not match components")
        object.__setattr__(
            self,
            "mean_fill_ratio",
            _probability(self.mean_fill_ratio, field="mean_fill_ratio"),
        )
        object.__setattr__(
            self,
            "maximum_drawdown",
            _probability(self.maximum_drawdown, field="maximum_drawdown"),
        )
        for name in (
            "total_fill_count",
            "total_pending_order_events",
            "total_cancel_replace_events",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), field=name))
        distribution = tuple(
            (
                _string(reason, field="termination_reason"),
                _integer(count, field="termination_count", minimum=1),
            )
            for reason, count in self.termination_distribution
        )
        if tuple(sorted(distribution)) != distribution:
            raise ValueError("termination_distribution must be sorted")
        if len({reason for reason, _ in distribution}) != len(distribution):
            raise ValueError("termination_distribution reasons must be unique")
        if sum(count for _, count in distribution) != self.observation_count:
            raise ValueError("termination_distribution count mismatch")
        object.__setattr__(self, "termination_distribution", distribution)

    @property
    def key(self) -> tuple[str, str]:
        return self.execution_scenario, self.policy_kind

    def to_payload(self) -> dict[str, object]:
        return {
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
            "termination_distribution": [
                [reason, count] for reason, count in self.termination_distribution
            ],
            "total_cancel_replace_events": self.total_cancel_replace_events,
            "total_fill_count": self.total_fill_count,
            "total_pending_order_events": self.total_pending_order_events,
        }


@dataclass(frozen=True, slots=True)
class C3AggregateSummary:
    source_run_digest: str
    core_report_digest: str
    config_digest: str
    folds: tuple[C3FoldSummary, ...]
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
    calibration_buckets: tuple[C3CalibrationBucketSummary, ...]
    neighbor_distance_p50: float
    neighbor_distance_p90: float
    neighbor_distance_p99: float
    unique_anchor_count: int
    anchor_max_share: float
    effective_anchor_count: float
    historical_coverage_fraction: float
    execution_summaries: tuple[C3ExecutionSummary, ...]
    all_required_adverse_passed: bool
    all_perfect_information_valid: bool
    failure_reasons: tuple[str, ...]
    bootstrap_resamples: int
    bootstrap_block_days: int
    summary_digest: str
    schema_version: str = C3_AGGREGATE_SUMMARY_SCHEMA
    production_status: str = PRODUCTION_STATUS

    def __post_init__(self) -> None:
        for name in (
            "source_run_digest",
            "core_report_digest",
            "config_digest",
            "summary_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), field=name))
        if self.schema_version != C3_AGGREGATE_SUMMARY_SCHEMA:
            raise ValueError("unsupported C3 aggregate summary schema")
        if self.production_status != PRODUCTION_STATUS:
            raise ValueError("C3 aggregate summary production status must remain NO-GO")
        folds = tuple(self.folds)
        if not folds:
            raise ValueError("folds must not be empty")
        fold_ids = tuple(item.fold_id for item in folds)
        if len(set(fold_ids)) != len(fold_ids):
            raise ValueError("fold IDs must be unique")
        if tuple(sorted(fold_ids)) != fold_ids:
            raise ValueError("fold IDs must be sorted")
        object.__setattr__(self, "folds", folds)
        total_selection = _integer(
            self.total_selection_days, field="total_selection_days", minimum=1
        )
        total_effective = _integer(
            self.total_effective_days, field="total_effective_days", minimum=1
        )
        if total_selection != sum(item.selection_days for item in folds):
            raise ValueError("total_selection_days does not match folds")
        if total_effective != sum(item.effective_days for item in folds):
            raise ValueError("total_effective_days does not match folds")
        object.__setattr__(self, "total_selection_days", total_selection)
        object.__setattr__(self, "total_effective_days", total_effective)
        positive = _integer(self.positive_uplift_folds, field="positive_uplift_folds")
        if positive != sum(item.mean_uplift > 0.0 for item in folds):
            raise ValueError("positive_uplift_folds does not match folds")
        object.__setattr__(self, "positive_uplift_folds", positive)
        for name in (
            "mean_uplift",
            "uplift_lower_ci",
            "uplift_upper_ci",
            "mean_spearman",
            "spearman_lower_ci",
            "spearman_upper_ci",
            "mean_regret_margin",
            "regret_margin_lower_ci",
            "regret_margin_upper_ci",
            "neighbor_distance_p50",
            "neighbor_distance_p90",
            "neighbor_distance_p99",
            "effective_anchor_count",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), field=name))
        _check_interval(
            self.uplift_lower_ci,
            self.mean_uplift,
            self.uplift_upper_ci,
            field="uplift",
        )
        _check_interval(
            self.spearman_lower_ci,
            self.mean_spearman,
            self.spearman_upper_ci,
            field="spearman",
        )
        _check_interval(
            self.regret_margin_lower_ci,
            self.mean_regret_margin,
            self.regret_margin_upper_ci,
            field="regret margin",
        )
        object.__setattr__(
            self,
            "uplift_p_value",
            _probability(self.uplift_p_value, field="uplift_p_value"),
        )
        object.__setattr__(
            self,
            "worst_scenario_oracle_drawdown",
            _probability(
                self.worst_scenario_oracle_drawdown,
                field="worst_scenario_oracle_drawdown",
            ),
        )
        object.__setattr__(
            self,
            "worst_trend_drawdown",
            _probability(self.worst_trend_drawdown, field="worst_trend_drawdown"),
        )
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
            _integer(self.unique_anchor_count, field="unique_anchor_count", minimum=1),
        )
        object.__setattr__(
            self,
            "anchor_max_share",
            _probability(self.anchor_max_share, field="anchor_max_share"),
        )
        if self.anchor_max_share <= 0.0:
            raise ValueError("anchor_max_share must be positive")
        if not 0.0 < self.effective_anchor_count <= self.unique_anchor_count + 1e-12:
            raise ValueError("effective_anchor_count is invalid")
        object.__setattr__(
            self,
            "historical_coverage_fraction",
            _probability(
                self.historical_coverage_fraction,
                field="historical_coverage_fraction",
            ),
        )
        if self.historical_coverage_fraction <= 0.0:
            raise ValueError("historical_coverage_fraction must be positive")
        buckets = tuple(self.calibration_buckets)
        if not buckets:
            raise ValueError("calibration_buckets must not be empty")
        if tuple(item.bucket_index for item in buckets) != tuple(range(len(buckets))):
            raise ValueError("calibration bucket indices must be contiguous")
        object.__setattr__(self, "calibration_buckets", buckets)
        summaries = tuple(self.execution_summaries)
        if not summaries:
            raise ValueError("execution_summaries must not be empty")
        keys = tuple(item.key for item in summaries)
        if tuple(sorted(keys)) != keys:
            raise ValueError("execution summaries must be sorted")
        if len(set(keys)) != len(keys):
            raise ValueError("execution summary keys must be unique")
        object.__setattr__(self, "execution_summaries", summaries)
        if not isinstance(self.all_required_adverse_passed, bool):
            raise ValueError("all_required_adverse_passed must be boolean")
        if not isinstance(self.all_perfect_information_valid, bool):
            raise ValueError("all_perfect_information_valid must be boolean")
        if self.all_required_adverse_passed != all(
            item.required_adverse_passed for item in folds
        ):
            raise ValueError("all_required_adverse_passed does not match folds")
        if self.all_perfect_information_valid != all(
            item.perfect_information_valid for item in folds
        ):
            raise ValueError("all_perfect_information_valid does not match folds")
        reasons = tuple(reason.strip() for reason in self.failure_reasons)
        if any(not reason for reason in reasons) or len(set(reasons)) != len(reasons):
            raise ValueError("failure_reasons must be unique and non-empty")
        object.__setattr__(self, "failure_reasons", reasons)
        object.__setattr__(
            self,
            "bootstrap_resamples",
            _integer(self.bootstrap_resamples, field="bootstrap_resamples", minimum=1),
        )
        object.__setattr__(
            self,
            "bootstrap_block_days",
            _integer(
                self.bootstrap_block_days, field="bootstrap_block_days", minimum=1
            ),
        )

    @property
    def fold_count(self) -> int:
        return len(self.folds)

    @property
    def execution_scenario_names(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.execution_scenario for item in self.execution_summaries})
        )

    def payload_without_digest(self) -> dict[str, object]:
        return {
            "all_perfect_information_valid": self.all_perfect_information_valid,
            "all_required_adverse_passed": self.all_required_adverse_passed,
            "anchor_max_share": self.anchor_max_share,
            "bootstrap_block_days": self.bootstrap_block_days,
            "bootstrap_resamples": self.bootstrap_resamples,
            "calibration_buckets": [
                item.to_payload() for item in self.calibration_buckets
            ],
            "config_digest": self.config_digest,
            "core_report_digest": self.core_report_digest,
            "effective_anchor_count": self.effective_anchor_count,
            "execution_summaries": [
                item.to_payload() for item in self.execution_summaries
            ],
            "failure_reasons": list(self.failure_reasons),
            "folds": [item.to_payload() for item in self.folds],
            "historical_coverage_fraction": self.historical_coverage_fraction,
            "mean_regret_margin": self.mean_regret_margin,
            "mean_spearman": self.mean_spearman,
            "mean_uplift": self.mean_uplift,
            "neighbor_distance_p50": self.neighbor_distance_p50,
            "neighbor_distance_p90": self.neighbor_distance_p90,
            "neighbor_distance_p99": self.neighbor_distance_p99,
            "positive_uplift_folds": self.positive_uplift_folds,
            "production_status": self.production_status,
            "regret_margin_lower_ci": self.regret_margin_lower_ci,
            "regret_margin_upper_ci": self.regret_margin_upper_ci,
            "schema_version": self.schema_version,
            "source_run_digest": self.source_run_digest,
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

    def to_payload(self) -> dict[str, object]:
        payload = self.payload_without_digest()
        payload["summary_digest"] = self.summary_digest
        return payload


@dataclass(frozen=True, slots=True)
class C3PhaseAGateConfig:
    required_folds: int = 6
    required_selection_days: int = 180
    required_positive_folds: int = 4
    max_oracle_drawdown: float = 0.20
    trend_drawdown_tolerance: float = 0.02
    schema_version: str = C3_PHASE_A_GATE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "required_folds",
            "required_selection_days",
            "required_positive_folds",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), field=name, minimum=1)
            )
        if self.required_positive_folds > self.required_folds:
            raise ValueError("required_positive_folds must not exceed required_folds")
        object.__setattr__(
            self,
            "max_oracle_drawdown",
            _probability(self.max_oracle_drawdown, field="max_oracle_drawdown"),
        )
        tolerance = _number(
            self.trend_drawdown_tolerance, field="trend_drawdown_tolerance"
        )
        if tolerance < 0.0:
            raise ValueError("trend_drawdown_tolerance must be non-negative")
        object.__setattr__(self, "trend_drawdown_tolerance", tolerance)
        if self.schema_version != C3_PHASE_A_GATE_CONFIG_SCHEMA:
            raise ValueError("unsupported Phase A gate config schema")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "max_oracle_drawdown": self.max_oracle_drawdown,
                "required_folds": self.required_folds,
                "required_positive_folds": self.required_positive_folds,
                "required_selection_days": self.required_selection_days,
                "schema_version": self.schema_version,
                "trend_drawdown_tolerance": self.trend_drawdown_tolerance,
            }
        )


@dataclass(frozen=True, slots=True)
class GateConditionResult:
    name: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, field="condition.name"))
        object.__setattr__(
            self, "detail", _string(self.detail, field="condition.detail")
        )
        if not isinstance(self.passed, bool):
            raise ValueError("condition.passed must be boolean")

    def to_payload(self) -> dict[str, object]:
        return {"detail": self.detail, "name": self.name, "passed": self.passed}


@dataclass(frozen=True, slots=True)
class PhaseAGateEvidence:
    report_digest: str
    config_digest: str
    conditions: tuple[GateConditionResult, ...]
    passed: bool
    schema_version: str = C3_PHASE_A_GATE_SCHEMA
    production_status: str = PRODUCTION_STATUS

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "report_digest", _digest(self.report_digest, field="report_digest")
        )
        object.__setattr__(
            self, "config_digest", _digest(self.config_digest, field="config_digest")
        )
        conditions = tuple(self.conditions)
        if len(conditions) != 9:
            raise ValueError("Phase A gate must contain exactly nine conditions")
        names = tuple(item.name for item in conditions)
        if len(set(names)) != len(names):
            raise ValueError("Phase A gate condition names must be unique")
        if not isinstance(self.passed, bool) or self.passed != all(
            item.passed for item in conditions
        ):
            raise ValueError("Phase A gate pass state does not match conditions")
        if self.schema_version != C3_PHASE_A_GATE_SCHEMA:
            raise ValueError("unsupported Phase A gate schema")
        if self.production_status != PRODUCTION_STATUS:
            raise ValueError("Phase A gate production status must remain NO-GO")
        object.__setattr__(self, "conditions", conditions)

    @property
    def failed_condition_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.conditions if not item.passed)

    @property
    def digest(self) -> str:
        return content_digest(self.payload_without_digest())

    def payload_without_digest(self) -> dict[str, object]:
        return {
            "conditions": [item.to_payload() for item in self.conditions],
            "config_digest": self.config_digest,
            "passed": self.passed,
            "production_status": self.production_status,
            "report_digest": self.report_digest,
            "schema_version": self.schema_version,
        }

    def to_payload(self) -> dict[str, object]:
        payload = self.payload_without_digest()
        payload["gate_digest"] = self.digest
        return payload


def _fold(value: object, *, field: str) -> C3FoldSummary:
    payload = _mapping(value, field=field)
    _require_fields(
        payload,
        {
            "effective_days",
            "failure_reasons",
            "fold_id",
            "mean_regret_margin",
            "mean_spearman",
            "mean_uplift",
            "perfect_information_valid",
            "required_adverse_passed",
            "scenario_oracle_max_drawdown",
            "selection_days",
            "trend_max_drawdown",
        },
        field=field,
    )
    return C3FoldSummary(
        fold_id=_string(payload["fold_id"], field=f"{field}.fold_id"),
        selection_days=_integer(
            payload["selection_days"], field=f"{field}.selection_days", minimum=1
        ),
        effective_days=_integer(
            payload["effective_days"], field=f"{field}.effective_days", minimum=1
        ),
        mean_uplift=_number(payload["mean_uplift"], field=f"{field}.mean_uplift"),
        mean_spearman=_number(payload["mean_spearman"], field=f"{field}.mean_spearman"),
        mean_regret_margin=_number(
            payload["mean_regret_margin"], field=f"{field}.mean_regret_margin"
        ),
        scenario_oracle_max_drawdown=_probability(
            payload["scenario_oracle_max_drawdown"],
            field=f"{field}.scenario_oracle_max_drawdown",
        ),
        trend_max_drawdown=_probability(
            payload["trend_max_drawdown"], field=f"{field}.trend_max_drawdown"
        ),
        required_adverse_passed=_boolean(
            payload["required_adverse_passed"], field=f"{field}.required_adverse_passed"
        ),
        perfect_information_valid=_boolean(
            payload["perfect_information_valid"],
            field=f"{field}.perfect_information_valid",
        ),
        failure_reasons=_strings(
            payload["failure_reasons"], field=f"{field}.failure_reasons"
        ),
    )


def _bucket(value: object, *, field: str) -> C3CalibrationBucketSummary:
    payload = _mapping(value, field=field)
    _require_fields(
        payload,
        {
            "bucket_index",
            "maximum_score",
            "minimum_score",
            "predicted_loss_cvar",
            "predicted_mean_advantage",
            "realized_downside_mean",
            "realized_mean_advantage",
            "sample_count",
        },
        field=field,
    )
    return C3CalibrationBucketSummary(
        bucket_index=_integer(payload["bucket_index"], field=f"{field}.bucket_index"),
        sample_count=_integer(
            payload["sample_count"], field=f"{field}.sample_count", minimum=1
        ),
        minimum_score=_number(payload["minimum_score"], field=f"{field}.minimum_score"),
        maximum_score=_number(payload["maximum_score"], field=f"{field}.maximum_score"),
        predicted_mean_advantage=_number(
            payload["predicted_mean_advantage"],
            field=f"{field}.predicted_mean_advantage",
        ),
        predicted_loss_cvar=_number(
            payload["predicted_loss_cvar"], field=f"{field}.predicted_loss_cvar"
        ),
        realized_mean_advantage=_number(
            payload["realized_mean_advantage"], field=f"{field}.realized_mean_advantage"
        ),
        realized_downside_mean=_number(
            payload["realized_downside_mean"], field=f"{field}.realized_downside_mean"
        ),
    )


def _execution(value: object, *, field: str) -> C3ExecutionSummary:
    payload = _mapping(value, field=field)
    _require_fields(
        payload,
        {
            "execution_scenario",
            "maximum_drawdown",
            "mean_borrow_paid",
            "mean_fees",
            "mean_fill_ratio",
            "mean_filled_turnover",
            "mean_funding_paid",
            "mean_gross_log_return",
            "mean_impact_cost",
            "mean_spread_cost",
            "mean_total_economic_cost",
            "observation_count",
            "policy_kind",
            "termination_distribution",
            "total_cancel_replace_events",
            "total_fill_count",
            "total_pending_order_events",
        },
        field=field,
    )
    distribution: list[tuple[str, int]] = []
    for index, raw in enumerate(
        _sequence(
            payload["termination_distribution"],
            field=f"{field}.termination_distribution",
        )
    ):
        item = _sequence(raw, field=f"{field}.termination_distribution[{index}]")
        if len(item) != 2:
            raise ValueError(
                f"{field}.termination_distribution[{index}] must contain two values"
            )
        distribution.append(
            (
                _string(item[0], field=f"{field}.termination_distribution[{index}][0]"),
                _integer(
                    item[1],
                    field=f"{field}.termination_distribution[{index}][1]",
                    minimum=1,
                ),
            )
        )
    return C3ExecutionSummary(
        execution_scenario=_string(
            payload["execution_scenario"], field=f"{field}.execution_scenario"
        ),
        policy_kind=_string(payload["policy_kind"], field=f"{field}.policy_kind"),
        observation_count=_integer(
            payload["observation_count"], field=f"{field}.observation_count", minimum=1
        ),
        mean_gross_log_return=_number(
            payload["mean_gross_log_return"], field=f"{field}.mean_gross_log_return"
        ),
        mean_filled_turnover=_number(
            payload["mean_filled_turnover"], field=f"{field}.mean_filled_turnover"
        ),
        mean_fees=_number(payload["mean_fees"], field=f"{field}.mean_fees"),
        mean_spread_cost=_number(
            payload["mean_spread_cost"], field=f"{field}.mean_spread_cost"
        ),
        mean_impact_cost=_number(
            payload["mean_impact_cost"], field=f"{field}.mean_impact_cost"
        ),
        mean_funding_paid=_number(
            payload["mean_funding_paid"], field=f"{field}.mean_funding_paid"
        ),
        mean_borrow_paid=_number(
            payload["mean_borrow_paid"], field=f"{field}.mean_borrow_paid"
        ),
        mean_total_economic_cost=_number(
            payload["mean_total_economic_cost"],
            field=f"{field}.mean_total_economic_cost",
        ),
        mean_fill_ratio=_number(
            payload["mean_fill_ratio"], field=f"{field}.mean_fill_ratio"
        ),
        total_fill_count=_integer(
            payload["total_fill_count"], field=f"{field}.total_fill_count"
        ),
        total_pending_order_events=_integer(
            payload["total_pending_order_events"],
            field=f"{field}.total_pending_order_events",
        ),
        total_cancel_replace_events=_integer(
            payload["total_cancel_replace_events"],
            field=f"{field}.total_cancel_replace_events",
        ),
        maximum_drawdown=_number(
            payload["maximum_drawdown"], field=f"{field}.maximum_drawdown"
        ),
        termination_distribution=tuple(distribution),
    )


def load_c3_aggregate_summary(path: str | Path) -> C3AggregateSummary:
    """Load a canonical, exact-closure aggregate summary produced by lane B."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"C3 aggregate summary file is missing: {source}")
    raw_bytes = source.read_bytes()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("C3 aggregate summary is invalid JSON") from error
    payload = _mapping(raw, field="summary")
    _require_fields(
        payload,
        {
            "all_perfect_information_valid",
            "all_required_adverse_passed",
            "anchor_max_share",
            "bootstrap_block_days",
            "bootstrap_resamples",
            "calibration_buckets",
            "config_digest",
            "core_report_digest",
            "effective_anchor_count",
            "execution_summaries",
            "failure_reasons",
            "folds",
            "historical_coverage_fraction",
            "mean_regret_margin",
            "mean_spearman",
            "mean_uplift",
            "neighbor_distance_p50",
            "neighbor_distance_p90",
            "neighbor_distance_p99",
            "positive_uplift_folds",
            "production_status",
            "regret_margin_lower_ci",
            "regret_margin_upper_ci",
            "schema_version",
            "source_run_digest",
            "spearman_lower_ci",
            "spearman_upper_ci",
            "summary_digest",
            "total_effective_days",
            "total_selection_days",
            "unique_anchor_count",
            "uplift_lower_ci",
            "uplift_p_value",
            "uplift_upper_ci",
            "worst_scenario_oracle_drawdown",
            "worst_trend_drawdown",
        },
        field="summary",
    )
    folds = tuple(
        _fold(item, field=f"folds[{index}]")
        for index, item in enumerate(_sequence(payload["folds"], field="folds"))
    )
    buckets = tuple(
        _bucket(item, field=f"calibration_buckets[{index}]")
        for index, item in enumerate(
            _sequence(payload["calibration_buckets"], field="calibration_buckets")
        )
    )
    execution = tuple(
        _execution(item, field=f"execution_summaries[{index}]")
        for index, item in enumerate(
            _sequence(payload["execution_summaries"], field="execution_summaries")
        )
    )
    summary = C3AggregateSummary(
        source_run_digest=_digest(
            payload["source_run_digest"], field="source_run_digest"
        ),
        core_report_digest=_digest(
            payload["core_report_digest"], field="core_report_digest"
        ),
        config_digest=_digest(payload["config_digest"], field="config_digest"),
        folds=folds,
        total_selection_days=_integer(
            payload["total_selection_days"], field="total_selection_days", minimum=1
        ),
        total_effective_days=_integer(
            payload["total_effective_days"], field="total_effective_days", minimum=1
        ),
        positive_uplift_folds=_integer(
            payload["positive_uplift_folds"], field="positive_uplift_folds"
        ),
        mean_uplift=_number(payload["mean_uplift"], field="mean_uplift"),
        uplift_lower_ci=_number(payload["uplift_lower_ci"], field="uplift_lower_ci"),
        uplift_upper_ci=_number(payload["uplift_upper_ci"], field="uplift_upper_ci"),
        uplift_p_value=_number(payload["uplift_p_value"], field="uplift_p_value"),
        mean_spearman=_number(payload["mean_spearman"], field="mean_spearman"),
        spearman_lower_ci=_number(
            payload["spearman_lower_ci"], field="spearman_lower_ci"
        ),
        spearman_upper_ci=_number(
            payload["spearman_upper_ci"], field="spearman_upper_ci"
        ),
        mean_regret_margin=_number(
            payload["mean_regret_margin"], field="mean_regret_margin"
        ),
        regret_margin_lower_ci=_number(
            payload["regret_margin_lower_ci"], field="regret_margin_lower_ci"
        ),
        regret_margin_upper_ci=_number(
            payload["regret_margin_upper_ci"], field="regret_margin_upper_ci"
        ),
        worst_scenario_oracle_drawdown=_number(
            payload["worst_scenario_oracle_drawdown"],
            field="worst_scenario_oracle_drawdown",
        ),
        worst_trend_drawdown=_number(
            payload["worst_trend_drawdown"], field="worst_trend_drawdown"
        ),
        calibration_buckets=buckets,
        neighbor_distance_p50=_number(
            payload["neighbor_distance_p50"], field="neighbor_distance_p50"
        ),
        neighbor_distance_p90=_number(
            payload["neighbor_distance_p90"], field="neighbor_distance_p90"
        ),
        neighbor_distance_p99=_number(
            payload["neighbor_distance_p99"], field="neighbor_distance_p99"
        ),
        unique_anchor_count=_integer(
            payload["unique_anchor_count"], field="unique_anchor_count", minimum=1
        ),
        anchor_max_share=_number(payload["anchor_max_share"], field="anchor_max_share"),
        effective_anchor_count=_number(
            payload["effective_anchor_count"], field="effective_anchor_count"
        ),
        historical_coverage_fraction=_number(
            payload["historical_coverage_fraction"],
            field="historical_coverage_fraction",
        ),
        execution_summaries=execution,
        all_required_adverse_passed=_boolean(
            payload["all_required_adverse_passed"], field="all_required_adverse_passed"
        ),
        all_perfect_information_valid=_boolean(
            payload["all_perfect_information_valid"],
            field="all_perfect_information_valid",
        ),
        failure_reasons=_strings(payload["failure_reasons"], field="failure_reasons"),
        bootstrap_resamples=_integer(
            payload["bootstrap_resamples"], field="bootstrap_resamples", minimum=1
        ),
        bootstrap_block_days=_integer(
            payload["bootstrap_block_days"], field="bootstrap_block_days", minimum=1
        ),
        summary_digest=_digest(payload["summary_digest"], field="summary_digest"),
        schema_version=_string(payload["schema_version"], field="schema_version"),
        production_status=_string(
            payload["production_status"], field="production_status"
        ),
    )
    if content_digest(summary.payload_without_digest()) != summary.summary_digest:
        raise ValueError("C3 aggregate summary digest mismatch")
    if canonical_json_bytes(summary.to_payload()) != raw_bytes:
        raise ValueError("C3 aggregate summary is not canonical JSON")
    return summary


def _condition(name: str, passed: bool, detail: str) -> GateConditionResult:
    return GateConditionResult(name=name, passed=passed, detail=detail)


def evaluate_phase_a_gate(
    summary: C3AggregateSummary,
    *,
    config: C3PhaseAGateConfig | None = None,
) -> PhaseAGateEvidence:
    """Evaluate the approved nine Phase A conditions without external access."""

    if not isinstance(summary, C3AggregateSummary):
        raise TypeError("summary must be C3AggregateSummary")
    resolved = C3PhaseAGateConfig() if config is None else config
    if not isinstance(resolved, C3PhaseAGateConfig):
        raise TypeError("config must be C3PhaseAGateConfig")
    diagnostics_complete = (
        bool(summary.calibration_buckets)
        and bool(summary.execution_summaries)
        and summary.unique_anchor_count > 0
        and summary.effective_anchor_count > 0.0
        and summary.historical_coverage_fraction > 0.0
    )
    integrity = (
        not summary.failure_reasons
        and all(not fold.failure_reasons for fold in summary.folds)
        and diagnostics_complete
    )
    support = (
        summary.fold_count >= resolved.required_folds
        and summary.total_selection_days >= resolved.required_selection_days
        and summary.total_effective_days >= resolved.required_selection_days
    )
    positive_folds = summary.positive_uplift_folds >= min(
        resolved.required_positive_folds, resolved.required_folds
    )
    uplift = summary.uplift_lower_ci > 0.0
    drawdown = (
        summary.worst_scenario_oracle_drawdown <= resolved.max_oracle_drawdown
        and summary.worst_scenario_oracle_drawdown
        <= summary.worst_trend_drawdown + resolved.trend_drawdown_tolerance
    )
    regret = summary.regret_margin_lower_ci > 0.0
    ranking = summary.mean_spearman > 0.0 and summary.spearman_lower_ci > 0.0
    perfect = summary.all_perfect_information_valid
    scenarios = summary.execution_scenario_names
    adverse = tuple(name for name in scenarios if name != "nominal")
    adverse_passed = (
        summary.all_required_adverse_passed and "nominal" in scenarios and bool(adverse)
    )
    conditions = (
        _condition(
            "integrity_and_determinism",
            integrity,
            "evidence and diagnostics are complete"
            if integrity
            else (
                f"aggregate_failures={summary.failure_reasons}; "
                f"fold_failures={tuple(fold.fold_id for fold in summary.folds if fold.failure_reasons)}; "
                f"diagnostics_complete={diagnostics_complete}"
            ),
        ),
        _condition(
            "fold_and_day_support",
            support,
            (
                f"folds={summary.fold_count}/{resolved.required_folds}; "
                f"selection_days={summary.total_selection_days}/{resolved.required_selection_days}; "
                f"effective_days={summary.total_effective_days}/{resolved.required_selection_days}"
            ),
        ),
        _condition(
            "positive_uplift_folds",
            positive_folds,
            (
                f"positive={summary.positive_uplift_folds}; "
                f"required={min(resolved.required_positive_folds, resolved.required_folds)}"
            ),
        ),
        _condition(
            "aggregate_uplift_confidence",
            uplift,
            f"paired_95pct_lower={summary.uplift_lower_ci:.12g}",
        ),
        _condition(
            "worst_fold_drawdown",
            drawdown,
            (
                f"oracle={summary.worst_scenario_oracle_drawdown:.12g}; "
                f"trend={summary.worst_trend_drawdown:.12g}"
            ),
        ),
        _condition(
            "realized_regret_vs_random",
            regret,
            f"paired_95pct_lower={summary.regret_margin_lower_ci:.12g}",
        ),
        _condition(
            "predicted_realized_ranking",
            ranking,
            (
                f"mean={summary.mean_spearman:.12g}; "
                f"lower={summary.spearman_lower_ci:.12g}"
            ),
        ),
        _condition(
            "perfect_information_compatibility",
            perfect,
            "all asserted bounds are compatible and ordered"
            if perfect
            else "one or more bounds are incompatible or unordered",
        ),
        _condition(
            "required_adverse_execution",
            adverse_passed,
            (
                f"nominal=true; adverse={adverse}; all_passed=true"
                if adverse_passed
                else (
                    f"scenarios={scenarios}; adverse={adverse}; "
                    f"all_passed={summary.all_required_adverse_passed}"
                )
            ),
        ),
    )
    return PhaseAGateEvidence(
        report_digest=summary.summary_digest,
        config_digest=resolved.digest,
        conditions=conditions,
        passed=all(item.passed for item in conditions),
    )


def render_c3_markdown(summary: C3AggregateSummary, gate: PhaseAGateEvidence) -> str:
    """Render a deterministic human-readable report from the strict read model."""

    if gate.report_digest != summary.summary_digest:
        raise ValueError("gate does not bind the supplied summary")
    lines = [
        "# Causal Scenario C3 Evaluation Report",
        "",
        f"Summary digest: `{summary.summary_digest}`",
        f"Source run digest: `{summary.source_run_digest}`",
        f"Core report digest: `{summary.core_report_digest}`",
        f"Config digest: `{summary.config_digest}`",
        f"Production status: **{summary.production_status}**",
        f"Phase A gate: **{'PASS' if gate.passed else 'BLOCKED'}**",
        "",
        "## Aggregate evidence",
        "",
        f"- Fold support: {summary.fold_count}",
        f"- Selection/effective days: {summary.total_selection_days}/{summary.total_effective_days}",
        f"- Positive uplift folds: {summary.positive_uplift_folds}",
        (
            f"- Uplift: {summary.mean_uplift:.12g} "
            f"[{summary.uplift_lower_ci:.12g}, {summary.uplift_upper_ci:.12g}], "
            f"p={summary.uplift_p_value:.12g}"
        ),
        (
            f"- Predicted-realized Spearman: {summary.mean_spearman:.12g} "
            f"[{summary.spearman_lower_ci:.12g}, {summary.spearman_upper_ci:.12g}]"
        ),
        (
            f"- Regret improvement: {summary.mean_regret_margin:.12g} "
            f"[{summary.regret_margin_lower_ci:.12g}, "
            f"{summary.regret_margin_upper_ci:.12g}]"
        ),
        (
            f"- Worst drawdown, scenario oracle/trend: "
            f"{summary.worst_scenario_oracle_drawdown:.12g}/"
            f"{summary.worst_trend_drawdown:.12g}"
        ),
        "",
        "## Fold evidence",
        "",
        "| Fold | Days | Uplift | Spearman | Regret margin | Adverse | Perfect info |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for fold in summary.folds:
        lines.append(
            f"| {fold.fold_id} | {fold.effective_days}/{fold.selection_days} | "
            f"{fold.mean_uplift:.12g} | {fold.mean_spearman:.12g} | "
            f"{fold.mean_regret_margin:.12g} | "
            f"{'pass' if fold.required_adverse_passed else 'fail'} | "
            f"{'valid' if fold.perfect_information_valid else 'invalid'} |"
        )
    lines.extend(
        [
            "",
            "## Execution scenarios",
            "",
            "| Scenario | Policy | Observations | Gross return | Economic cost | Fill ratio | Max DD |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary.execution_summaries:
        lines.append(
            f"| {item.execution_scenario} | {item.policy_kind} | "
            f"{item.observation_count} | {item.mean_gross_log_return:.12g} | "
            f"{item.mean_total_economic_cost:.12g} | {item.mean_fill_ratio:.12g} | "
            f"{item.maximum_drawdown:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            (
                f"- Neighbor distance p50/p90/p99: {summary.neighbor_distance_p50:.12g}/"
                f"{summary.neighbor_distance_p90:.12g}/{summary.neighbor_distance_p99:.12g}"
            ),
            (
                f"- Anchors unique/effective/max-share: {summary.unique_anchor_count}/"
                f"{summary.effective_anchor_count:.12g}/{summary.anchor_max_share:.12g}"
            ),
            f"- Historical coverage: {summary.historical_coverage_fraction:.12g}",
            f"- Calibration buckets: {len(summary.calibration_buckets)}",
            "",
            "## Phase A conditions",
            "",
        ]
    )
    lines.extend(
        f"- [{'x' if item.passed else ' '}] `{item.name}` — {item.detail}"
        for item in gate.conditions
    )
    lines.extend(
        [
            "",
            "## Failure reasons",
            "",
            *(f"- {reason}" for reason in summary.failure_reasons),
        ]
    )
    if not summary.failure_reasons:
        lines.append("- None")
    lines.extend(
        [
            "",
            "A passing Phase A gate authorizes only the next evaluation phase and does not authorize production.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LoadedC3ReportArtifact:
    summary: C3AggregateSummary
    gate: PhaseAGateEvidence
    artifact_digest: str
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_digest",
            _digest(self.artifact_digest, field="artifact_digest"),
        )
        object.__setattr__(self, "root", Path(self.root))


@dataclass(frozen=True, slots=True)
class LoadedPhaseAGateArtifact:
    gate: PhaseAGateEvidence
    report_artifact_digest: str
    artifact_digest: str
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_artifact_digest",
            _digest(self.report_artifact_digest, field="report_artifact_digest"),
        )
        object.__setattr__(
            self,
            "artifact_digest",
            _digest(self.artifact_digest, field="artifact_digest"),
        )
        object.__setattr__(self, "root", Path(self.root))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_closure(root: Path, expected: frozenset[str], *, label: str) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"{label} artifact directory is missing: {root}")
    names: set[str] = set()
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"{label} artifact contains an invalid file entry")
        names.add(entry.name)
    if names != set(expected):
        raise ValueError(f"{label} artifact file closure mismatch")


def _publish_exact(root: Path, expected: Mapping[str, bytes], *, label: str) -> None:
    if root.exists() and any(root.iterdir()):
        actual_names = {entry.name for entry in root.iterdir()}
        if actual_names != set(expected) or any(
            not (root / name).is_file()
            or (root / name).is_symlink()
            or (root / name).read_bytes() != payload
            for name, payload in expected.items()
        ):
            raise FileExistsError(
                f"conflicting {label} artifact already exists: {root}"
            )
        return
    root.mkdir(parents=True, exist_ok=True)
    for name in sorted(expected):
        _atomic_write(root / name, expected[name])


def _report_bytes(
    summary: C3AggregateSummary,
    gate: PhaseAGateEvidence,
) -> tuple[dict[str, bytes], str]:
    summary_bytes = canonical_json_bytes(summary.to_payload())
    markdown_bytes = render_c3_markdown(summary, gate).encode("utf-8")
    base_manifest: dict[str, object] = {
        "config_digest": summary.config_digest,
        "core_report_digest": summary.core_report_digest,
        "files": {
            "report.md": _sha256(markdown_bytes),
            "summary.json": _sha256(summary_bytes),
        },
        "gate_digest": gate.digest,
        "production_status": PRODUCTION_STATUS,
        "schema_version": C3_REPORT_ARTIFACT_SCHEMA,
        "source_run_digest": summary.source_run_digest,
        "summary_digest": summary.summary_digest,
    }
    artifact_digest = content_digest(base_manifest)
    manifest = dict(base_manifest)
    manifest["artifact_digest"] = artifact_digest
    return {
        "manifest.json": canonical_json_bytes(manifest),
        "report.md": markdown_bytes,
        "summary.json": summary_bytes,
    }, artifact_digest


def write_c3_report_artifact(
    root: str | Path,
    summary: C3AggregateSummary,
    gate: PhaseAGateEvidence,
) -> LoadedC3ReportArtifact:
    if gate.report_digest != summary.summary_digest:
        raise ValueError("gate does not bind the supplied summary")
    destination = Path(root)
    expected, _ = _report_bytes(summary, gate)
    _publish_exact(destination, expected, label="C3 report")
    return load_c3_report_artifact(destination)


def load_c3_report_artifact(root: str | Path) -> LoadedC3ReportArtifact:
    source = Path(root)
    _verify_closure(source, _REPORT_FILES, label="C3 report")
    manifest_bytes = (source / "manifest.json").read_bytes()
    try:
        manifest = _mapping(
            json.loads(manifest_bytes.decode("utf-8")), field="manifest"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("C3 report artifact manifest is invalid") from error
    _require_fields(
        manifest,
        {
            "artifact_digest",
            "config_digest",
            "core_report_digest",
            "files",
            "gate_digest",
            "production_status",
            "schema_version",
            "source_run_digest",
            "summary_digest",
        },
        field="manifest",
    )
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("C3 report artifact manifest is not canonical JSON")
    if manifest["schema_version"] != C3_REPORT_ARTIFACT_SCHEMA:
        raise ValueError("unsupported C3 report artifact schema")
    if manifest["production_status"] != PRODUCTION_STATUS:
        raise ValueError("C3 report artifact production status must remain NO-GO")
    artifact_digest = _digest(manifest["artifact_digest"], field="artifact_digest")
    base = dict(manifest)
    del base["artifact_digest"]
    if content_digest(base) != artifact_digest:
        raise ValueError("C3 report artifact digest mismatch")
    files = _mapping(manifest["files"], field="manifest.files")
    _require_fields(files, {"report.md", "summary.json"}, field="manifest.files")
    for name in ("report.md", "summary.json"):
        if _sha256((source / name).read_bytes()) != _digest(
            files[name], field=f"manifest.files.{name}"
        ):
            raise ValueError("C3 report artifact file digest mismatch")
    summary = load_c3_aggregate_summary(source / "summary.json")
    if summary.summary_digest != _digest(
        manifest["summary_digest"], field="summary_digest"
    ):
        raise ValueError("C3 report summary identity mismatch")
    if summary.source_run_digest != _digest(
        manifest["source_run_digest"], field="source_run_digest"
    ):
        raise ValueError("C3 report source run identity mismatch")
    if summary.core_report_digest != _digest(
        manifest["core_report_digest"], field="core_report_digest"
    ):
        raise ValueError("C3 report core report identity mismatch")
    if summary.config_digest != _digest(
        manifest["config_digest"], field="config_digest"
    ):
        raise ValueError("C3 report config identity mismatch")
    gate = evaluate_phase_a_gate(summary)
    if gate.digest != _digest(manifest["gate_digest"], field="gate_digest"):
        raise ValueError("C3 report gate identity mismatch")
    if (source / "report.md").read_text(encoding="utf-8") != render_c3_markdown(
        summary, gate
    ):
        raise ValueError("C3 report Markdown does not match evidence")
    return LoadedC3ReportArtifact(
        summary=summary,
        gate=gate,
        artifact_digest=artifact_digest,
        root=source,
    )


def _gate_from_payload(payload: Mapping[str, object]) -> PhaseAGateEvidence:
    _require_fields(
        payload,
        {
            "conditions",
            "config_digest",
            "gate_digest",
            "passed",
            "production_status",
            "report_digest",
            "schema_version",
        },
        field="gate",
    )
    conditions: list[GateConditionResult] = []
    for index, raw in enumerate(
        _sequence(payload["conditions"], field="gate.conditions")
    ):
        item = _mapping(raw, field=f"gate.conditions[{index}]")
        _require_fields(
            item, {"detail", "name", "passed"}, field=f"gate.conditions[{index}]"
        )
        conditions.append(
            GateConditionResult(
                name=_string(item["name"], field=f"gate.conditions[{index}].name"),
                passed=_boolean(
                    item["passed"], field=f"gate.conditions[{index}].passed"
                ),
                detail=_string(
                    item["detail"], field=f"gate.conditions[{index}].detail"
                ),
            )
        )
    gate = PhaseAGateEvidence(
        report_digest=_digest(payload["report_digest"], field="gate.report_digest"),
        config_digest=_digest(payload["config_digest"], field="gate.config_digest"),
        conditions=tuple(conditions),
        passed=_boolean(payload["passed"], field="gate.passed"),
        schema_version=_string(payload["schema_version"], field="gate.schema_version"),
        production_status=_string(
            payload["production_status"], field="gate.production_status"
        ),
    )
    if gate.digest != _digest(payload["gate_digest"], field="gate.gate_digest"):
        raise ValueError("Phase A gate digest mismatch")
    return gate


def _gate_bytes(
    gate: PhaseAGateEvidence,
    *,
    report_artifact_digest: str,
) -> tuple[dict[str, bytes], str]:
    report_digest = _digest(report_artifact_digest, field="report_artifact_digest")
    gate_bytes = canonical_json_bytes(gate.to_payload())
    base_manifest: dict[str, object] = {
        "file_digest": _sha256(gate_bytes),
        "gate_digest": gate.digest,
        "production_status": PRODUCTION_STATUS,
        "report_artifact_digest": report_digest,
        "schema_version": C3_GATE_ARTIFACT_SCHEMA,
    }
    artifact_digest = content_digest(base_manifest)
    manifest = dict(base_manifest)
    manifest["artifact_digest"] = artifact_digest
    return {
        "gate.json": gate_bytes,
        "manifest.json": canonical_json_bytes(manifest),
    }, artifact_digest


def write_phase_a_gate_artifact(
    root: str | Path,
    gate: PhaseAGateEvidence,
    *,
    report_artifact_digest: str,
) -> LoadedPhaseAGateArtifact:
    destination = Path(root)
    expected, _ = _gate_bytes(gate, report_artifact_digest=report_artifact_digest)
    _publish_exact(destination, expected, label="Phase A gate")
    return load_phase_a_gate_artifact(destination)


def load_phase_a_gate_artifact(root: str | Path) -> LoadedPhaseAGateArtifact:
    source = Path(root)
    _verify_closure(source, _GATE_FILES, label="Phase A gate")
    manifest_bytes = (source / "manifest.json").read_bytes()
    gate_bytes = (source / "gate.json").read_bytes()
    try:
        manifest = _mapping(
            json.loads(manifest_bytes.decode("utf-8")), field="manifest"
        )
        gate_payload = _mapping(json.loads(gate_bytes.decode("utf-8")), field="gate")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Phase A gate artifact is invalid JSON") from error
    _require_fields(
        manifest,
        {
            "artifact_digest",
            "file_digest",
            "gate_digest",
            "production_status",
            "report_artifact_digest",
            "schema_version",
        },
        field="manifest",
    )
    if (
        canonical_json_bytes(manifest) != manifest_bytes
        or canonical_json_bytes(gate_payload) != gate_bytes
    ):
        raise ValueError("Phase A gate artifact is not canonical JSON")
    if manifest["schema_version"] != C3_GATE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported Phase A gate artifact schema")
    if manifest["production_status"] != PRODUCTION_STATUS:
        raise ValueError("Phase A gate artifact production status must remain NO-GO")
    artifact_digest = _digest(manifest["artifact_digest"], field="artifact_digest")
    base = dict(manifest)
    del base["artifact_digest"]
    if content_digest(base) != artifact_digest:
        raise ValueError("Phase A gate artifact digest mismatch")
    if _sha256(gate_bytes) != _digest(manifest["file_digest"], field="file_digest"):
        raise ValueError("Phase A gate artifact file digest mismatch")
    gate = _gate_from_payload(gate_payload)
    if gate.digest != _digest(manifest["gate_digest"], field="gate_digest"):
        raise ValueError("Phase A gate manifest identity mismatch")
    return LoadedPhaseAGateArtifact(
        gate=gate,
        report_artifact_digest=_digest(
            manifest["report_artifact_digest"], field="report_artifact_digest"
        ),
        artifact_digest=artifact_digest,
        root=source,
    )


__all__ = [
    "C3AggregateSummary",
    "C3CalibrationBucketSummary",
    "C3ExecutionSummary",
    "C3FoldSummary",
    "C3PhaseAGateConfig",
    "GateConditionResult",
    "LoadedC3ReportArtifact",
    "LoadedPhaseAGateArtifact",
    "PhaseAGateEvidence",
    "evaluate_phase_a_gate",
    "load_c3_aggregate_summary",
    "load_c3_report_artifact",
    "load_phase_a_gate_artifact",
    "render_c3_markdown",
    "write_c3_report_artifact",
    "write_phase_a_gate_artifact",
]
