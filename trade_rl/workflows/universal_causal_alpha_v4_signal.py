"""Non-promotable signal diagnostics for Causal Alpha V4."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from statistics import fmean
from typing import Any, Final, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.bootstrap import moving_block_mean_test
from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.learning.causal_alpha_v4 import CausalAlphaV4Forecast
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    non_overlapping_causal_alpha_v3_rows,
)

CAUSAL_ALPHA_V4_LIVENESS_SCHEMA: Final = "causal_alpha_v4_liveness_evidence_v1"
CAUSAL_ALPHA_V4_SIGNAL_SCOPE_SCHEMA: Final = "causal_alpha_v4_signal_scope_v1"
CAUSAL_ALPHA_V4_SIGNAL_LANE_SCHEMA: Final = "causal_alpha_v4_signal_lane_evidence_v1"
CAUSAL_ALPHA_V4_SIGNAL_EVIDENCE_SCHEMA: Final = "causal_alpha_v4_signal_evidence_v1"
CAUSAL_ALPHA_V4_SIGNAL_BOOTSTRAP_SCHEMA: Final = "causal_alpha_v4_signal_bootstrap_v1"
_LIVENESS_TOLERANCE: Final = 1e-12
_CONTRIBUTION_FAMILIES: Final = (
    "existing_15m",
    "existing_1h",
    "existing_4h",
    "existing_1d",
    "local_cross_market",
    "global_market",
    "beta_scaled_proxy",
    "shared_residual",
)
_QUANTILES = np.asarray((0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0), dtype=np.float64)


def _finite_vector(value: object, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1).copy(order="C")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    array.setflags(write=False)
    return array


def _near_identical_run_lengths(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.asarray([1], dtype=np.int64)
    lengths: list[int] = []
    current = 1
    for difference in np.abs(np.diff(values)):
        if difference <= _LIVENESS_TOLERANCE:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return np.asarray(lengths, dtype=np.int64)


def _unique_count_at_tolerance(values: np.ndarray) -> int:
    ordered = np.sort(values, kind="mergesort")
    if ordered.size == 1:
        return 1
    return 1 + int(np.count_nonzero(np.diff(ordered) > _LIVENESS_TOLERANCE))


@dataclass(frozen=True, slots=True)
class CausalAlphaV4LivenessEvidence:
    fit_digest: str
    forecast_digest: str
    symbol: str
    horizon: str
    prediction_mean: float
    prediction_std: float
    prediction_min: float
    prediction_max: float
    prediction_quantiles: tuple[float, ...]
    unique_count_at_tolerance_1e_12: int
    median_near_identical_run_length: float
    maximum_near_identical_run_length: int
    intercept: float
    dynamic_prediction_std: float
    weighted_final_rmse: float
    dynamic_to_rmse_ratio: float | None
    constant_feature_count: int
    available_feature_count: int
    contribution_variance_existing_15m: float
    contribution_variance_existing_1h: float
    contribution_variance_existing_4h: float
    contribution_variance_existing_1d: float
    contribution_variance_local_cross_market: float
    contribution_variance_global_market: float
    contribution_variance_beta_scaled_proxy: float
    contribution_variance_shared_residual: float
    direction_score_mean: float
    direction_score_std: float
    direction_positive_fraction: float
    direction_negative_fraction: float
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V4_LIVENESS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.fit_digest, field="V4 liveness fit_digest")
        require_sha256(self.forecast_digest, field="V4 liveness forecast_digest")
        if not self.symbol or self.horizon not in {"4h", "24h", "72h"}:
            raise ValueError("V4 liveness identity is invalid")
        numeric = (
            self.prediction_mean,
            self.prediction_std,
            self.prediction_min,
            self.prediction_max,
            self.median_near_identical_run_length,
            self.intercept,
            self.dynamic_prediction_std,
            self.weighted_final_rmse,
            self.contribution_variance_existing_15m,
            self.contribution_variance_existing_1h,
            self.contribution_variance_existing_4h,
            self.contribution_variance_existing_1d,
            self.contribution_variance_local_cross_market,
            self.contribution_variance_global_market,
            self.contribution_variance_beta_scaled_proxy,
            self.contribution_variance_shared_residual,
            self.direction_score_mean,
            self.direction_score_std,
            self.direction_positive_fraction,
            self.direction_negative_fraction,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("V4 liveness metrics must be finite")
        if self.dynamic_to_rmse_ratio is not None and (
            not math.isfinite(self.dynamic_to_rmse_ratio)
            or self.dynamic_to_rmse_ratio < 0.0
        ):
            raise ValueError("V4 liveness dynamic/RMSE ratio is invalid")
        if len(self.prediction_quantiles) != len(_QUANTILES) or not all(
            math.isfinite(value) for value in self.prediction_quantiles
        ):
            raise ValueError("V4 liveness prediction quantiles are invalid")
        for value in (
            self.prediction_std,
            self.median_near_identical_run_length,
            self.dynamic_prediction_std,
            self.weighted_final_rmse,
            self.contribution_variance_existing_15m,
            self.contribution_variance_existing_1h,
            self.contribution_variance_existing_4h,
            self.contribution_variance_existing_1d,
            self.contribution_variance_local_cross_market,
            self.contribution_variance_global_market,
            self.contribution_variance_beta_scaled_proxy,
            self.contribution_variance_shared_residual,
            self.direction_score_std,
            self.direction_positive_fraction,
            self.direction_negative_fraction,
        ):
            if value < 0.0:
                raise ValueError("V4 liveness non-negative metric became negative")
        if (
            not 0.0 <= self.direction_positive_fraction <= 1.0
            or not 0.0 <= self.direction_negative_fraction <= 1.0
        ):
            raise ValueError("V4 liveness direction fractions are invalid")
        for value in (
            self.unique_count_at_tolerance_1e_12,
            self.maximum_near_identical_run_length,
            self.constant_feature_count,
            self.available_feature_count,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("V4 liveness counts must be non-negative integers")
        if (
            self.unique_count_at_tolerance_1e_12 <= 0
            or self.maximum_near_identical_run_length <= 0
        ):
            raise ValueError("V4 liveness prediction support must be positive")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V4 liveness evidence must remain non-promotable")
        if self.schema_version != CAUSAL_ALPHA_V4_LIVENESS_SCHEMA:
            raise ValueError("unsupported V4 liveness schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 liveness digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "available_feature_count": self.available_feature_count,
            "constant_feature_count": self.constant_feature_count,
            "contribution_variance_beta_scaled_proxy": self.contribution_variance_beta_scaled_proxy,
            "contribution_variance_existing_15m": self.contribution_variance_existing_15m,
            "contribution_variance_existing_1d": self.contribution_variance_existing_1d,
            "contribution_variance_existing_1h": self.contribution_variance_existing_1h,
            "contribution_variance_existing_4h": self.contribution_variance_existing_4h,
            "contribution_variance_global_market": self.contribution_variance_global_market,
            "contribution_variance_local_cross_market": self.contribution_variance_local_cross_market,
            "contribution_variance_shared_residual": self.contribution_variance_shared_residual,
            "direction_negative_fraction": self.direction_negative_fraction,
            "direction_positive_fraction": self.direction_positive_fraction,
            "direction_score_mean": self.direction_score_mean,
            "direction_score_std": self.direction_score_std,
            "dynamic_prediction_std": self.dynamic_prediction_std,
            "dynamic_to_rmse_ratio": self.dynamic_to_rmse_ratio,
            "fit_digest": self.fit_digest,
            "forecast_digest": self.forecast_digest,
            "horizon": self.horizon,
            "intercept": self.intercept,
            "maximum_near_identical_run_length": self.maximum_near_identical_run_length,
            "median_near_identical_run_length": self.median_near_identical_run_length,
            "prediction_max": self.prediction_max,
            "prediction_mean": self.prediction_mean,
            "prediction_min": self.prediction_min,
            "prediction_quantiles": self.prediction_quantiles,
            "prediction_std": self.prediction_std,
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "unique_count_at_tolerance_1e_12": self.unique_count_at_tolerance_1e_12,
            "weighted_final_rmse": self.weighted_final_rmse,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_causal_alpha_v4_liveness_evidence(
    *,
    fit_digest: str,
    forecast_digest: str,
    symbol: str,
    horizon: str,
    prediction: object,
    direction_score: object,
    intercept: float,
    weighted_final_rmse: float,
    feature_available: object,
    constant_feature_mask: object,
    contribution_series: Mapping[str, object],
) -> CausalAlphaV4LivenessEvidence:
    """Build descriptive V4 signal-liveness evidence without changing a gate."""

    predicted = _finite_vector(prediction, field="V4 liveness prediction")
    direction = _finite_vector(direction_score, field="V4 liveness direction score")
    if direction.shape != predicted.shape:
        raise ValueError("V4 liveness prediction and direction score must align")
    if not math.isfinite(intercept):
        raise ValueError("V4 liveness intercept must be finite")
    if not math.isfinite(weighted_final_rmse) or weighted_final_rmse < 0.0:
        raise ValueError("V4 liveness weighted final RMSE must be non-negative")

    availability = np.asarray(feature_available, dtype=np.bool_)
    constant_mask = np.asarray(constant_feature_mask, dtype=np.bool_).reshape(-1)
    if availability.ndim != 2 or availability.shape[0] != predicted.size:
        raise ValueError("V4 liveness feature availability must be row aligned")
    if constant_mask.shape != (availability.shape[1],):
        raise ValueError("V4 liveness constant mask must match feature width")
    available_columns = np.any(availability, axis=0)
    available_feature_count = int(np.count_nonzero(available_columns))
    constant_feature_count = int(np.count_nonzero(constant_mask))
    supported_dynamic_feature_count = int(
        np.count_nonzero(available_columns & ~constant_mask)
    )

    dynamic = predicted - float(intercept)
    dynamic_std = float(np.std(dynamic))
    if dynamic_std == 0.0 and supported_dynamic_feature_count >= 2:
        raise ValueError(
            "V4 dynamic prediction is exactly zero despite live non-constant features"
        )

    if tuple(contribution_series) != _CONTRIBUTION_FAMILIES:
        raise ValueError(
            "V4 liveness contribution families are incomplete or reordered"
        )
    contribution_variance: dict[str, float] = {}
    for family in _CONTRIBUTION_FAMILIES:
        values = _finite_vector(
            contribution_series[family],
            field=f"V4 liveness contribution {family}",
        )
        if values.shape != predicted.shape:
            raise ValueError("V4 liveness contribution series must align")
        contribution_variance[family] = float(np.var(values))

    runs = _near_identical_run_lengths(predicted)
    quantiles = tuple(float(value) for value in np.quantile(predicted, _QUANTILES))
    dynamic_to_rmse = (
        None if weighted_final_rmse == 0.0 else dynamic_std / weighted_final_rmse
    )
    return CausalAlphaV4LivenessEvidence(
        fit_digest=fit_digest,
        forecast_digest=forecast_digest,
        symbol=symbol,
        horizon=horizon,
        prediction_mean=float(np.mean(predicted)),
        prediction_std=float(np.std(predicted)),
        prediction_min=float(np.min(predicted)),
        prediction_max=float(np.max(predicted)),
        prediction_quantiles=quantiles,
        unique_count_at_tolerance_1e_12=_unique_count_at_tolerance(predicted),
        median_near_identical_run_length=float(np.median(runs)),
        maximum_near_identical_run_length=int(np.max(runs)),
        intercept=float(intercept),
        dynamic_prediction_std=dynamic_std,
        weighted_final_rmse=float(weighted_final_rmse),
        dynamic_to_rmse_ratio=dynamic_to_rmse,
        constant_feature_count=constant_feature_count,
        available_feature_count=available_feature_count,
        contribution_variance_existing_15m=contribution_variance["existing_15m"],
        contribution_variance_existing_1h=contribution_variance["existing_1h"],
        contribution_variance_existing_4h=contribution_variance["existing_4h"],
        contribution_variance_existing_1d=contribution_variance["existing_1d"],
        contribution_variance_local_cross_market=contribution_variance[
            "local_cross_market"
        ],
        contribution_variance_global_market=contribution_variance["global_market"],
        contribution_variance_beta_scaled_proxy=contribution_variance[
            "beta_scaled_proxy"
        ],
        contribution_variance_shared_residual=contribution_variance["shared_residual"],
        direction_score_mean=float(np.mean(direction)),
        direction_score_std=float(np.std(direction)),
        direction_positive_fraction=float(np.mean(direction > 0.0)),
        direction_negative_fraction=float(np.mean(direction < 0.0)),
    )


class CausalAlphaV4SignalLane(str, Enum):
    FAST_4H = "fast_4h"
    SLOW_FUSED = "slow_fused"


_V4_SIGNAL_LANES: Final = (
    CausalAlphaV4SignalLane.FAST_4H,
    CausalAlphaV4SignalLane.SLOW_FUSED,
)


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SignalGateConfig:
    independent_episode_count: int = 8
    minimum_rank_ic_lower_ci: float = 0.0
    minimum_top_bottom_spread_lower_ci: float = 0.0
    minimum_direction_accuracy_excess_lower_ci: float = 0.0
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 20260823
    bootstrap_block_size: int = 2

    def __post_init__(self) -> None:
        if self.independent_episode_count != 8:
            raise ValueError("V4 signal independent episode count must remain 8")
        if self.minimum_rank_ic_lower_ci != 0.0:
            raise ValueError("V4 signal rank lower bound must remain zero")
        if self.minimum_top_bottom_spread_lower_ci != 0.0:
            raise ValueError("V4 signal spread lower bound must remain zero")
        if self.minimum_direction_accuracy_excess_lower_ci != 0.0:
            raise ValueError("V4 signal direction lower bound must remain zero")
        if self.bootstrap_resamples != 10000:
            raise ValueError("V4 signal bootstrap resamples must remain 10000")
        if self.bootstrap_seed != 20260823:
            raise ValueError("V4 signal bootstrap seed must remain frozen")
        if self.bootstrap_block_size != 2:
            raise ValueError("V4 signal bootstrap block size must remain 2")

    @classmethod
    def from_mapping(cls, raw: object) -> "CausalAlphaV4SignalGateConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("V4 signal gate config must be an object")
        values = dict(raw)
        expected = {
            "independent_episode_count",
            "minimum_rank_ic_lower_ci",
            "minimum_top_bottom_spread_lower_ci",
            "minimum_direction_accuracy_excess_lower_ci",
            "bootstrap_resamples",
            "bootstrap_seed",
            "bootstrap_block_size",
        }
        if set(values) != expected:
            missing = sorted(expected - set(values))
            unknown = sorted(set(values) - expected)
            raise ValueError(
                f"V4 signal gate fields mismatch; missing={missing}, unknown={unknown}"
            )
        return cls(
            independent_episode_count=int(values["independent_episode_count"]),
            minimum_rank_ic_lower_ci=float(values["minimum_rank_ic_lower_ci"]),
            minimum_top_bottom_spread_lower_ci=float(
                values["minimum_top_bottom_spread_lower_ci"]
            ),
            minimum_direction_accuracy_excess_lower_ci=float(
                values["minimum_direction_accuracy_excess_lower_ci"]
            ),
            bootstrap_resamples=int(values["bootstrap_resamples"]),
            bootstrap_seed=int(values["bootstrap_seed"]),
            bootstrap_block_size=int(values["bootstrap_block_size"]),
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "bootstrap_block_size": self.bootstrap_block_size,
                "bootstrap_resamples": self.bootstrap_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "independent_episode_count": self.independent_episode_count,
                "minimum_direction_accuracy_excess_lower_ci": (
                    self.minimum_direction_accuracy_excess_lower_ci
                ),
                "minimum_rank_ic_lower_ci": self.minimum_rank_ic_lower_ci,
                "minimum_top_bottom_spread_lower_ci": (
                    self.minimum_top_bottom_spread_lower_ci
                ),
                "schema_version": "causal_alpha_v4_signal_gate_config_v1",
            }
        )


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SignalScopeMetric:
    run_manifest_digest: str
    fit_config_digest: str
    lane: CausalAlphaV4SignalLane
    symbol: str
    episode_index: int
    contract_start: int
    contract_stop: int
    contract_digest: str
    fit_digest: str
    forecast_digest: str
    liveness_digest: str
    sample_count: int
    direction_sample_count: int
    rank_correlation: float
    direction_accuracy: float
    top_bottom_realized_spread: float
    cohort_indices: tuple[int, ...]
    schema_version: str = CAUSAL_ALPHA_V4_SIGNAL_SCOPE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "run_manifest_digest",
            "fit_config_digest",
            "contract_digest",
            "fit_digest",
            "forecast_digest",
            "liveness_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"V4 signal {field_name}")
        lane = CausalAlphaV4SignalLane(self.lane)
        if not self.symbol:
            raise ValueError("V4 signal symbol must be non-empty")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
        ):
            raise ValueError("V4 signal episode index must be non-negative")
        if (
            isinstance(self.contract_start, bool)
            or not isinstance(self.contract_start, int)
            or isinstance(self.contract_stop, bool)
            or not isinstance(self.contract_stop, int)
            or self.contract_start < 0
            or self.contract_stop <= self.contract_start
        ):
            raise ValueError("V4 signal contract interval is invalid")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 2
        ):
            raise ValueError("V4 signal scope requires at least two samples")
        if (
            isinstance(self.direction_sample_count, bool)
            or not isinstance(self.direction_sample_count, int)
            or not 1 <= self.direction_sample_count <= self.sample_count
        ):
            raise ValueError("V4 signal direction support is invalid")
        if (
            not math.isfinite(self.rank_correlation)
            or not -1.0 <= self.rank_correlation <= 1.0
        ):
            raise ValueError("V4 signal rank correlation is invalid")
        if (
            not math.isfinite(self.direction_accuracy)
            or not 0.0 <= self.direction_accuracy <= 1.0
        ):
            raise ValueError("V4 signal direction accuracy is invalid")
        if not math.isfinite(self.top_bottom_realized_spread):
            raise ValueError("V4 signal top-bottom spread must be finite")
        cohort = tuple(self.cohort_indices)
        if (
            len(cohort) != self.sample_count
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in cohort
            )
            or tuple(sorted(set(cohort))) != cohort
        ):
            raise ValueError("V4 signal cohort indices are invalid")
        if self.schema_version != CAUSAL_ALPHA_V4_SIGNAL_SCOPE_SCHEMA:
            raise ValueError("unsupported V4 signal scope schema")
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "cohort_indices", cohort)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 signal scope digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, str, str, int]:
        return (
            self.lane.value,
            self.fit_config_digest,
            self.symbol,
            self.episode_index,
        )

    @property
    def cluster_identity(self) -> tuple[int, int]:
        return (self.contract_start, self.contract_stop)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "cohort_indices": self.cohort_indices,
            "contract_digest": self.contract_digest,
            "contract_start": self.contract_start,
            "contract_stop": self.contract_stop,
            "direction_accuracy": self.direction_accuracy,
            "direction_sample_count": self.direction_sample_count,
            "episode_index": self.episode_index,
            "fit_config_digest": self.fit_config_digest,
            "fit_digest": self.fit_digest,
            "forecast_digest": self.forecast_digest,
            "lane": self.lane.value,
            "liveness_digest": self.liveness_digest,
            "rank_correlation": self.rank_correlation,
            "run_manifest_digest": self.run_manifest_digest,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "top_bottom_realized_spread": self.top_bottom_realized_spread,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SignalBootstrapEvidence:
    mean: float
    p_value: float
    lower_ci: float
    upper_ci: float
    block_size: int
    schema_version: str = CAUSAL_ALPHA_V4_SIGNAL_BOOTSTRAP_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.mean, self.p_value, self.lower_ci, self.upper_ci)
        ):
            raise ValueError("V4 signal bootstrap values must be finite")
        if not 0.0 <= self.p_value <= 1.0 or self.lower_ci > self.upper_ci:
            raise ValueError("V4 signal bootstrap interval/probability is invalid")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("V4 signal bootstrap block size must be positive")
        if self.schema_version != CAUSAL_ALPHA_V4_SIGNAL_BOOTSTRAP_SCHEMA:
            raise ValueError("unsupported V4 signal bootstrap schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 signal bootstrap digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "block_size": self.block_size,
            "lower_ci": self.lower_ci,
            "mean": self.mean,
            "p_value": self.p_value,
            "schema_version": self.schema_version,
            "upper_ci": self.upper_ci,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV4LaneSignalEvidence:
    lane: CausalAlphaV4SignalLane
    metrics: tuple[CausalAlphaV4SignalScopeMetric, ...]
    run_manifest_digest: str
    raw_scope_count: int
    expected_raw_scope_count: int
    independent_episode_count: int
    rank_ic: CausalAlphaV4SignalBootstrapEvidence
    top_bottom_spread: CausalAlphaV4SignalBootstrapEvidence
    direction_accuracy_excess: CausalAlphaV4SignalBootstrapEvidence
    gate_digest: str
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V4_SIGNAL_LANE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        lane = CausalAlphaV4SignalLane(self.lane)
        metrics = tuple(self.metrics)
        if not metrics or any(metric.lane is not lane for metric in metrics):
            raise ValueError("V4 lane evidence metric scope is invalid")
        if len({metric.identity for metric in metrics}) != len(metrics):
            raise ValueError("V4 lane evidence contains duplicate metrics")
        require_sha256(self.run_manifest_digest, field="V4 signal run manifest digest")
        require_sha256(self.gate_digest, field="V4 signal gate digest")
        if {metric.run_manifest_digest for metric in metrics} != {
            self.run_manifest_digest
        }:
            raise ValueError("V4 lane evidence run identity drifted")
        if len({metric.fit_config_digest for metric in metrics}) != 1:
            raise ValueError("V4 lane evidence fit config drifted")
        if self.raw_scope_count != len(metrics) or self.raw_scope_count <= 0:
            raise ValueError("V4 lane raw scope count is invalid")
        if (
            self.expected_raw_scope_count <= 0
            or self.raw_scope_count > self.expected_raw_scope_count
        ):
            raise ValueError("V4 lane expected raw scope count is invalid")
        observed_clusters = len({metric.cluster_identity for metric in metrics})
        if (
            self.independent_episode_count != observed_clusters
            or self.independent_episode_count <= 0
        ):
            raise ValueError("V4 lane independent episode count is invalid")
        for field_name in ("rank_ic", "top_bottom_spread", "direction_accuracy_excess"):
            if not isinstance(
                getattr(self, field_name), CausalAlphaV4SignalBootstrapEvidence
            ):
                raise TypeError(f"V4 lane {field_name} evidence is invalid")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V4 lane pass state and rejection reasons disagree")
        if self.promotion_eligible:
            raise ValueError("V4 lane signal evidence cannot be promotion eligible")
        if self.schema_version != CAUSAL_ALPHA_V4_SIGNAL_LANE_SCHEMA:
            raise ValueError("unsupported V4 lane signal schema")
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 lane signal digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "direction_accuracy_excess_digest": self.direction_accuracy_excess.digest,
            "expected_raw_scope_count": self.expected_raw_scope_count,
            "gate_digest": self.gate_digest,
            "independent_episode_count": self.independent_episode_count,
            "lane": self.lane.value,
            "metric_digests": tuple(metric.digest for metric in self.metrics),
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "rank_ic_digest": self.rank_ic.digest,
            "raw_scope_count": self.raw_scope_count,
            "rejection_reasons": self.rejection_reasons,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "top_bottom_spread_digest": self.top_bottom_spread.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SignalEvidence:
    fast_4h: CausalAlphaV4LaneSignalEvidence
    slow_fused: CausalAlphaV4LaneSignalEvidence
    gate_digest: str
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V4_SIGNAL_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.fast_4h.lane is not CausalAlphaV4SignalLane.FAST_4H:
            raise ValueError("V4 fast signal evidence lane is invalid")
        if self.slow_fused.lane is not CausalAlphaV4SignalLane.SLOW_FUSED:
            raise ValueError("V4 slow signal evidence lane is invalid")
        require_sha256(self.gate_digest, field="V4 signal evidence gate digest")
        if (
            self.fast_4h.gate_digest != self.gate_digest
            or self.slow_fused.gate_digest != self.gate_digest
        ):
            raise ValueError("V4 signal evidence gate identity drifted")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V4 signal evidence pass state and reasons disagree")
        if self.promotion_eligible:
            raise ValueError("V4 signal evidence cannot be promotion eligible")
        if self.schema_version != CAUSAL_ALPHA_V4_SIGNAL_EVIDENCE_SCHEMA:
            raise ValueError("unsupported V4 signal evidence schema")
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 signal evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "fast_4h_digest": self.fast_4h.digest,
            "gate_digest": self.gate_digest,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "slow_fused_digest": self.slow_fused.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _signal_array(
    value: object, *, rows: int, field_name: str, dtype: Any
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"V4 signal {field_name} is not decision aligned")
    return array


def _v4_scope_metric(
    *,
    run_manifest_digest: str,
    fit_config_digest: str,
    lane: CausalAlphaV4SignalLane,
    symbol: str,
    episode_index: int,
    contract_start: int,
    contract_stop: int,
    contract_digest: str,
    fit_digest: str,
    forecast_digest: str,
    liveness_digest: str,
    decisions: np.ndarray,
    cohort_rows: np.ndarray,
    prediction: np.ndarray,
    realized: np.ndarray,
    direction_score: np.ndarray,
) -> CausalAlphaV4SignalScopeMetric:
    selected_prediction = prediction[cohort_rows]
    selected_realized = realized[cohort_rows]
    selected_direction = direction_score[cohort_rows]
    diagnostics = evaluate_causal_alpha_signal_diagnostics(
        selected_prediction, selected_realized
    )
    if diagnostics.rank_correlation is None:
        raise ValueError("V4 signal scope rank correlation is undefined")
    direction_mask = np.sign(selected_realized) != 0.0
    direction_support = int(np.count_nonzero(direction_mask))
    if direction_support == 0:
        raise ValueError("V4 signal scope has no non-zero direction support")
    direction_accuracy = float(
        np.mean(
            np.sign(selected_direction[direction_mask])
            == np.sign(selected_realized[direction_mask])
        )
    )
    order = np.argsort(selected_prediction, kind="mergesort")
    bucket = max(1, selected_prediction.size // 5)
    bottom = order[:bucket]
    top = order[-bucket:]
    spread = float(
        np.mean(selected_realized[top], dtype=np.float64)
        - np.mean(selected_realized[bottom], dtype=np.float64)
    )
    return CausalAlphaV4SignalScopeMetric(
        run_manifest_digest=run_manifest_digest,
        fit_config_digest=fit_config_digest,
        lane=lane,
        symbol=symbol,
        episode_index=episode_index,
        contract_start=contract_start,
        contract_stop=contract_stop,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast_digest,
        liveness_digest=liveness_digest,
        sample_count=int(cohort_rows.size),
        direction_sample_count=direction_support,
        rank_correlation=float(diagnostics.rank_correlation),
        direction_accuracy=direction_accuracy,
        top_bottom_realized_spread=spread,
        cohort_indices=tuple(int(decisions[row]) for row in cohort_rows),
    )


def build_causal_alpha_v4_signal_scope_metrics(
    *,
    run_manifest_digest: str,
    fit_config_digest: str,
    symbol: str,
    episode_index: int,
    contract_start: int,
    contract_stop: int,
    contract_digest: str,
    fit_digest: str,
    forecast: CausalAlphaV4Forecast,
    liveness_digests: Mapping[str, str],
    actionable_mask: object,
    labels_4h: object,
    label_end_indices_4h: object,
    labels_24h: object,
    label_end_indices_24h: object,
    labels_72h: object,
    label_end_indices_72h: object,
) -> Mapping[CausalAlphaV4SignalLane, CausalAlphaV4SignalScopeMetric]:
    """Build independent 4h and slow-fused canonical V4 signal metrics."""

    for field_name, digest in (
        ("run_manifest_digest", run_manifest_digest),
        ("fit_config_digest", fit_config_digest),
        ("contract_digest", contract_digest),
        ("fit_digest", fit_digest),
    ):
        require_sha256(digest, field=f"V4 signal {field_name}")
    if forecast.symbol != symbol or forecast.fit_digest != fit_digest:
        raise ValueError("V4 signal forecast identity drifted")
    if tuple(liveness_digests) != tuple(lane.value for lane in _V4_SIGNAL_LANES):
        raise ValueError("V4 signal liveness digests must cover fast and slow lanes")
    for lane in _V4_SIGNAL_LANES:
        require_sha256(
            str(liveness_digests[lane.value]),
            field=f"V4 signal {lane.value} liveness digest",
        )
    decisions = np.asarray(forecast.decision_indices, dtype=np.int64).reshape(-1)
    rows = int(decisions.size)
    if (
        rows < 2
        or np.any(decisions < contract_start)
        or np.any(decisions >= contract_stop)
    ):
        raise ValueError("V4 signal forecast decisions are outside the contract")
    actionable = _signal_array(
        actionable_mask, rows=rows, field_name="actionable_mask", dtype=np.bool_
    ).astype(np.bool_, copy=False)
    labels4 = _signal_array(
        labels_4h, rows=rows, field_name="labels_4h", dtype=np.float64
    )
    ends4 = _signal_array(
        label_end_indices_4h,
        rows=rows,
        field_name="label_end_indices_4h",
        dtype=np.int64,
    )
    labels24 = _signal_array(
        labels_24h, rows=rows, field_name="labels_24h", dtype=np.float64
    )
    ends24 = _signal_array(
        label_end_indices_24h,
        rows=rows,
        field_name="label_end_indices_24h",
        dtype=np.int64,
    )
    labels72 = _signal_array(
        labels_72h, rows=rows, field_name="labels_72h", dtype=np.float64
    )
    ends72 = _signal_array(
        label_end_indices_72h,
        rows=rows,
        field_name="label_end_indices_72h",
        dtype=np.int64,
    )
    beta_available = np.asarray(forecast.beta_available, dtype=np.bool_).reshape(-1)
    fast_prediction = np.asarray(forecast.final_predictions["4h"], dtype=np.float64)
    slow_prediction = 0.5 * (
        np.asarray(forecast.final_predictions["24h"], dtype=np.float64)
        + np.asarray(forecast.final_predictions["72h"], dtype=np.float64) / 3.0
    )
    fast_direction = np.asarray(forecast.direction_scores["4h"], dtype=np.float64)
    slow_direction = 0.5 * (
        np.asarray(forecast.direction_scores["24h"], dtype=np.float64)
        + np.asarray(forecast.direction_scores["72h"], dtype=np.float64)
    )
    slow_realized = 0.5 * (labels24 + labels72 / 3.0)

    fast_eligible = (
        actionable
        & beta_available
        & np.isfinite(labels4)
        & np.isfinite(fast_prediction)
        & np.isfinite(fast_direction)
        & (ends4 >= decisions)
        & (ends4 < contract_stop)
    )
    slow_eligible = (
        actionable
        & beta_available
        & np.isfinite(labels24)
        & np.isfinite(labels72)
        & np.isfinite(slow_prediction)
        & np.isfinite(slow_direction)
        & (ends24 >= decisions)
        & (ends72 >= decisions)
        & (ends24 < contract_stop)
        & (ends72 < contract_stop)
    )
    fast_rows = non_overlapping_causal_alpha_v3_rows(
        decision_indices=decisions,
        label_end_indices=ends4,
        eligible_mask=fast_eligible,
    )
    slow_rows = non_overlapping_causal_alpha_v3_rows(
        decision_indices=decisions,
        label_end_indices=ends72,
        eligible_mask=slow_eligible,
    )
    if fast_rows.size < 2 or slow_rows.size < 2:
        raise ValueError("V4 signal scope has insufficient non-overlapping support")

    fast_metric = _v4_scope_metric(
        run_manifest_digest=run_manifest_digest,
        fit_config_digest=fit_config_digest,
        lane=CausalAlphaV4SignalLane.FAST_4H,
        symbol=symbol,
        episode_index=episode_index,
        contract_start=contract_start,
        contract_stop=contract_stop,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast.digest,
        liveness_digest=str(liveness_digests["fast_4h"]),
        decisions=decisions,
        cohort_rows=fast_rows,
        prediction=fast_prediction,
        realized=labels4,
        direction_score=fast_direction,
    )
    slow_metric = _v4_scope_metric(
        run_manifest_digest=run_manifest_digest,
        fit_config_digest=fit_config_digest,
        lane=CausalAlphaV4SignalLane.SLOW_FUSED,
        symbol=symbol,
        episode_index=episode_index,
        contract_start=contract_start,
        contract_stop=contract_stop,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast.digest,
        liveness_digest=str(liveness_digests["slow_fused"]),
        decisions=decisions,
        cohort_rows=slow_rows,
        prediction=slow_prediction,
        realized=slow_realized,
        direction_score=slow_direction,
    )
    return {
        CausalAlphaV4SignalLane.FAST_4H: fast_metric,
        CausalAlphaV4SignalLane.SLOW_FUSED: slow_metric,
    }


def _v4_bootstrap(
    values: tuple[float, ...], gate: CausalAlphaV4SignalGateConfig
) -> CausalAlphaV4SignalBootstrapEvidence:
    result = moving_block_mean_test(
        values,
        n_bootstrap=gate.bootstrap_resamples,
        seed=gate.bootstrap_seed,
        block_size=gate.bootstrap_block_size,
    )
    return CausalAlphaV4SignalBootstrapEvidence(
        mean=float(fmean(values)),
        p_value=result.p_value,
        lower_ci=result.lower_ci,
        upper_ci=result.upper_ci,
        block_size=result.block_size,
    )


def _v4_episode_clusters(
    metrics: tuple[CausalAlphaV4SignalScopeMetric, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    grouped: dict[tuple[int, int], list[CausalAlphaV4SignalScopeMetric]] = defaultdict(
        list
    )
    for metric in metrics:
        grouped[metric.cluster_identity].append(metric)
    ranks: list[float] = []
    spreads: list[float] = []
    directions: list[float] = []
    for interval in sorted(grouped):
        cluster = grouped[interval]
        if len({metric.symbol for metric in cluster}) != len(cluster):
            raise ValueError("V4 signal episode cluster contains duplicate symbols")
        if len({metric.fit_digest for metric in cluster}) != 1:
            raise ValueError("V4 signal episode cluster fit digest drifted")
        ranks.append(float(fmean(metric.rank_correlation for metric in cluster)))
        spreads.append(
            float(fmean(metric.top_bottom_realized_spread for metric in cluster))
        )
        directions.append(
            float(fmean(metric.direction_accuracy - 0.5 for metric in cluster))
        )
    return tuple(ranks), tuple(spreads), tuple(directions)


def _evaluate_v4_lane(
    metrics: tuple[CausalAlphaV4SignalScopeMetric, ...],
    *,
    lane: CausalAlphaV4SignalLane,
    expected_raw_scope_count: int,
    gate: CausalAlphaV4SignalGateConfig,
) -> CausalAlphaV4LaneSignalEvidence:
    if not metrics or any(metric.lane is not lane for metric in metrics):
        raise ValueError("V4 signal lane metrics are unavailable or mixed")
    if expected_raw_scope_count <= 0 or len(metrics) > expected_raw_scope_count:
        raise ValueError("V4 signal expected raw scope count is invalid")
    run_digests = {metric.run_manifest_digest for metric in metrics}
    fit_config_digests = {metric.fit_config_digest for metric in metrics}
    if len(run_digests) != 1 or len(fit_config_digests) != 1:
        raise ValueError("V4 signal lane run/fit identity drifted")
    ranks, spreads, directions = _v4_episode_clusters(metrics)
    independent_count = len(ranks)
    rank = _v4_bootstrap(ranks, gate)
    spread = _v4_bootstrap(spreads, gate)
    direction = _v4_bootstrap(directions, gate)
    reasons: list[str] = []
    if len(metrics) != expected_raw_scope_count:
        reasons.append("raw_scope_count")
    if independent_count != gate.independent_episode_count:
        reasons.append("independent_episode_count")
    if rank.lower_ci < gate.minimum_rank_ic_lower_ci:
        reasons.append("rank_ic_lower_ci")
    if spread.lower_ci < gate.minimum_top_bottom_spread_lower_ci:
        reasons.append("top_bottom_spread_lower_ci")
    if direction.lower_ci < gate.minimum_direction_accuracy_excess_lower_ci:
        reasons.append("direction_accuracy_excess_lower_ci")
    return CausalAlphaV4LaneSignalEvidence(
        lane=lane,
        metrics=metrics,
        run_manifest_digest=next(iter(run_digests)),
        raw_scope_count=len(metrics),
        expected_raw_scope_count=expected_raw_scope_count,
        independent_episode_count=independent_count,
        rank_ic=rank,
        top_bottom_spread=spread,
        direction_accuracy_excess=direction,
        gate_digest=gate.digest,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


def evaluate_causal_alpha_v4_signal_gate(
    metrics: tuple[CausalAlphaV4SignalScopeMetric, ...],
    *,
    expected_raw_scope_count_per_lane: int,
    gate: CausalAlphaV4SignalGateConfig,
) -> CausalAlphaV4SignalEvidence:
    """Require both independent fast and slow signal lanes to clear the frozen gate."""

    values = tuple(metrics)
    if not values or len({metric.identity for metric in values}) != len(values):
        raise ValueError("V4 signal gate requires unique scope metrics")
    if not isinstance(gate, CausalAlphaV4SignalGateConfig):
        raise TypeError("V4 signal gate config is invalid")
    lane_metrics = {
        lane: tuple(metric for metric in values if metric.lane is lane)
        for lane in _V4_SIGNAL_LANES
    }
    fast = _evaluate_v4_lane(
        lane_metrics[CausalAlphaV4SignalLane.FAST_4H],
        lane=CausalAlphaV4SignalLane.FAST_4H,
        expected_raw_scope_count=expected_raw_scope_count_per_lane,
        gate=gate,
    )
    slow = _evaluate_v4_lane(
        lane_metrics[CausalAlphaV4SignalLane.SLOW_FUSED],
        lane=CausalAlphaV4SignalLane.SLOW_FUSED,
        expected_raw_scope_count=expected_raw_scope_count_per_lane,
        gate=gate,
    )
    reasons = tuple(
        f"{evidence.lane.value}:{reason}"
        for evidence in (fast, slow)
        for reason in evidence.rejection_reasons
    )
    return CausalAlphaV4SignalEvidence(
        fast_4h=fast,
        slow_fused=slow,
        gate_digest=gate.digest,
        passed=fast.passed and slow.passed,
        rejection_reasons=reasons,
    )


__all__ = [
    "CAUSAL_ALPHA_V4_LIVENESS_SCHEMA",
    "CAUSAL_ALPHA_V4_SIGNAL_BOOTSTRAP_SCHEMA",
    "CAUSAL_ALPHA_V4_SIGNAL_EVIDENCE_SCHEMA",
    "CAUSAL_ALPHA_V4_SIGNAL_LANE_SCHEMA",
    "CAUSAL_ALPHA_V4_SIGNAL_SCOPE_SCHEMA",
    "CausalAlphaV4LivenessEvidence",
    "CausalAlphaV4LaneSignalEvidence",
    "CausalAlphaV4SignalBootstrapEvidence",
    "CausalAlphaV4SignalEvidence",
    "CausalAlphaV4SignalGateConfig",
    "CausalAlphaV4SignalLane",
    "CausalAlphaV4SignalScopeMetric",
    "build_causal_alpha_v4_liveness_evidence",
    "build_causal_alpha_v4_signal_scope_metrics",
    "evaluate_causal_alpha_v4_signal_gate",
]
