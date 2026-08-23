"""Non-promotable signal diagnostics for Causal Alpha V4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

CAUSAL_ALPHA_V4_LIVENESS_SCHEMA: Final = "causal_alpha_v4_liveness_evidence_v1"
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
        if not 0.0 <= self.direction_positive_fraction <= 1.0 or not 0.0 <= self.direction_negative_fraction <= 1.0:
            raise ValueError("V4 liveness direction fractions are invalid")
        for value in (
            self.unique_count_at_tolerance_1e_12,
            self.maximum_near_identical_run_length,
            self.constant_feature_count,
            self.available_feature_count,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("V4 liveness counts must be non-negative integers")
        if self.unique_count_at_tolerance_1e_12 <= 0 or self.maximum_near_identical_run_length <= 0:
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
        raise ValueError("V4 liveness contribution families are incomplete or reordered")
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


__all__ = [
    "CAUSAL_ALPHA_V4_LIVENESS_SCHEMA",
    "CausalAlphaV4LivenessEvidence",
    "build_causal_alpha_v4_liveness_evidence",
]
