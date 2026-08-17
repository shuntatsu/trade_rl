"""Typed summary contracts for Causal Alpha V3 Signal Forensics V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_diagnostics import CausalAlphaSignalDiagnostics
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticModel,
)

CAUSAL_ALPHA_V3_SIGNAL_FORENSICS_V2_ANALYSIS_SCHEMA: Final = (
    "causal_alpha_v3_signal_forensics_v2_sidecar_analysis_v1"
)
Horizon = Literal["24h", "72h", "fused"]


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ForensicsQuantile:
    quantile: float
    value: float

    def to_payload(self) -> dict[str, object]:
        return {"quantile": self.quantile, "value": self.value}


@dataclass(frozen=True, slots=True)
class CausalAlphaV3PredictionDistribution:
    name: str
    count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    quantiles: tuple[CausalAlphaV3ForensicsQuantile, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "count": self.count,
            "maximum": self.maximum,
            "mean": self.mean,
            "minimum": self.minimum,
            "name": self.name,
            "quantiles": tuple(item.to_payload() for item in self.quantiles),
            "standard_deviation": self.standard_deviation,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3PairedHorizonDiagnostics:
    decision_indices: tuple[int, ...]
    sample_count: int
    diagnostics_24h: CausalAlphaSignalDiagnostics | None
    diagnostics_72h: CausalAlphaSignalDiagnostics | None
    direction_accuracy_delta_24h_minus_72h: float | None
    rank_correlation_delta_24h_minus_72h: float | None
    unavailable_reason: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "decision_indices": self.decision_indices,
            "diagnostics_24h": (
                None if self.diagnostics_24h is None else self.diagnostics_24h.to_payload()
            ),
            "diagnostics_72h": (
                None if self.diagnostics_72h is None else self.diagnostics_72h.to_payload()
            ),
            "direction_accuracy_delta_24h_minus_72h": (
                self.direction_accuracy_delta_24h_minus_72h
            ),
            "rank_correlation_delta_24h_minus_72h": (
                self.rank_correlation_delta_24h_minus_72h
            ),
            "sample_count": self.sample_count,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3AvailabilityPartition:
    row_count: int
    diagnostics: CausalAlphaSignalDiagnostics | None
    unavailable_reason: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "diagnostics": (
                None if self.diagnostics is None else self.diagnostics.to_payload()
            ),
            "row_count": self.row_count,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3AvailabilityHorizon:
    horizon: Horizon
    complete: CausalAlphaV3AvailabilityPartition
    incomplete: CausalAlphaV3AvailabilityPartition

    def to_payload(self) -> dict[str, object]:
        return {
            "complete": self.complete.to_payload(),
            "horizon": self.horizon,
            "incomplete": self.incomplete.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3AvailabilitySummary:
    feature_names: tuple[str, ...]
    per_feature_available_fraction: tuple[float, ...]
    complete_prediction_row_count: int
    incomplete_prediction_row_count: int
    row_available_fraction: CausalAlphaV3PredictionDistribution
    horizons: tuple[CausalAlphaV3AvailabilityHorizon, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "complete_prediction_row_count": self.complete_prediction_row_count,
            "feature_names": self.feature_names,
            "horizons": tuple(item.to_payload() for item in self.horizons),
            "incomplete_prediction_row_count": self.incomplete_prediction_row_count,
            "per_feature_available_fraction": self.per_feature_available_fraction,
            "row_available_fraction": self.row_available_fraction.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ScopeSidecarSummary:
    fit_config_digest: str
    symbol: str
    episode_index: int
    contract_start: int
    contract_stop: int
    metric_digest: str
    diagnostic_digest: str
    horizon_24h: CausalAlphaSignalDiagnostics
    horizon_72h: CausalAlphaSignalDiagnostics
    horizon_fused: CausalAlphaSignalDiagnostics
    paired_24h_72h: CausalAlphaV3PairedHorizonDiagnostics
    prediction_distributions: tuple[CausalAlphaV3PredictionDistribution, ...]
    availability: CausalAlphaV3AvailabilitySummary

    def to_payload(self) -> dict[str, object]:
        return {
            "availability": self.availability.to_payload(),
            "contract_start": self.contract_start,
            "contract_stop": self.contract_stop,
            "diagnostic_digest": self.diagnostic_digest,
            "episode_index": self.episode_index,
            "fit_config_digest": self.fit_config_digest,
            "horizon_24h": self.horizon_24h.to_payload(),
            "horizon_72h": self.horizon_72h.to_payload(),
            "horizon_fused": self.horizon_fused.to_payload(),
            "metric_digest": self.metric_digest,
            "paired_24h_72h": self.paired_24h_72h.to_payload(),
            "prediction_distributions": tuple(
                item.to_payload() for item in self.prediction_distributions
            ),
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ModelSnapshot:
    fit_config_digest: str
    contract_start: int
    contract_stop: int
    horizon: Literal["24h", "72h"]
    model: CausalAlphaV3SignalDiagnosticModel

    def to_payload(self) -> dict[str, object]:
        return {
            "contract_start": self.contract_start,
            "contract_stop": self.contract_stop,
            "fit_config_digest": self.fit_config_digest,
            "horizon": self.horizon,
            "model": self.model.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ModelTransition:
    previous_contract_start: int
    previous_contract_stop: int
    current_contract_start: int
    current_contract_stop: int
    coefficient_cosine_similarity: float | None
    coefficient_cosine_unavailable_reason: str | None
    active_paired_coefficient_count: int
    coefficient_sign_flip_rate: float | None
    coefficient_sign_flip_unavailable_reason: str | None
    location_shift_rms: float
    log_scale_ratio_rms: float

    def to_payload(self) -> dict[str, object]:
        return {
            "active_paired_coefficient_count": self.active_paired_coefficient_count,
            "coefficient_cosine_similarity": self.coefficient_cosine_similarity,
            "coefficient_cosine_unavailable_reason": (
                self.coefficient_cosine_unavailable_reason
            ),
            "coefficient_sign_flip_rate": self.coefficient_sign_flip_rate,
            "coefficient_sign_flip_unavailable_reason": (
                self.coefficient_sign_flip_unavailable_reason
            ),
            "current_contract_start": self.current_contract_start,
            "current_contract_stop": self.current_contract_stop,
            "location_shift_rms": self.location_shift_rms,
            "log_scale_ratio_rms": self.log_scale_ratio_rms,
            "previous_contract_start": self.previous_contract_start,
            "previous_contract_stop": self.previous_contract_stop,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ChronologicalMetric:
    values: tuple[float | None, ...]
    count: int
    defined_count: int
    undefined_count: int
    minimum: float | None
    mean: float | None
    maximum: float | None
    early_mean: float | None
    late_mean: float | None
    slope: float | None

    def to_payload(self) -> dict[str, object]:
        return {
            "count": self.count,
            "defined_count": self.defined_count,
            "early_mean": self.early_mean,
            "late_mean": self.late_mean,
            "maximum": self.maximum,
            "mean": self.mean,
            "minimum": self.minimum,
            "slope": self.slope,
            "undefined_count": self.undefined_count,
            "values": self.values,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3PerSymbolEssSeries:
    symbol: str
    weighted_ess: CausalAlphaV3ChronologicalMetric

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "weighted_ess": self.weighted_ess.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ModelSeries:
    fit_config_digest: str
    horizon: Literal["24h", "72h"]
    snapshots: tuple[CausalAlphaV3ModelSnapshot, ...]
    transitions: tuple[CausalAlphaV3ModelTransition, ...]
    weighted_residual_rmse: CausalAlphaV3ChronologicalMetric
    pooled_weighted_ess: CausalAlphaV3ChronologicalMetric
    fitted_row_count: CausalAlphaV3ChronologicalMetric
    per_symbol_weighted_ess: tuple[CausalAlphaV3PerSymbolEssSeries, ...]
    overlap_weight_digest_unique_count: int
    overlap_weight_digest_transition_count: int

    def to_payload(self) -> dict[str, object]:
        return {
            "fit_config_digest": self.fit_config_digest,
            "fitted_row_count": self.fitted_row_count.to_payload(),
            "horizon": self.horizon,
            "overlap_weight_digest_transition_count": (
                self.overlap_weight_digest_transition_count
            ),
            "overlap_weight_digest_unique_count": (
                self.overlap_weight_digest_unique_count
            ),
            "per_symbol_weighted_ess": tuple(
                item.to_payload() for item in self.per_symbol_weighted_ess
            ),
            "pooled_weighted_ess": self.pooled_weighted_ess.to_payload(),
            "snapshots": tuple(item.to_payload() for item in self.snapshots),
            "transitions": tuple(item.to_payload() for item in self.transitions),
            "weighted_residual_rmse": self.weighted_residual_rmse.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3FitPredictionDistributions:
    fit_config_digest: str
    distributions: tuple[CausalAlphaV3PredictionDistribution, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "distributions": tuple(item.to_payload() for item in self.distributions),
            "fit_config_digest": self.fit_config_digest,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3EpisodePredictionDistributions:
    fit_config_digest: str
    contract_start: int
    contract_stop: int
    episode_indices: tuple[int, ...]
    distributions: tuple[CausalAlphaV3PredictionDistribution, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "contract_start": self.contract_start,
            "contract_stop": self.contract_stop,
            "distributions": tuple(item.to_payload() for item in self.distributions),
            "episode_indices": self.episode_indices,
            "fit_config_digest": self.fit_config_digest,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ChronologicalHorizonSeries:
    fit_config_digest: str
    horizon: Horizon
    episode_count: int
    contract_intervals: tuple[tuple[int, int], ...]
    direction_accuracy: CausalAlphaV3ChronologicalMetric
    rank_correlation: CausalAlphaV3ChronologicalMetric
    pearson_correlation: CausalAlphaV3ChronologicalMetric
    prediction_standard_deviation: CausalAlphaV3ChronologicalMetric
    weighted_residual_rmse: CausalAlphaV3ChronologicalMetric
    pooled_weighted_ess: CausalAlphaV3ChronologicalMetric
    mean_feature_availability: CausalAlphaV3ChronologicalMetric

    def to_payload(self) -> dict[str, object]:
        return {
            "contract_intervals": self.contract_intervals,
            "direction_accuracy": self.direction_accuracy.to_payload(),
            "episode_count": self.episode_count,
            "fit_config_digest": self.fit_config_digest,
            "horizon": self.horizon,
            "mean_feature_availability": self.mean_feature_availability.to_payload(),
            "pearson_correlation": self.pearson_correlation.to_payload(),
            "pooled_weighted_ess": self.pooled_weighted_ess.to_payload(),
            "prediction_standard_deviation": (
                self.prediction_standard_deviation.to_payload()
            ),
            "rank_correlation": self.rank_correlation.to_payload(),
            "weighted_residual_rmse": self.weighted_residual_rmse.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalForensicsV2Analysis:
    scope_summaries: tuple[CausalAlphaV3ScopeSidecarSummary, ...]
    model_series: tuple[CausalAlphaV3ModelSeries, ...]
    fit_prediction_distributions: tuple[CausalAlphaV3FitPredictionDistributions, ...]
    episode_prediction_distributions: tuple[
        CausalAlphaV3EpisodePredictionDistributions, ...
    ]
    chronological_horizon_series: tuple[CausalAlphaV3ChronologicalHorizonSeries, ...]
    overlapping_realized_rows_are_descriptive: bool = True
    schema_version: str = CAUSAL_ALPHA_V3_SIGNAL_FORENSICS_V2_ANALYSIS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != CAUSAL_ALPHA_V3_SIGNAL_FORENSICS_V2_ANALYSIS_SCHEMA:
            raise ValueError("unsupported V3 signal forensics V2 analysis schema")
        if self.overlapping_realized_rows_are_descriptive is not True:
            raise ValueError("V3 sidecar realized rows must remain descriptive")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 signal forensics V2 sidecar analysis digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "chronological_horizon_series": tuple(
                item.to_payload() for item in self.chronological_horizon_series
            ),
            "episode_prediction_distributions": tuple(
                item.to_payload() for item in self.episode_prediction_distributions
            ),
            "fit_prediction_distributions": tuple(
                item.to_payload() for item in self.fit_prediction_distributions
            ),
            "model_series": tuple(item.to_payload() for item in self.model_series),
            "overlapping_realized_rows_are_descriptive": (
                self.overlapping_realized_rows_are_descriptive
            ),
            "schema_version": self.schema_version,
            "scope_summaries": tuple(
                item.to_payload() for item in self.scope_summaries
            ),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

__all__ = [
    "CAUSAL_ALPHA_V3_SIGNAL_FORENSICS_V2_ANALYSIS_SCHEMA",
    "CausalAlphaV3AvailabilityHorizon",
    "CausalAlphaV3AvailabilityPartition",
    "CausalAlphaV3AvailabilitySummary",
    "CausalAlphaV3ChronologicalHorizonSeries",
    "CausalAlphaV3ChronologicalMetric",
    "CausalAlphaV3EpisodePredictionDistributions",
    "CausalAlphaV3FitPredictionDistributions",
    "CausalAlphaV3ForensicsQuantile",
    "CausalAlphaV3ModelSeries",
    "CausalAlphaV3ModelSnapshot",
    "CausalAlphaV3ModelTransition",
    "CausalAlphaV3PairedHorizonDiagnostics",
    "CausalAlphaV3PerSymbolEssSeries",
    "CausalAlphaV3PredictionDistribution",
    "CausalAlphaV3ScopeSidecarSummary",
    "CausalAlphaV3SignalForensicsV2Analysis",
    "Horizon",
]
