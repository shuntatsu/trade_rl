"""Research-only diagnostic evidence for causal alpha V3 Signal scopes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

CAUSAL_ALPHA_V3_SIGNAL_DIAGNOSTIC_SCOPE_SCHEMA: Final = (
    "causal_alpha_v3_signal_diagnostic_scope_v1"
)
_EPSILON: Final = 1e-12


def weighted_effective_sample_size(weights: object) -> float:
    """Return Kish-style effective sample size for positive finite weights."""

    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or not np.any(values > 0.0)
    ):
        raise ValueError(
            "weights must be finite, non-negative, non-empty, and contain positive weight"
        )
    positive = values[values > 0.0]
    total = float(positive.sum(dtype=np.float64))
    squared = float(np.square(positive).sum(dtype=np.float64))
    if total <= 0.0 or squared <= 0.0:
        raise ValueError("weights have invalid positive mass")
    result = float(total * total / squared)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("weights produced an invalid effective sample size")
    return result


def _finite(value: float, *, field: str, non_negative: bool = False) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or (non_negative and resolved < 0.0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return resolved


def _available_fraction(value: float) -> float:
    resolved = _finite(value, field="available_feature_fraction")
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("available_feature_fraction must be within [0, 1]")
    return resolved


def _available_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("available_feature_count must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalDiagnosticPredictionRow:
    decision_index: int
    actionable: bool
    available_feature_count: int
    available_feature_fraction: float
    prediction_24h: float
    prediction_72h: float
    prediction_72h_24h_equivalent: float
    expected_return_24h_equivalent: float
    uncertainty_24h_equivalent: float
    signal_to_uncertainty: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, int)
            or self.decision_index < 0
        ):
            raise ValueError("diagnostic decision_index must be a non-negative integer")
        if not isinstance(self.actionable, bool):
            raise TypeError("diagnostic actionable must be a boolean")
        object.__setattr__(
            self, "available_feature_count", _available_count(self.available_feature_count)
        )
        object.__setattr__(
            self,
            "available_feature_fraction",
            _available_fraction(self.available_feature_fraction),
        )
        for field in (
            "prediction_24h",
            "prediction_72h",
            "prediction_72h_24h_equivalent",
            "expected_return_24h_equivalent",
            "signal_to_uncertainty",
        ):
            object.__setattr__(
                self,
                field,
                _finite(float(getattr(self, field)), field=f"diagnostic {field}"),
            )
        object.__setattr__(
            self,
            "uncertainty_24h_equivalent",
            _finite(
                self.uncertainty_24h_equivalent,
                field="diagnostic uncertainty_24h_equivalent",
                non_negative=True,
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "actionable": self.actionable,
            "available_feature_count": self.available_feature_count,
            "available_feature_fraction": self.available_feature_fraction,
            "decision_index": self.decision_index,
            "expected_return_24h_equivalent": self.expected_return_24h_equivalent,
            "prediction_24h": self.prediction_24h,
            "prediction_72h": self.prediction_72h,
            "prediction_72h_24h_equivalent": self.prediction_72h_24h_equivalent,
            "signal_to_uncertainty": self.signal_to_uncertainty,
            "uncertainty_24h_equivalent": self.uncertainty_24h_equivalent,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalDiagnosticRealizedRow:
    decision_index: int
    label_end_index: int
    available_feature_count: int
    available_feature_fraction: float
    prediction: float
    realized_return: float
    raw_prediction: float | None = None
    raw_realized_return: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, int)
            or self.decision_index < 0
        ):
            raise ValueError("diagnostic decision_index must be a non-negative integer")
        if (
            isinstance(self.label_end_index, bool)
            or not isinstance(self.label_end_index, int)
            or self.label_end_index < self.decision_index
        ):
            raise ValueError("diagnostic label_end_index must cover the decision")
        object.__setattr__(
            self, "available_feature_count", _available_count(self.available_feature_count)
        )
        object.__setattr__(
            self,
            "available_feature_fraction",
            _available_fraction(self.available_feature_fraction),
        )
        for field in ("prediction", "realized_return"):
            object.__setattr__(
                self,
                field,
                _finite(float(getattr(self, field)), field=f"diagnostic {field}"),
            )
        for field in ("raw_prediction", "raw_realized_return"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(
                    self,
                    field,
                    _finite(float(value), field=f"diagnostic {field}"),
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "available_feature_count": self.available_feature_count,
            "available_feature_fraction": self.available_feature_fraction,
            "decision_index": self.decision_index,
            "label_end_index": self.label_end_index,
            "prediction": self.prediction,
            "raw_prediction": self.raw_prediction,
            "raw_realized_return": self.raw_realized_return,
            "realized_return": self.realized_return,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalDiagnosticModel:
    model_digest: str
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    location: tuple[float, ...]
    scale: tuple[float, ...]
    constant_mask: tuple[bool, ...]
    fitted_row_count: int
    weighted_residual_rmse: float
    pooled_weighted_ess: float
    per_symbol_weighted_ess: tuple[tuple[str, float], ...]
    overlap_weight_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.model_digest, field="V3 diagnostic model digest")
        require_sha256(
            self.overlap_weight_digest, field="V3 diagnostic overlap weight digest"
        )
        names = tuple(self.feature_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("V3 diagnostic model feature names must be non-empty and unique")
        width = len(names)
        coefficients = tuple(float(value) for value in self.coefficients)
        location = tuple(float(value) for value in self.location)
        scale = tuple(float(value) for value in self.scale)
        constant_mask = tuple(self.constant_mask)
        if any(len(values) != width for values in (coefficients, location, scale, constant_mask)):
            raise ValueError("V3 diagnostic model feature vectors must match feature names")
        if not all(math.isfinite(value) for value in (*coefficients, *location, *scale)):
            raise ValueError("V3 diagnostic model feature vectors must be finite")
        if any(value <= 0.0 for value in scale):
            raise ValueError("V3 diagnostic model scale must be positive")
        if any(not isinstance(value, bool) for value in constant_mask):
            raise TypeError("V3 diagnostic model constant mask must be boolean")
        if (
            isinstance(self.fitted_row_count, bool)
            or not isinstance(self.fitted_row_count, int)
            or self.fitted_row_count < 2
        ):
            raise ValueError("V3 diagnostic fitted_row_count must be at least two")
        if not math.isfinite(self.intercept):
            raise ValueError("V3 diagnostic model intercept must be finite")
        residual = _finite(
            self.weighted_residual_rmse,
            field="V3 diagnostic weighted_residual_rmse",
            non_negative=True,
        )
        pooled_ess = _finite(
            self.pooled_weighted_ess,
            field="V3 diagnostic pooled_weighted_ess",
        )
        if pooled_ess <= 0.0:
            raise ValueError("V3 diagnostic pooled_weighted_ess must be positive")
        per_symbol = tuple((str(symbol), float(value)) for symbol, value in self.per_symbol_weighted_ess)
        symbols = tuple(symbol for symbol, _ in per_symbol)
        if (
            not per_symbol
            or any(not symbol for symbol in symbols)
            or len(set(symbols)) != len(symbols)
            or symbols != tuple(sorted(symbols))
            or any(not math.isfinite(value) or value <= 0.0 for _, value in per_symbol)
        ):
            raise ValueError(
                "V3 diagnostic per_symbol_weighted_ess must be sorted, unique, and positive"
            )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "constant_mask", constant_mask)
        object.__setattr__(self, "weighted_residual_rmse", residual)
        object.__setattr__(self, "pooled_weighted_ess", pooled_ess)
        object.__setattr__(self, "per_symbol_weighted_ess", per_symbol)

    def to_payload(self) -> dict[str, object]:
        return {
            "coefficients": self.coefficients,
            "constant_mask": self.constant_mask,
            "feature_names": self.feature_names,
            "fitted_row_count": self.fitted_row_count,
            "intercept": self.intercept,
            "location": self.location,
            "model_digest": self.model_digest,
            "overlap_weight_digest": self.overlap_weight_digest,
            "per_symbol_weighted_ess": self.per_symbol_weighted_ess,
            "pooled_weighted_ess": self.pooled_weighted_ess,
            "scale": self.scale,
            "weighted_residual_rmse": self.weighted_residual_rmse,
        }


def _strictly_increasing(values: tuple[int, ...], *, field: str) -> None:
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"{field} must be strictly increasing")


def _validate_row_availability(
    *, count: int, fraction: float, feature_width: int
) -> None:
    if count > feature_width:
        raise ValueError("diagnostic available feature count exceeds feature width")
    expected = count / float(feature_width)
    if not math.isclose(fraction, expected, rel_tol=0.0, abs_tol=_EPSILON):
        raise ValueError("diagnostic available feature count/fraction disagree")


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalDiagnosticScope:
    run_manifest_digest: str
    fit_config_digest: str
    symbol: str
    episode_index: int
    contract_start: int
    contract_stop: int
    contract_digest: str
    signal_metric_digest: str
    fit_digest: str
    forecast_digest: str
    feature_schema_digest: str
    model_24h: CausalAlphaV3SignalDiagnosticModel
    model_72h: CausalAlphaV3SignalDiagnosticModel
    prediction_rows: tuple[CausalAlphaV3SignalDiagnosticPredictionRow, ...]
    realized_24h_rows: tuple[CausalAlphaV3SignalDiagnosticRealizedRow, ...]
    realized_72h_rows: tuple[CausalAlphaV3SignalDiagnosticRealizedRow, ...]
    realized_fused_rows: tuple[CausalAlphaV3SignalDiagnosticRealizedRow, ...]
    canonical_cohort_indices: tuple[int, ...]
    per_feature_available_fraction: tuple[float, ...]
    complete_feature_row_count: int
    incomplete_feature_row_count: int
    available_feature_fraction_minimum: float
    available_feature_fraction_mean: float
    available_feature_fraction_maximum: float
    schema_version: str = CAUSAL_ALPHA_V3_SIGNAL_DIAGNOSTIC_SCOPE_SCHEMA
    research_only: bool = True
    promotion_eligible: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        for field in (
            "run_manifest_digest",
            "fit_config_digest",
            "contract_digest",
            "signal_metric_digest",
            "fit_digest",
            "forecast_digest",
            "feature_schema_digest",
        ):
            require_sha256(str(getattr(self, field)), field=f"V3 diagnostic {field}")
        if not self.symbol:
            raise ValueError("V3 diagnostic symbol must be non-empty")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
        ):
            raise ValueError("V3 diagnostic episode_index must be non-negative")
        if (
            isinstance(self.contract_start, bool)
            or isinstance(self.contract_stop, bool)
            or not isinstance(self.contract_start, int)
            or not isinstance(self.contract_stop, int)
            or self.contract_start < 0
            or self.contract_stop <= self.contract_start
        ):
            raise ValueError("V3 diagnostic contract interval is invalid")
        if self.schema_version != CAUSAL_ALPHA_V3_SIGNAL_DIAGNOSTIC_SCOPE_SCHEMA:
            raise ValueError("V3 diagnostic scope schema is unsupported")
        if self.research_only is not True or self.promotion_eligible is not False:
            raise ValueError("V3 diagnostic scope must remain research-only")
        if not isinstance(self.model_24h, CausalAlphaV3SignalDiagnosticModel) or not isinstance(
            self.model_72h, CausalAlphaV3SignalDiagnosticModel
        ):
            raise TypeError("V3 diagnostic models are invalid")
        if self.model_24h.feature_names != self.model_72h.feature_names:
            raise ValueError("V3 diagnostic model feature order drifted across horizons")
        feature_width = len(self.model_24h.feature_names)
        feature_fractions = tuple(float(value) for value in self.per_feature_available_fraction)
        if len(feature_fractions) != feature_width or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in feature_fractions
        ):
            raise ValueError("V3 diagnostic per-feature availability is invalid")
        prediction_rows = tuple(self.prediction_rows)
        if not prediction_rows or any(
            not isinstance(row, CausalAlphaV3SignalDiagnosticPredictionRow)
            for row in prediction_rows
        ):
            raise ValueError("V3 diagnostic prediction rows are invalid")
        prediction_decisions = tuple(row.decision_index for row in prediction_rows)
        _strictly_increasing(prediction_decisions, field="V3 diagnostic prediction decisions")
        if prediction_decisions[0] < self.contract_start or prediction_decisions[-1] >= self.contract_stop:
            raise ValueError("V3 diagnostic prediction rows are outside the contract")
        for row in prediction_rows:
            _validate_row_availability(
                count=row.available_feature_count,
                fraction=row.available_feature_fraction,
                feature_width=feature_width,
            )
        realized_sets = (
            tuple(self.realized_24h_rows),
            tuple(self.realized_72h_rows),
            tuple(self.realized_fused_rows),
        )
        for rows in realized_sets:
            if any(not isinstance(row, CausalAlphaV3SignalDiagnosticRealizedRow) for row in rows):
                raise ValueError("V3 diagnostic realized rows are invalid")
            decisions = tuple(row.decision_index for row in rows)
            _strictly_increasing(decisions, field="V3 diagnostic realized decisions")
            for row in rows:
                if (
                    row.decision_index < self.contract_start
                    or row.decision_index >= self.contract_stop
                    or row.label_end_index >= self.contract_stop
                ):
                    raise ValueError("V3 diagnostic realized row is outside the contract")
                _validate_row_availability(
                    count=row.available_feature_count,
                    fraction=row.available_feature_fraction,
                    feature_width=feature_width,
                )
        cohort = tuple(int(value) for value in self.canonical_cohort_indices)
        if not cohort:
            raise ValueError("V3 diagnostic canonical cohort must not be empty")
        _strictly_increasing(cohort, field="V3 diagnostic canonical cohort")
        fused_decisions = {row.decision_index for row in realized_sets[2]}
        if any(value not in fused_decisions for value in cohort):
            raise ValueError("V3 diagnostic canonical cohort must be covered by fused rows")
        for field in ("complete_feature_row_count", "incomplete_feature_row_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V3 diagnostic {field} must be non-negative")
        if self.complete_feature_row_count + self.incomplete_feature_row_count != len(
            prediction_rows
        ):
            raise ValueError("V3 diagnostic feature availability row counts disagree")
        minimum = _available_fraction(self.available_feature_fraction_minimum)
        mean = _available_fraction(self.available_feature_fraction_mean)
        maximum = _available_fraction(self.available_feature_fraction_maximum)
        if not minimum <= mean <= maximum:
            raise ValueError("V3 diagnostic available feature fraction summary is invalid")
        observed = tuple(row.available_feature_fraction for row in prediction_rows)
        if (
            not math.isclose(minimum, min(observed), rel_tol=0.0, abs_tol=_EPSILON)
            or not math.isclose(maximum, max(observed), rel_tol=0.0, abs_tol=_EPSILON)
            or not math.isclose(
                mean,
                float(np.mean(np.asarray(observed, dtype=np.float64))),
                rel_tol=0.0,
                abs_tol=_EPSILON,
            )
        ):
            raise ValueError("V3 diagnostic available feature fraction summary drifted")
        object.__setattr__(self, "prediction_rows", prediction_rows)
        object.__setattr__(self, "realized_24h_rows", realized_sets[0])
        object.__setattr__(self, "realized_72h_rows", realized_sets[1])
        object.__setattr__(self, "realized_fused_rows", realized_sets[2])
        object.__setattr__(self, "canonical_cohort_indices", cohort)
        object.__setattr__(self, "per_feature_available_fraction", feature_fractions)
        object.__setattr__(self, "available_feature_fraction_minimum", minimum)
        object.__setattr__(self, "available_feature_fraction_mean", mean)
        object.__setattr__(self, "available_feature_fraction_maximum", maximum)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 diagnostic scope digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.fit_config_digest, self.symbol, self.episode_index

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "available_feature_fraction_maximum": self.available_feature_fraction_maximum,
            "available_feature_fraction_mean": self.available_feature_fraction_mean,
            "available_feature_fraction_minimum": self.available_feature_fraction_minimum,
            "canonical_cohort_indices": self.canonical_cohort_indices,
            "complete_feature_row_count": self.complete_feature_row_count,
            "contract_digest": self.contract_digest,
            "contract_start": self.contract_start,
            "contract_stop": self.contract_stop,
            "episode_index": self.episode_index,
            "feature_schema_digest": self.feature_schema_digest,
            "fit_config_digest": self.fit_config_digest,
            "fit_digest": self.fit_digest,
            "forecast_digest": self.forecast_digest,
            "incomplete_feature_row_count": self.incomplete_feature_row_count,
            "model_24h": self.model_24h.to_payload(),
            "model_72h": self.model_72h.to_payload(),
            "per_feature_available_fraction": self.per_feature_available_fraction,
            "prediction_rows": tuple(row.to_payload() for row in self.prediction_rows),
            "promotion_eligible": self.promotion_eligible,
            "realized_24h_rows": tuple(row.to_payload() for row in self.realized_24h_rows),
            "realized_72h_rows": tuple(row.to_payload() for row in self.realized_72h_rows),
            "realized_fused_rows": tuple(
                row.to_payload() for row in self.realized_fused_rows
            ),
            "research_only": self.research_only,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "signal_metric_digest": self.signal_metric_digest,
            "symbol": self.symbol,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


__all__ = [
    "CAUSAL_ALPHA_V3_SIGNAL_DIAGNOSTIC_SCOPE_SCHEMA",
    "CausalAlphaV3SignalDiagnosticModel",
    "CausalAlphaV3SignalDiagnosticPredictionRow",
    "CausalAlphaV3SignalDiagnosticRealizedRow",
    "CausalAlphaV3SignalDiagnosticScope",
    "weighted_effective_sample_size",
]
