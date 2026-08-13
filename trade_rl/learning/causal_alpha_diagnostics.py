"""Train-only diagnostics for causal-alpha predictions and realized returns."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest

CAUSAL_ALPHA_SIGNAL_DIAGNOSTICS_SCHEMA: Final = (
    "causal_alpha_signal_diagnostics_v1"
)
CAUSAL_ALPHA_SIGNAL_QUANTILES: Final = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
_BIN_QUANTILES: Final = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_EPSILON: Final = 1e-15


@dataclass(frozen=True, slots=True)
class CausalAlphaSignalBin:
    index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_prediction: float | None
    mean_realized_return: float | None
    direction_accuracy: float | None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not 0 <= self.index < 5:
            raise ValueError("signal bin index must be within [0, 5)")
        if not math.isfinite(self.lower_bound) or not math.isfinite(self.upper_bound):
            raise ValueError("signal bin bounds must be finite")
        if self.lower_bound > self.upper_bound:
            raise ValueError("signal bin bounds are reversed")
        if isinstance(self.count, bool) or self.count < 0:
            raise ValueError("signal bin count must be non-negative")
        for field in ("mean_prediction", "mean_realized_return"):
            value = getattr(self, field)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"signal bin {field} must be finite or null")
        if self.direction_accuracy is not None and not (
            math.isfinite(self.direction_accuracy)
            and 0.0 <= self.direction_accuracy <= 1.0
        ):
            raise ValueError("signal bin direction_accuracy must be within [0, 1]")
        if self.count == 0 and any(
            value is not None
            for value in (
                self.mean_prediction,
                self.mean_realized_return,
                self.direction_accuracy,
            )
        ):
            raise ValueError("empty signal bins cannot contain summary values")

    @property
    def digest(self) -> str:
        return content_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class CausalAlphaSignalDiagnostics:
    sample_count: int
    prediction_mean: float
    prediction_std: float
    prediction_minimum: float
    prediction_maximum: float
    prediction_quantiles: tuple[float, ...]
    realized_mean: float
    realized_std: float
    realized_minimum: float
    realized_maximum: float
    realized_quantiles: tuple[float, ...]
    pearson_correlation: float | None
    rank_correlation: float | None
    undefined_correlation_reason: str | None
    direction_accuracy: float
    prediction_negative_rate: float
    prediction_flat_rate: float
    prediction_positive_rate: float
    realized_negative_rate: float
    realized_flat_rate: float
    realized_positive_rate: float
    bins: tuple[CausalAlphaSignalBin, ...]
    schema_version: str = CAUSAL_ALPHA_SIGNAL_DIAGNOSTICS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or self.sample_count < 2:
            raise ValueError("signal diagnostics require at least two samples")
        if len(self.prediction_quantiles) != 7 or len(self.realized_quantiles) != 7:
            raise ValueError("signal diagnostics require seven fixed quantiles")
        numeric = (
            self.prediction_mean,
            self.prediction_std,
            self.prediction_minimum,
            self.prediction_maximum,
            *self.prediction_quantiles,
            self.realized_mean,
            self.realized_std,
            self.realized_minimum,
            self.realized_maximum,
            *self.realized_quantiles,
            self.direction_accuracy,
            self.prediction_negative_rate,
            self.prediction_flat_rate,
            self.prediction_positive_rate,
            self.realized_negative_rate,
            self.realized_flat_rate,
            self.realized_positive_rate,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("signal diagnostics summaries must be finite")
        if len(self.bins) != 5 or sum(item.count for item in self.bins) != self.sample_count:
            raise ValueError("signal diagnostic bins must cover every sample")
        correlations = (self.pearson_correlation, self.rank_correlation)
        if any(value is not None and not math.isfinite(value) for value in correlations):
            raise ValueError("signal correlations must be finite or null")
        if (None in correlations) != (self.undefined_correlation_reason is not None):
            raise ValueError("undefined signal correlations require an explicit reason")
        if self.schema_version != CAUSAL_ALPHA_SIGNAL_DIAGNOSTICS_SCHEMA:
            raise ValueError("unsupported causal alpha signal diagnostics schema")
        payload = self.to_payload(include_digest=False)
        expected = content_digest(payload)
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha signal diagnostics digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "digest"
        }
        payload["bins"] = tuple(asdict(item) for item in self.bins)
        if include_digest:
            payload["digest"] = self.digest
        return payload


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first_centered = first - float(np.mean(first, dtype=np.float64))
    second_centered = second - float(np.mean(second, dtype=np.float64))
    denominator = math.sqrt(
        float(np.dot(first_centered, first_centered))
        * float(np.dot(second_centered, second_centered))
    )
    if denominator <= _EPSILON:
        return None
    return float(np.dot(first_centered, second_centered) / denominator)


def _direction_rates(values: np.ndarray) -> tuple[float, float, float]:
    signs = np.sign(values)
    count = float(values.size)
    return (
        float(np.count_nonzero(signs < 0.0) / count),
        float(np.count_nonzero(signs == 0.0) / count),
        float(np.count_nonzero(signs > 0.0) / count),
    )


def evaluate_causal_alpha_signal_diagnostics(
    predicted: object, realized: object
) -> CausalAlphaSignalDiagnostics:
    prediction = np.asarray(predicted, dtype=np.float64).reshape(-1)
    outcome = np.asarray(realized, dtype=np.float64).reshape(-1)
    if prediction.shape != outcome.shape or prediction.size < 2:
        raise ValueError("predicted and realized signals must align with at least two samples")
    if not np.isfinite(prediction).all() or not np.isfinite(outcome).all():
        raise ValueError("predicted and realized signals must be finite")

    pearson = _correlation(prediction, outcome)
    rank = _correlation(_average_ranks(prediction), _average_ranks(outcome))
    prediction_constant = float(np.ptp(prediction)) <= _EPSILON
    realized_constant = float(np.ptp(outcome)) <= _EPSILON
    reason: str | None = None
    if prediction_constant and realized_constant:
        reason = "constant_prediction_and_realized"
    elif prediction_constant:
        reason = "constant_prediction"
    elif realized_constant:
        reason = "constant_realized"
    if reason is not None:
        pearson = None
        rank = None

    bin_edges = np.quantile(prediction, _BIN_QUANTILES, method="linear")
    bin_indices = np.searchsorted(bin_edges[1:-1], prediction, side="right")
    bins: list[CausalAlphaSignalBin] = []
    for index in range(5):
        selected = bin_indices == index
        count = int(np.count_nonzero(selected))
        bins.append(
            CausalAlphaSignalBin(
                index=index,
                lower_bound=float(bin_edges[index]),
                upper_bound=float(bin_edges[index + 1]),
                count=count,
                mean_prediction=(
                    None
                    if count == 0
                    else float(np.mean(prediction[selected], dtype=np.float64))
                ),
                mean_realized_return=(
                    None
                    if count == 0
                    else float(np.mean(outcome[selected], dtype=np.float64))
                ),
                direction_accuracy=(
                    None
                    if count == 0
                    else float(
                        np.mean(
                            np.sign(prediction[selected]) == np.sign(outcome[selected])
                        )
                    )
                ),
            )
        )
    prediction_rates = _direction_rates(prediction)
    realized_rates = _direction_rates(outcome)
    prediction_quantiles = tuple(
        float(value)
        for value in np.quantile(
            prediction, CAUSAL_ALPHA_SIGNAL_QUANTILES, method="linear"
        )
    )
    realized_quantiles = tuple(
        float(value)
        for value in np.quantile(outcome, CAUSAL_ALPHA_SIGNAL_QUANTILES, method="linear")
    )
    return CausalAlphaSignalDiagnostics(
        sample_count=int(prediction.size),
        prediction_mean=float(np.mean(prediction, dtype=np.float64)),
        prediction_std=float(np.std(prediction, dtype=np.float64)),
        prediction_minimum=float(np.min(prediction)),
        prediction_maximum=float(np.max(prediction)),
        prediction_quantiles=prediction_quantiles,
        realized_mean=float(np.mean(outcome, dtype=np.float64)),
        realized_std=float(np.std(outcome, dtype=np.float64)),
        realized_minimum=float(np.min(outcome)),
        realized_maximum=float(np.max(outcome)),
        realized_quantiles=realized_quantiles,
        pearson_correlation=pearson,
        rank_correlation=rank,
        undefined_correlation_reason=reason,
        direction_accuracy=float(np.mean(np.sign(prediction) == np.sign(outcome))),
        prediction_negative_rate=prediction_rates[0],
        prediction_flat_rate=prediction_rates[1],
        prediction_positive_rate=prediction_rates[2],
        realized_negative_rate=realized_rates[0],
        realized_flat_rate=realized_rates[1],
        realized_positive_rate=realized_rates[2],
        bins=tuple(bins),
    )


def causal_alpha_signal_diagnostics_from_payload(
    payload: object,
) -> CausalAlphaSignalDiagnostics:
    if not isinstance(payload, dict):
        raise ValueError("causal alpha signal diagnostics payload must be a mapping")
    raw_bins = payload.get("bins")
    if not isinstance(raw_bins, list | tuple):
        raise ValueError("causal alpha signal diagnostics bins are missing")
    bins = tuple(CausalAlphaSignalBin(**dict(item)) for item in raw_bins)
    return CausalAlphaSignalDiagnostics(
        sample_count=int(payload["sample_count"]),
        prediction_mean=float(payload["prediction_mean"]),
        prediction_std=float(payload["prediction_std"]),
        prediction_minimum=float(payload["prediction_minimum"]),
        prediction_maximum=float(payload["prediction_maximum"]),
        prediction_quantiles=tuple(float(v) for v in payload["prediction_quantiles"]),
        realized_mean=float(payload["realized_mean"]),
        realized_std=float(payload["realized_std"]),
        realized_minimum=float(payload["realized_minimum"]),
        realized_maximum=float(payload["realized_maximum"]),
        realized_quantiles=tuple(float(v) for v in payload["realized_quantiles"]),
        pearson_correlation=(
            None
            if payload["pearson_correlation"] is None
            else float(payload["pearson_correlation"])
        ),
        rank_correlation=(
            None
            if payload["rank_correlation"] is None
            else float(payload["rank_correlation"])
        ),
        undefined_correlation_reason=(
            None
            if payload["undefined_correlation_reason"] is None
            else str(payload["undefined_correlation_reason"])
        ),
        direction_accuracy=float(payload["direction_accuracy"]),
        prediction_negative_rate=float(payload["prediction_negative_rate"]),
        prediction_flat_rate=float(payload["prediction_flat_rate"]),
        prediction_positive_rate=float(payload["prediction_positive_rate"]),
        realized_negative_rate=float(payload["realized_negative_rate"]),
        realized_flat_rate=float(payload["realized_flat_rate"]),
        realized_positive_rate=float(payload["realized_positive_rate"]),
        bins=bins,
        schema_version=str(payload["schema_version"]),
        digest=str(payload["digest"]),
    )


__all__ = [
    "CAUSAL_ALPHA_SIGNAL_DIAGNOSTICS_SCHEMA",
    "CAUSAL_ALPHA_SIGNAL_QUANTILES",
    "CausalAlphaSignalBin",
    "CausalAlphaSignalDiagnostics",
    "causal_alpha_signal_diagnostics_from_payload",
    "evaluate_causal_alpha_signal_diagnostics",
]
