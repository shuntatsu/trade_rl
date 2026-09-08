"""Small causal Ridge forecaster for single-symbol return prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecast import (
    ForecastIntentConfig,
    ForecastIntentController,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent

_SCALE_FLOOR = 1e-12


def _readonly_vector(value: np.ndarray, *, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, slots=True)
class RidgeForecastModel:
    """Frozen standardized linear return forecast."""

    feature_indices: tuple[int, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    horizon_hours: int
    alpha: float
    n_samples: int
    fit_cutoff: np.datetime64

    def __post_init__(self) -> None:
        indices = tuple(self.feature_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("feature_indices must be non-empty and unique")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise ValueError("feature_indices must contain non-negative integers")
        mean = _readonly_vector(self.feature_mean, field="feature_mean")
        scale = _readonly_vector(self.feature_scale, field="feature_scale")
        coefficients = _readonly_vector(self.coefficients, field="coefficients")
        if mean.size != len(indices) or scale.size != len(indices):
            raise ValueError("feature statistics must match feature_indices")
        if coefficients.size != len(indices):
            raise ValueError("coefficients must match feature_indices")
        if np.any(scale <= 0.0):
            raise ValueError("feature_scale must be positive")
        if not math.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        if (
            isinstance(self.horizon_hours, bool)
            or not isinstance(self.horizon_hours, int)
            or self.horizon_hours <= 0
        ):
            raise ValueError("horizon_hours must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        if (
            isinstance(self.n_samples, bool)
            or not isinstance(self.n_samples, int)
            or self.n_samples <= 0
        ):
            raise ValueError("n_samples must be a positive integer")

        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "fit_cutoff", np.datetime64(self.fit_cutoff, "ns"))

    def predict(self, features: np.ndarray) -> float:
        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if max(self.feature_indices) >= vector.size:
            raise ValueError("model feature index is outside observation features")
        selected = vector[list(self.feature_indices)]
        if not np.isfinite(selected).all():
            raise ValueError("forecast features must be finite")
        standardized = (selected - self.feature_mean) / self.feature_scale
        return float(self.intercept + standardized @ self.coefficients)


class RidgeForecastStrategy:
    """Apply one frozen Ridge forecast through the shared intent controller."""

    def __init__(
        self,
        model: RidgeForecastModel,
        *,
        entry_threshold: float,
        exit_threshold: float,
    ) -> None:
        self.model = model
        self.controller = ForecastIntentController(
            ForecastIntentConfig(
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )
        )

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        indices = self.model.feature_indices
        if max(indices) >= observation.features.size:
            raise ValueError("model feature index is outside observation features")
        if not bool(np.all(observation.feature_available[list(indices)])):
            return PositionIntent.FLAT
        selected = np.asarray(observation.features[list(indices)], dtype=np.float64)
        if not np.isfinite(selected).all():
            return PositionIntent.FLAT
        forecast = self.model.predict(observation.features)
        return self.controller.decide(forecast, current=observation.current_intent)


def _validated_feature_indices(
    dataset: MarketDataset,
    feature_indices: tuple[int, ...],
) -> tuple[int, ...]:
    indices = tuple(feature_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("feature_indices must be non-empty and unique")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError("feature_indices must contain non-negative integers")
    if max(indices) >= dataset.n_features:
        raise ValueError("feature index is outside dataset features")
    return indices


def fit_ridge_forecast(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_cutoff: np.datetime64,
    horizon_hours: int = 24,
    alpha: float = 1.0,
) -> RidgeForecastModel:
    """Fit Ridge on rows whose complete forward label ends before ``fit_cutoff``."""

    if dataset.n_symbols != 1:
        raise ValueError("ridge forecast fitting requires exactly one symbol")
    indices = _validated_feature_indices(dataset, feature_indices)
    if (
        isinstance(horizon_hours, bool)
        or not isinstance(horizon_hours, int)
        or horizon_hours <= 0
    ):
        raise ValueError("horizon_hours must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")

    cutoff = np.datetime64(fit_cutoff, "ns")
    cutoff_ns = int(cutoff.astype(np.int64))
    timestamps_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    horizon_ns = int(
        np.timedelta64(horizon_hours, "h").astype("timedelta64[ns]").astype(np.int64)
    )
    time_to_index = {int(value): index for index, value in enumerate(timestamps_ns)}
    close = np.asarray(dataset.close[:, 0], dtype=np.float64)
    feature_available = np.asarray(dataset.feature_available[:, 0], dtype=np.bool_)

    rows: list[np.ndarray] = []
    labels: list[float] = []
    for start_index, start_ns in enumerate(timestamps_ns):
        end_ns = int(start_ns) + horizon_ns
        if end_ns >= cutoff_ns:
            continue
        end_index = time_to_index.get(end_ns)
        if end_index is None or end_index <= start_index:
            continue
        selected_available = feature_available[start_index, list(indices)]
        if not bool(np.all(selected_available)):
            continue
        selected = np.asarray(
            dataset.features[start_index, 0, list(indices)],
            dtype=np.float64,
        )
        if not np.isfinite(selected).all():
            continue
        start_price = float(close[start_index])
        end_price = float(close[end_index])
        if not math.isfinite(start_price) or not math.isfinite(end_price):
            continue
        if start_price <= 0.0 or end_price <= 0.0:
            continue
        rows.append(selected)
        labels.append(math.log(end_price / start_price))

    if len(rows) < 2:
        raise ValueError("ridge forecast requires at least two eligible training rows")

    x = np.stack(rows, axis=0)
    y = np.asarray(labels, dtype=np.float64)
    feature_mean = x.mean(axis=0)
    raw_scale = x.std(axis=0)
    feature_scale = np.where(raw_scale > _SCALE_FLOOR, raw_scale, 1.0)
    standardized = (x - feature_mean) / feature_scale
    intercept = float(y.mean())
    centered_y = y - intercept
    gram = standardized.T @ standardized
    regularized = gram + alpha * np.eye(len(indices), dtype=np.float64)
    coefficients = np.linalg.solve(regularized, standardized.T @ centered_y)

    return RidgeForecastModel(
        feature_indices=indices,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        intercept=intercept,
        horizon_hours=horizon_hours,
        alpha=alpha,
        n_samples=len(rows),
        fit_cutoff=cutoff,
    )


__all__ = ["RidgeForecastModel", "RidgeForecastStrategy", "fit_ridge_forecast"]
