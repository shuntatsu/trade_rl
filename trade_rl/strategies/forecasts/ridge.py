"""Small deterministic Ridge return forecast candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.controller import (
    ForecastIntentConfig,
    ForecastIntentController,
)
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent

_SCALE_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class RidgeForecastModel:
    """Frozen shared linear forecast without symbol identity or symbol coefficients."""

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
        mean = np.asarray(self.feature_mean, dtype=np.float64).reshape(-1).copy()
        scale = np.asarray(self.feature_scale, dtype=np.float64).reshape(-1).copy()
        coefficients = (
            np.asarray(self.coefficients, dtype=np.float64).reshape(-1).copy()
        )
        expected = (len(indices),)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("feature_indices must be non-empty and unique")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise ValueError("feature_indices must contain non-negative integers")
        if (
            mean.shape != expected
            or scale.shape != expected
            or coefficients.shape != expected
        ):
            raise ValueError("Ridge arrays must match feature_indices")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(coefficients).all()
            or np.any(scale <= 0.0)
            or not math.isfinite(self.intercept)
        ):
            raise ValueError("Ridge parameters must be finite with positive scale")
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
        mean.setflags(write=False)
        scale.setflags(write=False)
        coefficients.setflags(write=False)
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


def fit_ridge_forecast(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None = None,
    fit_cutoff: np.datetime64,
    horizon_hours: int = 24,
    alpha: float = 1.0,
) -> RidgeForecastModel:
    """Fit one symbol-agnostic Ridge model on pooled, symbol-balanced rows."""

    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    training = build_causal_forecast_training_set(
        dataset,
        feature_indices=feature_indices,
        fit_symbol_indices=fit_symbol_indices,
        fit_cutoff=fit_cutoff,
        horizon_hours=horizon_hours,
    )
    x = training.features
    y = training.labels
    weights = training.sample_weights
    weight_sum = float(weights.sum())

    feature_mean = np.sum(x * weights[:, None], axis=0) / weight_sum
    centered_x = x - feature_mean
    weighted_variance = (
        np.sum(
            centered_x**2 * weights[:, None],
            axis=0,
        )
        / weight_sum
    )
    raw_scale = np.sqrt(weighted_variance)
    feature_scale = np.where(raw_scale > _SCALE_FLOOR, raw_scale, 1.0)
    standardized = centered_x / feature_scale

    intercept = float(np.dot(weights, y) / weight_sum)
    centered_y = y - intercept
    sqrt_weights = np.sqrt(weights)
    weighted_x = standardized * sqrt_weights[:, None]
    weighted_y = centered_y * sqrt_weights
    gram = weighted_x.T @ weighted_x
    regularized = gram + alpha * np.eye(len(training.feature_indices), dtype=np.float64)
    coefficients = np.linalg.solve(regularized, weighted_x.T @ weighted_y)

    return RidgeForecastModel(
        feature_indices=training.feature_indices,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        intercept=intercept,
        horizon_hours=horizon_hours,
        alpha=alpha,
        n_samples=training.n_samples,
        fit_cutoff=training.fit_cutoff,
    )


__all__ = ["RidgeForecastModel", "RidgeForecastStrategy", "fit_ridge_forecast"]
