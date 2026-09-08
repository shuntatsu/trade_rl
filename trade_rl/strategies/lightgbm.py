"""Optional shallow LightGBM return forecast candidate."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecast import (
    ForecastIntentConfig,
    ForecastIntentController,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.supervised import build_causal_forecast_training_set


class _RegressorPredictor(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray: ...


class _TrainableRegressor(_RegressorPredictor, Protocol):
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class LightGBMForecastModel:
    """Frozen metadata around one fitted shallow LightGBM predictor."""

    feature_indices: tuple[int, ...]
    predictor: _RegressorPredictor
    horizon_hours: int
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
        if not callable(getattr(self.predictor, "predict", None)):
            raise TypeError("predictor must provide predict")
        if (
            isinstance(self.horizon_hours, bool)
            or not isinstance(self.horizon_hours, int)
            or self.horizon_hours <= 0
        ):
            raise ValueError("horizon_hours must be a positive integer")
        if (
            isinstance(self.n_samples, bool)
            or not isinstance(self.n_samples, int)
            or self.n_samples <= 0
        ):
            raise ValueError("n_samples must be a positive integer")
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "fit_cutoff", np.datetime64(self.fit_cutoff, "ns"))

    def predict(self, features: np.ndarray) -> float:
        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if max(self.feature_indices) >= vector.size:
            raise ValueError("model feature index is outside observation features")
        selected = vector[list(self.feature_indices)]
        if not np.isfinite(selected).all():
            raise ValueError("forecast features must be finite")
        prediction = np.asarray(
            self.predictor.predict(selected.reshape(1, -1)),
            dtype=np.float64,
        ).reshape(-1)
        if prediction.size != 1 or not np.isfinite(prediction[0]):
            raise ValueError("predictor must return one finite forecast")
        return float(prediction[0])


class LightGBMForecastStrategy:
    """Apply one fitted shallow LightGBM model through the shared controller."""

    def __init__(
        self,
        model: LightGBMForecastModel,
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


def _lightgbm_regressor_factory() -> Callable[..., object]:
    try:
        module = importlib.import_module("lightgbm")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LightGBM fitting requires the optional 'forecast-gbm' dependency"
        ) from exc
    regressor_class = getattr(module, "LGBMRegressor", None)
    if regressor_class is None or not callable(regressor_class):
        raise RuntimeError("lightgbm.LGBMRegressor is unavailable")
    return cast(Callable[..., object], regressor_class)


def fit_lightgbm_forecast(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_cutoff: np.datetime64,
    horizon_hours: int = 24,
    random_state: int = 0,
) -> LightGBMForecastModel:
    """Fit one shallow symbol-agnostic LightGBM on pooled balanced rows."""

    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")
    training = build_causal_forecast_training_set(
        dataset,
        feature_indices=feature_indices,
        fit_cutoff=fit_cutoff,
        horizon_hours=horizon_hours,
    )
    regressor_factory = _lightgbm_regressor_factory()
    predictor = cast(
        _TrainableRegressor,
        regressor_factory(
            objective="regression",
            n_estimators=64,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=3,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        ),
    )
    predictor.fit(
        training.features,
        training.labels,
        sample_weight=training.sample_weights,
    )
    return LightGBMForecastModel(
        feature_indices=training.feature_indices,
        predictor=predictor,
        horizon_hours=horizon_hours,
        n_samples=training.n_samples,
        fit_cutoff=training.fit_cutoff,
    )


__all__ = [
    "LightGBMForecastModel",
    "LightGBMForecastStrategy",
    "fit_lightgbm_forecast",
]
