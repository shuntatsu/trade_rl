"""Direct-simple Ridge projection on the existing selector's exact mature rows.

The pooled fit-prefix variance is an uncalibrated marginal regularizer. Neither
the same-close price label nor this statistic establishes held-quantity wealth,
conditional risk, calibrated forecasts, or profitable execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.strategies.forecasts._ridge_math import _solve_weighted_ridge
from trade_rl.strategies.forecasts.supervised import (
    CausalForecastTrainingSet,
    _immutable_array,
)
from trade_rl.strategies.forecasts.training_trace import (
    ForecastTrainingTrace,
    _timestamp,
)


def _number(value: float, *, name: str, nonnegative: bool = False) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
        or (nonnegative and value < 0.0)
    ):
        raise ValueError(f"{name} must be a finite admissible number")


def _indices(values: tuple[int, ...]) -> tuple[int, ...]:
    indices = tuple(values)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
    ):
        raise ValueError("feature_indices must be unique non-negative integers")
    return indices


def _real_array(value: np.ndarray) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf":
        raise ValueError("model arrays and features must contain real numeric values")
    return np.asarray(original, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SimpleReturnTrainingSet:
    """Derive new labels without selecting, reordering, or reweighting rows."""

    log_training: CausalForecastTrainingSet
    labels: np.ndarray = field(init=False, repr=False)
    fit_prefix_marginal_variance: float = field(init=False)

    def __post_init__(self) -> None:
        original = self.log_training
        if (
            not isinstance(original, CausalForecastTrainingSet)
            or original.trace is None
        ):
            raise ValueError("simple-return training requires actual traced rows")
        _indices(original.feature_indices)
        trace = original.trace
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                labels = trace.end_close / trace.start_close - 1.0
                total = float(original.sample_weights.sum())
                if not math.isfinite(total) or total <= 0.0:
                    raise ValueError(
                        "training weight total must be finite and positive"
                    )
                mean = float(np.dot(original.sample_weights, labels) / total)
                variance = float(
                    np.dot(original.sample_weights, (labels - mean) ** 2) / total
                )
        except FloatingPointError as error:
            raise ValueError(
                "simple-return label/variance arithmetic must be finite"
            ) from error
        if not np.isfinite(labels).all() or np.any(labels < -1.0):
            raise ValueError("direct simple labels must be finite and at least -1")
        _number(mean, name="weighted label mean")
        _number(variance, name="fit-prefix marginal variance", nonnegative=True)
        object.__setattr__(self, "labels", _immutable_array(labels, dtype="float64"))
        object.__setattr__(self, "fit_prefix_marginal_variance", variance)

    @property
    def feature_indices(self) -> tuple[int, ...]:
        return self.log_training.feature_indices

    @property
    def selected_feature_names(self) -> tuple[str, ...]:
        return self.log_training.feature_names

    @property
    def features(self) -> np.ndarray:
        return self.log_training.features

    @property
    def sample_weights(self) -> np.ndarray:
        return self.log_training.sample_weights

    @property
    def label_end_times(self) -> np.ndarray:
        return self.log_training.label_end_times

    @property
    def fit_cutoff(self) -> np.datetime64:
        return self.log_training.fit_cutoff

    @property
    def horizon_hours(self) -> int:
        return self.log_training.horizon_hours

    @property
    def n_samples(self) -> int:
        return self.log_training.n_samples

    @property
    def trace(self) -> ForecastTrainingTrace:
        trace = self.log_training.trace
        assert trace is not None
        return trace

    def scope_payload(self) -> dict[str, object]:
        return {
            "schema": "simple_return_training_scope_v1",
            "log_training": self.log_training.scope_payload(),
            "labels": self.labels.tolist(),
            "return_unit": "expected_simple_return",
            "valuation_basis": "same_close_price_return",
            "variance_kind": "fit_prefix_marginal_variance",
            "fit_prefix_marginal_variance": self.fit_prefix_marginal_variance,
        }

    @property
    def scope_digest(self) -> str:
        return content_digest(self.scope_payload())


@dataclass(frozen=True, slots=True)
class SimpleReturnRidgeModel:
    """Distinct uncalibrated simple-price-return projection and marginal proxy."""

    feature_indices: tuple[int, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    horizon_hours: int
    alpha: float
    n_samples: int
    fit_cutoff: np.datetime64
    fit_prefix_marginal_variance: float
    return_unit: str = "expected_simple_return"
    valuation_basis: str = "same_close_price_return"
    variance_kind: str = "fit_prefix_marginal_variance"

    def __post_init__(self) -> None:
        indices = _indices(self.feature_indices)
        for name in ("feature_mean", "feature_scale", "coefficients"):
            raw = _real_array(getattr(self, name))
            array = _immutable_array(raw.reshape(-1), dtype="float64")
            if array.shape != (len(indices),) or not np.isfinite(array).all():
                raise ValueError(
                    "model arrays must be finite and match feature_indices"
                )
            if name == "feature_scale" and np.any(array <= 0.0):
                raise ValueError("feature_scale must be positive")
            object.__setattr__(self, name, array)
        _number(self.intercept, name="intercept")
        _number(self.alpha, name="alpha")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        _number(
            self.fit_prefix_marginal_variance,
            name="fit-prefix marginal variance",
            nonnegative=True,
        )
        for name in ("intercept", "alpha", "fit_prefix_marginal_variance"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("horizon_hours", "n_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.return_unit != "expected_simple_return"
            or self.valuation_basis != "same_close_price_return"
            or self.variance_kind != "fit_prefix_marginal_variance"
        ):
            raise ValueError(
                "model must declare simple price units and marginal variance"
            )
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(
            self, "fit_cutoff", _timestamp(self.fit_cutoff, field="fit_cutoff")
        )

    def predict(self, features: np.ndarray) -> float:
        vector = _real_array(features).reshape(-1)
        if max(self.feature_indices) >= vector.size:
            raise ValueError("model feature index is outside observation features")
        return self.predict_selected(vector[list(self.feature_indices)])

    def predict_selected(self, selected_features: np.ndarray) -> float:
        selected = _real_array(selected_features)
        if selected.shape != self.feature_mean.shape or not np.isfinite(selected).all():
            raise ValueError(
                "selected features must be finite and match the model layout"
            )
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                standardized = (selected - self.feature_mean) / self.feature_scale
                prediction = float(self.intercept + standardized @ self.coefficients)
        except FloatingPointError as error:
            raise ValueError(
                "simple-return prediction arithmetic must be finite"
            ) from error
        _number(prediction, name="simple-return prediction")
        if prediction < -1.0:
            raise ValueError("predicted simple return cannot be below -1")
        return prediction


def _fit_simple_return_training(
    training: SimpleReturnTrainingSet, *, alpha: float
) -> SimpleReturnRidgeModel:
    """Fit the supplied direct labels once without another Dataset selector."""
    if not isinstance(training, SimpleReturnTrainingSet):
        raise ValueError("fit requires SimpleReturnTrainingSet")
    _number(alpha, name="alpha")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            mean, scale, coefficients, intercept = _solve_weighted_ridge(
                training.features, training.labels, training.sample_weights, alpha=alpha
            )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "simple-return Ridge arithmetic must be finite and solvable"
        ) from error
    return SimpleReturnRidgeModel(
        feature_indices=training.feature_indices,
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        intercept=intercept,
        horizon_hours=training.horizon_hours,
        alpha=alpha,
        n_samples=training.n_samples,
        fit_cutoff=training.fit_cutoff,
        fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
    )


__all__ = ["SimpleReturnTrainingSet", "SimpleReturnRidgeModel"]
