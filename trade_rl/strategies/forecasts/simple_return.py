"""Direct simple-price-return Ridge projections with strict source maturity.

This is a distinct model unit, not a transformation of a log-return prediction.
The fit-prefix marginal variance is a regularizer, not calibrated conditional
risk or realized account wealth. This module grants no execution authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.ridge import _solve_weighted_ridge
from trade_rl.strategies.forecasts.supervised import (
    _immutable_array,
    _select_causal_forecast_rows,
    _validated_fit_cutoff_ns,
    validate_supervised_forecast_horizon_hours,
)


def _finite_real(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite real number")
    return result


def _alpha(value: object) -> float:
    alpha = _finite_real(value, field="alpha")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return alpha


def _real_vector(value: object, *, field: str) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{field} must not be a masked array")
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{field} must be a one-dimensional real vector")
    return np.asarray(raw, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SimpleReturnRidgeModel:
    """Direct close-ratio projection; deliberately not a log-Ridge model."""

    feature_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    horizon_hours: int
    alpha: float
    n_samples: int
    fit_cutoff: np.datetime64
    fit_prefix_marginal_variance: float
    training_scope_sha256: str

    return_unit: ClassVar[str] = "expected_simple_return"
    valuation_basis: ClassVar[str] = "same_close_price_return"
    variance_kind: ClassVar[str] = "fit_prefix_marginal_variance"

    def __post_init__(self) -> None:
        indices = self.feature_indices
        names = self.feature_names
        if (
            type(indices) is not tuple
            or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(
                "feature_indices must be unique native nonnegative integers"
            )
        if (
            type(names) is not tuple
            or len(names) != len(indices)
            or any(type(name) is not str or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("feature_names must bind the selected feature layout")
        for name in ("feature_mean", "feature_scale", "coefficients"):
            array = _real_vector(getattr(self, name), field=name)
            if array.shape != (len(indices),) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite and match feature_indices")
            if name == "feature_scale" and np.any(array <= 0):
                raise ValueError("feature_scale must be positive")
            object.__setattr__(self, name, _immutable_array(array, dtype="float64"))
        variance = _finite_real(
            self.fit_prefix_marginal_variance, field=self.variance_kind
        )
        if variance < 0.0:
            raise ValueError("fit-prefix marginal variance cannot be negative")
        if type(self.n_samples) is not int or self.n_samples < 2:
            raise ValueError("n_samples must count at least two fit rows")
        require_sha256(self.training_scope_sha256, field="training_scope_sha256")
        object.__setattr__(
            self, "intercept", _finite_real(self.intercept, field="intercept")
        )
        object.__setattr__(self, "alpha", _alpha(self.alpha))
        object.__setattr__(self, "fit_prefix_marginal_variance", variance)
        object.__setattr__(
            self, "fit_cutoff", _validated_fit_cutoff_ns(self.fit_cutoff)
        )
        object.__setattr__(
            self,
            "horizon_hours",
            validate_supervised_forecast_horizon_hours(self.horizon_hours),
        )

    def predict_simple_return(
        self,
        features: np.ndarray,
        *,
        decision_time: np.datetime64,
        feature_available: np.ndarray,
        feature_names: tuple[str, ...],
    ) -> float:
        """Project available features at or after the model's declared cutoff.

        The cutoff is a causal floor, not proof of measured model readiness.
        Callers must still bind the vintage's actual activation/readiness.
        """
        if _validated_fit_cutoff_ns(decision_time) < self.fit_cutoff:
            raise ValueError("decision precedes simple-return fit cutoff")
        vector = _real_vector(features, field="features")
        if np.ma.isMaskedArray(feature_available):
            raise ValueError("feature_available must not be masked")
        available = np.asarray(feature_available)
        if (
            available.dtype != np.dtype(np.bool_)
            or available.shape != vector.shape
            or max(self.feature_indices) >= vector.size
        ):
            raise ValueError(
                "feature_available and features must match the model layout"
            )
        indices = list(self.feature_indices)
        if (
            type(feature_names) is not tuple
            or len(feature_names) != vector.size
            or any(type(name) is not str or not name.strip() for name in feature_names)
            or len(set(feature_names)) != len(feature_names)
            or tuple(feature_names[index] for index in indices) != self.feature_names
        ):
            raise ValueError("feature_names differ from the simple-return fit layout")
        if not available[indices].all():
            raise ValueError("selected features must be available")
        selected = vector[indices]
        if not np.isfinite(selected).all():
            raise ValueError("selected features must be finite")
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                prediction = float(
                    self.intercept
                    + ((selected - self.feature_mean) / self.feature_scale)
                    @ self.coefficients
                )
        except FloatingPointError as error:
            raise ValueError("simple-return prediction must be finite") from error
        if not math.isfinite(prediction) or prediction < -1.0:
            raise ValueError("simple-return prediction must be finite and at least -1")
        return prediction


def fit_simple_return_ridge(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None = None,
    fit_cutoff: np.datetime64,
    horizon_hours: int = 24,
    alpha: float = 1.0,
) -> SimpleReturnRidgeModel:
    """Fit raw end_close/start_close - 1 on strictly mature, balanced rows."""
    alpha = _alpha(alpha)
    horizon_hours = validate_supervised_forecast_horizon_hours(horizon_hours)
    cutoff = _validated_fit_cutoff_ns(fit_cutoff)
    rows = _select_causal_forecast_rows(
        dataset,
        feature_indices=feature_indices,
        fit_symbol_indices=fit_symbol_indices,
        cutoff=cutoff,
        horizon_hours=horizon_hours,
    )
    start_close = np.asarray(
        [dataset.close[start, symbol] for symbol, start, _ in rows.source_rows],
        dtype=np.float64,
    )
    end_close = np.asarray(
        [dataset.close[end, symbol] for symbol, _, end in rows.source_rows],
        dtype=np.float64,
    )
    weights = rows.sample_weights
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            labels = end_close / start_close - 1.0
            total = float(weights.sum())
            label_mean = float(np.dot(weights, labels) / total)
            variance = float(np.dot(weights, (labels - label_mean) ** 2) / total)
            mean, scale, coefficients, intercept = _solve_weighted_ridge(
                rows.features, labels, weights, alpha=alpha
            )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "simple-return Ridge arithmetic must be finite and solvable"
        ) from error
    if not np.isfinite(labels).all() or np.any(labels < -1.0):
        raise ValueError("simple-return labels must be finite and at least -1")

    # Bind the selected fit prefix, not the unseen suffix or its Dataset ID.
    # A future execution artifact must separately bind the full Dataset and
    # the model's provenance; this digest is not producer authentication.
    names = tuple(dataset.feature_names[index] for index in rows.feature_indices)
    timestamps_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    available_ns = (
        dataset.resolved_array("available_at").astype("datetime64[ns]").astype(np.int64)
    )
    scope_digest = content_digest(
        {
            "schema": "direct_simple_return_fit_scope_v1",
            "return_unit": SimpleReturnRidgeModel.return_unit,
            "valuation_basis": SimpleReturnRidgeModel.valuation_basis,
            "feature_indices": rows.feature_indices,
            "feature_names": names,
            "fit_symbols": tuple(
                dataset.symbols[index] for index in rows.symbol_indices
            ),
            "fit_cutoff_ns": int(cutoff.astype(np.int64)),
            "horizon_hours": horizon_hours,
            "rows": [
                (
                    dataset.symbols[symbol],
                    int(timestamps_ns[start]),
                    int(timestamps_ns[end]),
                    int(available_ns[start, symbol]),
                    int(available_ns[end, symbol]),
                    float(start_close[i]),
                    float(end_close[i]),
                )
                for i, (symbol, start, end) in enumerate(rows.source_rows)
            ],
            "features": rows.features.tolist(),
            "labels": labels.tolist(),
            "sample_weights": weights.tolist(),
        }
    )
    return SimpleReturnRidgeModel(
        feature_indices=rows.feature_indices,
        feature_names=names,
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        intercept=intercept,
        horizon_hours=horizon_hours,
        alpha=alpha,
        n_samples=len(rows.source_rows),
        fit_cutoff=cutoff,
        fit_prefix_marginal_variance=variance,
        training_scope_sha256=scope_digest,
    )


__all__ = ["SimpleReturnRidgeModel", "fit_simple_return_ridge"]
