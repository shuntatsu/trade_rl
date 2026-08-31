"""Causal feature, calibration-row, and attribution-boundary science for V7."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from trade_rl.learning.causal_alpha_v4 import CausalAlphaV4Forecast
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7CalibrationRange
from trade_rl.learning.causal_alpha_v7_calibration import (
    CausalAlphaV7CalibrationFit,
    CausalAlphaV7CalibrationRows,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionBoundaries,
)

_EPSILON = 1e-12


def _aligned(value: object, *, rows: int, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"V7 {field} is not aligned")
    return array


def build_causal_alpha_v7_feature_matrix(
    *,
    forecast: CausalAlphaV4Forecast,
    uncertainty: Mapping[str, np.ndarray],
    state: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed symbol-free V7 feature matrix and availability mask."""

    if not isinstance(forecast, CausalAlphaV4Forecast):
        raise TypeError("V7 features require a V4 forecast")
    rows = len(forecast.decision_indices)
    fast_uncertainty = _aligned(
        uncertainty.get("4h"),
        rows=rows,
        field="fast uncertainty",
    )
    if np.any(fast_uncertainty < 0.0):
        raise ValueError("V7 fast uncertainty must be non-negative")
    columns = (
        _aligned(forecast.final_predictions["4h"], rows=rows, field="fast return"),
        _aligned(forecast.direction_scores["4h"], rows=rows, field="fast direction"),
        np.log(np.maximum(fast_uncertainty, _EPSILON)),
        _aligned(state.realized_volatility, rows=rows, field="realized volatility"),
        _aligned(state.liquidity, rows=rows, field="liquidity"),
        _aligned(
            state.basis_positioning_stress,
            rows=rows,
            field="basis positioning stress",
        ),
        _aligned(forecast.final_predictions["24h"], rows=rows, field="slow return 24h"),
        _aligned(forecast.final_predictions["72h"], rows=rows, field="slow return 72h"),
        _aligned(
            forecast.direction_scores["24h"], rows=rows, field="slow direction 24h"
        ),
        _aligned(
            forecast.direction_scores["72h"], rows=rows, field="slow direction 72h"
        ),
    )
    features = np.column_stack(columns)
    row_available = (
        np.asarray(state.actionable, dtype=np.bool_).reshape(-1)
        & np.asarray(state.state_eligible, dtype=np.bool_).reshape(-1)
        & np.isfinite(features).all(axis=1)
    )
    if row_available.shape != (rows,):
        raise ValueError("V7 feature availability is not aligned")
    available = np.repeat(row_available[:, None], features.shape[1], axis=1)
    features = np.where(available, features, 0.0)
    features.setflags(write=False)
    available.setflags(write=False)
    return features, available


def build_causal_alpha_v7_calibration_rows(
    *,
    sample: Any,
    forecast: CausalAlphaV4Forecast,
    uncertainty: Mapping[str, np.ndarray],
    state: Any,
    calibration_range: CausalAlphaV7CalibrationRange,
) -> CausalAlphaV7CalibrationRows:
    """Select strictly causal 4h calibration labels from one symbol."""

    if not isinstance(calibration_range, CausalAlphaV7CalibrationRange):
        raise TypeError("V7 calibration range is invalid")
    symbol = str(getattr(sample, "symbol", ""))
    decisions = np.asarray(sample.decision_indices, dtype=np.int64).reshape(-1)
    if forecast.symbol != symbol or not np.array_equal(
        forecast.decision_indices,
        decisions,
    ):
        raise ValueError("V7 calibration sample/forecast identity drifted")
    labels = _aligned(sample.labels_4h, rows=len(decisions), field="4h labels")
    ends = np.asarray(sample.label_end_indices_4h, dtype=np.int64).reshape(-1)
    if ends.shape != decisions.shape:
        raise ValueError("V7 calibration 4h label ends are not aligned")
    features, available = build_causal_alpha_v7_feature_matrix(
        forecast=forecast,
        uncertainty=uncertainty,
        state=state,
    )
    selected = (
        (decisions >= calibration_range.calibration_start)
        & (decisions < calibration_range.train_stop)
        & (ends >= decisions)
        & (ends < calibration_range.train_stop)
        & np.isfinite(labels)
        & np.asarray(state.state_eligible, dtype=np.bool_).reshape(-1)
        & available.all(axis=1)
    )
    return CausalAlphaV7CalibrationRows(
        symbol=symbol,
        decision_indices=decisions[selected],
        label_end_indices=ends[selected],
        features=features[selected],
        feature_available=available[selected],
        realized_returns=labels[selected],
        range_digest=calibration_range.digest,
    )


def _quartiles(values: np.ndarray, *, field: str) -> tuple[float, float, float]:
    result = tuple(float(value) for value in np.quantile(values, (0.25, 0.50, 0.75)))
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError(f"V7 calibration {field} quartiles are not distinct")
    return result  # type: ignore[return-value]


def build_causal_alpha_v7_attribution_boundaries(
    *,
    rows: Mapping[str, CausalAlphaV7CalibrationRows],
    fit: CausalAlphaV7CalibrationFit,
) -> CausalAlphaV7AttributionBoundaries:
    """Freeze diagnostic quartiles from calibration rows only."""

    if not isinstance(fit, CausalAlphaV7CalibrationFit):
        raise TypeError("V7 attribution boundaries require a calibration fit")
    ordered = tuple(rows[symbol] for symbol in sorted(rows))
    if (
        not ordered
        or tuple((item.symbol, item.digest) for item in ordered) != fit.rows_digests
    ):
        raise ValueError("V7 attribution calibration row identity drifted")
    features = np.concatenate(tuple(item.features for item in ordered))
    available = np.concatenate(tuple(item.feature_available for item in ordered))
    _returns, direction = fit.predict(features, feature_available=available)
    return CausalAlphaV7AttributionBoundaries(
        confidence=_quartiles(np.abs(direction), field="confidence"),
        realized_volatility=_quartiles(features[:, 3], field="realized volatility"),
        liquidity=_quartiles(features[:, 4], field="liquidity"),
        calibration_range_digest=fit.calibration_range.digest,
    )


__all__ = [
    "build_causal_alpha_v7_attribution_boundaries",
    "build_causal_alpha_v7_calibration_rows",
    "build_causal_alpha_v7_feature_matrix",
]
