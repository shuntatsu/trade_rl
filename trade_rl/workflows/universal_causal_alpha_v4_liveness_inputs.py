"""Pure liveness-attribution inputs for the Causal Alpha V4 stage runner."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from trade_rl.learning.causal_alpha_teacher import CausalAlphaRidgeModel
from trade_rl.learning.causal_alpha_v4 import (
    CAUSAL_ALPHA_V4_HORIZONS,
    CausalAlphaV4Forecast,
)

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


@dataclass(frozen=True, slots=True)
class CausalAlphaV4LivenessInputs:
    intercept: float
    feature_available: np.ndarray
    constant_feature_mask: np.ndarray
    contribution_series: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not math.isfinite(self.intercept):
            raise ValueError("V4 liveness intercept must be finite")
        available = np.asarray(self.feature_available, dtype=np.bool_).copy(order="C")
        constant = np.asarray(self.constant_feature_mask, dtype=np.bool_).reshape(-1).copy()
        if available.ndim != 2 or available.shape[1] != constant.size:
            raise ValueError("V4 liveness feature masks are misaligned")
        raw = dict(self.contribution_series)
        if tuple(raw) != _CONTRIBUTION_FAMILIES:
            raise ValueError("V4 liveness contribution family order drifted")
        contributions: dict[str, np.ndarray] = {}
        for family in _CONTRIBUTION_FAMILIES:
            values = np.asarray(raw[family], dtype=np.float64).reshape(-1).copy(order="C")
            if values.shape != (available.shape[0],) or not np.isfinite(values).all():
                raise ValueError("V4 liveness contribution series are misaligned")
            values.setflags(write=False)
            contributions[family] = values
        available.setflags(write=False)
        constant.setflags(write=False)
        object.__setattr__(self, "feature_available", available)
        object.__setattr__(self, "constant_feature_mask", constant)
        object.__setattr__(self, "contribution_series", MappingProxyType(contributions))


def _matrix(value: object, *, rows: int, width: int, field: str, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (rows, width):
        raise ValueError(f"V4 liveness {field} shape drifted")
    if dtype is not np.bool_ and not np.isfinite(result).all():
        raise ValueError(f"V4 liveness {field} must be finite")
    return result


def build_causal_alpha_v4_liveness_inputs(
    *,
    fit: object,
    sample: object,
    forecast: CausalAlphaV4Forecast,
    horizon: str,
) -> CausalAlphaV4LivenessInputs:
    """Decompose the shared residual ridge into stable descriptive families."""

    if horizon not in CAUSAL_ALPHA_V4_HORIZONS:
        raise ValueError("unsupported V4 liveness horizon")
    if not isinstance(forecast, CausalAlphaV4Forecast):
        raise TypeError("V4 liveness forecast is invalid")
    residual_models = getattr(fit, "residual_models", None)
    if not isinstance(residual_models, Mapping):
        raise TypeError("V4 liveness fit does not expose residual models")
    model = residual_models.get(horizon)
    if not isinstance(model, CausalAlphaRidgeModel):
        raise TypeError("V4 liveness residual model is invalid")

    target_names = tuple(getattr(sample, "target_local_feature_names", ()))
    local = getattr(sample, "local_context", None)
    global_market = getattr(sample, "global_context", None)
    descriptor_names = tuple(getattr(sample, "instrument_descriptor_names", ()))
    local_names = tuple(getattr(local, "feature_names", ()))
    global_names = tuple(getattr(global_market, "feature_names", ()))
    names = (*target_names, *local_names, *global_names, *descriptor_names, "causal_beta")
    if model.feature_names != names:
        raise ValueError("V4 liveness residual feature schema drifted")

    rows = int(np.asarray(getattr(sample, "target_local_features", None)).shape[0])
    if rows <= 0 or forecast.decision_indices.size != rows:
        raise ValueError("V4 liveness sample/forecast rows drifted")
    target = _matrix(
        getattr(sample, "target_local_features", None),
        rows=rows,
        width=len(target_names),
        field="target local features",
        dtype=np.float64,
    )
    target_available = _matrix(
        getattr(sample, "target_local_available", None),
        rows=rows,
        width=len(target_names),
        field="target local availability",
        dtype=np.bool_,
    )
    local_values = _matrix(
        getattr(local, "values", None),
        rows=rows,
        width=len(local_names),
        field="local context",
        dtype=np.float64,
    )
    local_available = _matrix(
        getattr(local, "available", None),
        rows=rows,
        width=len(local_names),
        field="local availability",
        dtype=np.bool_,
    )
    global_values = _matrix(
        getattr(global_market, "values", None),
        rows=rows,
        width=len(global_names),
        field="global context",
        dtype=np.float64,
    )
    global_available = _matrix(
        getattr(global_market, "available", None),
        rows=rows,
        width=len(global_names),
        field="global availability",
        dtype=np.bool_,
    )
    descriptors = _matrix(
        getattr(sample, "instrument_descriptors", None),
        rows=rows,
        width=len(descriptor_names),
        field="instrument descriptors",
        dtype=np.float64,
    )
    descriptor_available = _matrix(
        getattr(sample, "instrument_descriptor_available", None),
        rows=rows,
        width=len(descriptor_names),
        field="descriptor availability",
        dtype=np.bool_,
    )
    beta = np.asarray(getattr(sample, "beta", None), dtype=np.float64).reshape(-1)
    beta_available = np.asarray(
        getattr(sample, "beta_available", None), dtype=np.bool_
    ).reshape(-1)
    if beta.shape != (rows,) or beta_available.shape != (rows,) or not np.isfinite(beta).all():
        raise ValueError("V4 liveness beta arrays are misaligned")

    features = np.column_stack((target, local_values, global_values, descriptors, beta[:, None]))
    available = np.column_stack(
        (target_available, local_available, global_available, descriptor_available, beta_available[:, None])
    ).astype(np.bool_, copy=False)
    scaled = model.transform(features, feature_available=available)
    per_feature = scaled * model.coefficients[None, :]

    target_stop = len(target_names)
    local_stop = target_stop + len(local_names)
    global_stop = local_stop + len(global_names)
    contributions = {
        "existing_15m": np.zeros(rows, dtype=np.float64),
        "existing_1h": np.zeros(rows, dtype=np.float64),
        "existing_4h": np.zeros(rows, dtype=np.float64),
        "existing_1d": np.zeros(rows, dtype=np.float64),
        "local_cross_market": per_feature[:, target_stop:local_stop].sum(axis=1),
        "global_market": per_feature[:, local_stop:global_stop].sum(axis=1),
        "beta_scaled_proxy": np.asarray(
            forecast.beta_scaled_market_contributions[horizon], dtype=np.float64
        ),
        "shared_residual": np.asarray(forecast.residual_predictions[horizon], dtype=np.float64),
    }
    for index, name in enumerate(target_names):
        for timeframe in ("15m", "1h", "4h", "1d"):
            if name.startswith(f"{timeframe}__"):
                contributions[f"existing_{timeframe}"] += per_feature[:, index]
                break

    return CausalAlphaV4LivenessInputs(
        intercept=float(model.intercept),
        feature_available=available,
        constant_feature_mask=model.constant_mask,
        contribution_series=contributions,
    )


__all__ = [
    "CausalAlphaV4LivenessInputs",
    "build_causal_alpha_v4_liveness_inputs",
]
