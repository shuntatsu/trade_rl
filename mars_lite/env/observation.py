"""Pure portfolio observation construction shared by training and serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ProgressMode = Literal["episode", "zero"]


@dataclass(frozen=True)
class ObservationSchema:
    include_risk_state: bool = False
    version: int = 1
    progress_mode: ProgressMode = "episode"

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported observation schema version: {self.version}")
        if self.progress_mode not in ("episode", "zero"):
            raise ValueError(f"unsupported observation progress mode: {self.progress_mode}")


@dataclass(frozen=True)
class ObservationState:
    weights: np.ndarray
    portfolio_value: float
    peak_value: float
    progress: float
    vol_scale: float = 1.0
    dd_scale: float = 1.0
    disagreement_scale: float = 1.0
    est_port_vol: float = 0.0


def _finite_array(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def build_observation(
    per_symbol_features: np.ndarray,
    global_features: np.ndarray,
    state: ObservationState,
    schema: ObservationSchema,
) -> np.ndarray:
    """Build the exact policy input from explicit market and portfolio state."""
    schema.validate()
    features = _finite_array(per_symbol_features, "per_symbol_features")
    if features.ndim != 2:
        raise ValueError("per_symbol_features must be a 2D array")
    globals_ = _finite_array(global_features, "global_features")
    if globals_.ndim != 1:
        raise ValueError("global_features must be a 1D array")
    weights = _finite_array(state.weights, "weights")
    if weights.ndim != 1 or len(weights) != features.shape[0]:
        raise ValueError("weights must match the per-symbol feature dimension")

    scalars = np.asarray(
        [
            state.portfolio_value,
            state.peak_value,
            state.progress,
            state.vol_scale,
            state.dd_scale,
            state.disagreement_scale,
            state.est_port_vol,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(scalars).all():
        raise ValueError("observation state contains non-finite values")
    if state.portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    if state.peak_value <= 0 or state.peak_value < state.portfolio_value:
        raise ValueError("peak_value must be positive and at least portfolio_value")

    per_symbol = np.concatenate([features, weights.reshape(-1, 1)], axis=1).ravel()
    drawdown = 1.0 - state.portfolio_value / state.peak_value
    gross = float(np.abs(weights).sum())
    progress = 0.0 if schema.progress_mode == "zero" else state.progress
    portfolio_globals = [drawdown, gross, progress]
    if schema.include_risk_state:
        portfolio_globals.extend(
            [
                state.vol_scale,
                state.dd_scale,
                state.disagreement_scale,
                state.est_port_vol,
            ]
        )
    return np.concatenate(
        [per_symbol, globals_, np.asarray(portfolio_globals, dtype=np.float64)]
    ).astype(np.float32)
