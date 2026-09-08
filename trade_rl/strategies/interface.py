"""Minimal strategy contract for lean single-symbol research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from trade_rl.strategies.position_intent import PositionIntent


def _readonly_vector(value: np.ndarray, *, field: str) -> np.ndarray:
    vector = np.asarray(value).reshape(-1).copy()
    if vector.size == 0:
        raise ValueError(f"{field} must not be empty")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, slots=True)
class StrategyObservation:
    """Point-in-time observation passed to one single-symbol strategy decision."""

    index: int
    timestamp: np.datetime64
    symbol: str
    features: np.ndarray
    feature_available: np.ndarray
    global_features: np.ndarray
    global_feature_available: np.ndarray
    current_intent: PositionIntent
    current_weight: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("index must be a non-negative integer")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be non-empty")
        if not isinstance(self.current_intent, PositionIntent):
            raise TypeError("current_intent must be a PositionIntent")
        if not math.isfinite(self.current_weight):
            raise ValueError("current_weight must be finite")

        features = _readonly_vector(self.features, field="features")
        feature_available = (
            np.asarray(self.feature_available, dtype=np.bool_).reshape(-1).copy()
        )
        global_features = _readonly_vector(
            self.global_features,
            field="global_features",
        )
        global_feature_available = (
            np.asarray(
                self.global_feature_available,
                dtype=np.bool_,
            )
            .reshape(-1)
            .copy()
        )
        if feature_available.shape != features.shape:
            raise ValueError("feature_available must match features")
        if global_feature_available.shape != global_features.shape:
            raise ValueError("global_feature_available must match global_features")
        feature_available.setflags(write=False)
        global_feature_available.setflags(write=False)

        object.__setattr__(self, "timestamp", np.datetime64(self.timestamp, "ns"))
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_available", feature_available)
        object.__setattr__(self, "global_features", global_features)
        object.__setattr__(
            self,
            "global_feature_available",
            global_feature_available,
        )


class SingleSymbolStrategy(Protocol):
    """Strategy interface shared by rule, forecast and later PPO adapters."""

    def decide(self, observation: StrategyObservation) -> PositionIntent: ...


__all__ = ["SingleSymbolStrategy", "StrategyObservation"]
