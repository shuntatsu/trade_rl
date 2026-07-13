"""Deterministic multi-horizon trend baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset


def _finite_vector(value: np.ndarray, *, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    return vector


def _normalize_gross(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).copy()
    gross = float(np.abs(vector).sum())
    if gross > 1.0:
        vector /= gross
    return vector


@dataclass(frozen=True, slots=True)
class TrendConfig:
    fast_lookback: int = 12
    base_lookback: int = 24
    slow_lookback: int = 48

    def __post_init__(self) -> None:
        if self.fast_lookback <= 0:
            raise ValueError("fast_lookback must be positive")
        if not self.fast_lookback < self.base_lookback < self.slow_lookback:
            raise ValueError("trend lookbacks must be strictly increasing")


@dataclass(frozen=True, slots=True)
class TrendTargets:
    fast: np.ndarray
    base: np.ndarray
    slow: np.ndarray

    def __post_init__(self) -> None:
        fast = _normalize_gross(_finite_vector(self.fast, field="fast"))
        base = _normalize_gross(_finite_vector(self.base, field="base"))
        slow = _normalize_gross(_finite_vector(self.slow, field="slow"))
        if fast.shape != base.shape or base.shape != slow.shape:
            raise ValueError("trend target vectors must have identical shapes")
        object.__setattr__(self, "fast", fast)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "slow", slow)


class TrendStrategy:
    """Cross-sectionally demeaned momentum normalized to unit gross exposure."""

    def __init__(self, config: TrendConfig | None = None) -> None:
        self.config = config or TrendConfig()

    @property
    def minimum_history(self) -> int:
        return self.config.slow_lookback

    @staticmethod
    def _target(dataset: MarketDataset, index: int, lookback: int) -> np.ndarray:
        if index < lookback:
            raise ValueError("insufficient history for trend target")
        momentum = np.log(dataset.close[index] / dataset.close[index - lookback])
        centered = momentum - float(momentum.mean())
        gross = float(np.abs(centered).sum())
        if gross <= 1e-15:
            return np.zeros(dataset.n_symbols, dtype=np.float64)
        return centered / gross

    def targets(self, dataset: MarketDataset, index: int) -> TrendTargets:
        if not 0 <= index < dataset.n_bars:
            raise ValueError("trend target index is outside the dataset")
        return TrendTargets(
            fast=self._target(dataset, index, self.config.fast_lookback),
            base=self._target(dataset, index, self.config.base_lookback),
            slow=self._target(dataset, index, self.config.slow_lookback),
        )
