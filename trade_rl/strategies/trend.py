"""Deterministic multi-horizon trend baseline."""

from __future__ import annotations

import math
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
    fast_hours: float = 24.0
    base_hours: float = 48.0
    slow_hours: float = 96.0
    fast_lookback: int | None = None
    base_lookback: int | None = None
    slow_lookback: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("fast_hours", self.fast_hours),
            ("base_hours", self.base_hours),
            ("slow_hours", self.slow_hours),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not self.fast_hours < self.base_hours < self.slow_hours:
            raise ValueError("trend hour horizons must be strictly increasing")

        legacy = (
            self.fast_lookback,
            self.base_lookback,
            self.slow_lookback,
        )
        if any(value is not None for value in legacy):
            if any(value is None for value in legacy):
                raise ValueError("legacy trend lookbacks must be provided together")
            resolved = tuple(int(value) for value in legacy if value is not None)
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in legacy
            ):
                raise ValueError("legacy trend lookbacks must be positive integers")
            if not resolved[0] < resolved[1] < resolved[2]:
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
        """Compatibility value for explicit legacy bar configurations."""

        if self.config.slow_lookback is not None:
            return self.config.slow_lookback
        return int(round(self.config.slow_hours))

    def lookbacks(self, dataset: MarketDataset) -> tuple[int, int, int]:
        if self.config.fast_lookback is not None:
            assert self.config.base_lookback is not None
            assert self.config.slow_lookback is not None
            return (
                self.config.fast_lookback,
                self.config.base_lookback,
                self.config.slow_lookback,
            )
        return (
            dataset.bars_for_hours(self.config.fast_hours),
            dataset.bars_for_hours(self.config.base_hours),
            dataset.bars_for_hours(self.config.slow_hours),
        )

    def minimum_history_for(self, dataset: MarketDataset) -> int:
        return self.lookbacks(dataset)[2]

    @staticmethod
    def _target(dataset: MarketDataset, index: int, lookback: int) -> np.ndarray:
        if index < lookback:
            raise ValueError("insufficient history for trend target")
        eligible = np.all(
            dataset.symbol_active[index - lookback : index + 1],
            axis=0,
        )
        result = np.zeros(dataset.n_symbols, dtype=np.float64)
        if not np.any(eligible):
            return result
        momentum = np.log(
            dataset.close[index, eligible]
            / dataset.close[index - lookback, eligible]
        )
        centered = momentum - float(momentum.mean())
        gross = float(np.abs(centered).sum())
        if gross <= 1e-15:
            return result
        result[eligible] = centered / gross
        return result

    def targets(self, dataset: MarketDataset, index: int) -> TrendTargets:
        if not 0 <= index < dataset.n_bars:
            raise ValueError("trend target index is outside the dataset")
        fast, base, slow = self.lookbacks(dataset)
        return TrendTargets(
            fast=self._target(dataset, index, fast),
            base=self._target(dataset, index, base),
            slow=self._target(dataset, index, slow),
        )
