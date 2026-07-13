# mypy: disable-error-code="index"
"""Deterministic multi-horizon trend baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from trade_rl.data.market import MarketDataset


class TrendMode(str, Enum):
    """Portfolio construction applied to multi-horizon momentum."""

    AUTO = "auto"
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"
    LONG_ONLY = "long_only"
    MARKET_NEUTRAL = "market_neutral"
    CASH_OR_TREND = "cash_or_trend"


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
    mode: TrendMode | str = TrendMode.AUTO
    signal_scale: float = 0.05

    def __post_init__(self) -> None:
        for field_name, value in (
            ("fast_hours", self.fast_hours),
            ("base_hours", self.base_hours),
            ("slow_hours", self.slow_hours),
            ("signal_scale", self.signal_scale),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not self.fast_hours < self.base_hours < self.slow_hours:
            raise ValueError("trend hour horizons must be strictly increasing")
        try:
            mode = TrendMode(self.mode)
        except ValueError as error:
            raise ValueError("trend mode is not supported") from error
        object.__setattr__(self, "mode", mode)

        legacy = (
            self.fast_lookback,
            self.base_lookback,
            self.slow_lookback,
        )
        if any(value is not None for value in legacy):
            if any(value is None for value in legacy):
                raise ValueError("legacy trend lookbacks must be provided together")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in legacy
            ):
                raise ValueError("legacy trend lookbacks must be positive integers")
            resolved = tuple(int(value) for value in legacy if value is not None)
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
    """Causal trend family with one-symbol-safe defaults."""

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
        if self.config.slow_lookback is not None:
            return self.config.slow_lookback
        return dataset.minimum_index_for_history(self.config.slow_hours)

    def _resolved_mode(self, dataset: MarketDataset) -> TrendMode:
        mode = TrendMode(self.config.mode)
        if mode is TrendMode.AUTO:
            return (
                TrendMode.TIME_SERIES
                if dataset.n_symbols == 1
                else TrendMode.CROSS_SECTIONAL
            )
        return mode

    def _previous_index(
        self,
        dataset: MarketDataset,
        index: int,
        *,
        hours: float,
        lookback: int | None,
    ) -> int:
        if lookback is not None:
            if index < lookback:
                raise ValueError("insufficient history for trend target")
            return index - lookback
        return dataset.lookback_index(index, hours)

    def _target(
        self,
        dataset: MarketDataset,
        index: int,
        *,
        hours: float,
        lookback: int | None,
    ) -> np.ndarray:
        previous = self._previous_index(
            dataset,
            index,
            hours=hours,
            lookback=lookback,
        )
        momentum = np.log(dataset.close[index] / dataset.close[previous])
        active = dataset.eligibility_mask(
            index,
            lookback=index - previous,
            require_features=False,
        )
        mode = self._resolved_mode(dataset)

        if not np.any(active):
            return np.zeros(dataset.n_symbols, dtype=np.float64)
        if mode in (TrendMode.CROSS_SECTIONAL, TrendMode.MARKET_NEUTRAL):
            active_mean = float(momentum[active].mean())
            raw = np.where(active, momentum - active_mean, 0.0)
        elif mode is TrendMode.TIME_SERIES:
            raw = np.where(
                active,
                np.tanh(momentum / self.config.signal_scale),
                0.0,
            )
        elif mode is TrendMode.LONG_ONLY:
            raw = np.where(active, np.maximum(momentum, 0.0), 0.0)
        elif mode is TrendMode.CASH_OR_TREND:
            raw = np.where(
                active,
                np.maximum(
                    np.tanh(momentum / self.config.signal_scale),
                    0.0,
                ),
                0.0,
            )
        else:  # pragma: no cover - enum validation makes this unreachable
            raise RuntimeError("unhandled trend mode")

        gross = float(np.abs(raw).sum())
        if gross <= 1e-15:
            return np.zeros(dataset.n_symbols, dtype=np.float64)
        if mode in (TrendMode.CROSS_SECTIONAL, TrendMode.MARKET_NEUTRAL):
            return raw / gross
        # Directional modes preserve signal confidence and therefore an explicit
        # cash allocation. They are only capped when the aggregate gross exceeds
        # one, rather than being forced to full investment on tiny momentum.
        return _normalize_gross(raw)

    def targets(self, dataset: MarketDataset, index: int) -> TrendTargets:
        if not 0 <= index < dataset.n_bars:
            raise ValueError("trend target index is outside the dataset")
        return TrendTargets(
            fast=self._target(
                dataset,
                index,
                hours=self.config.fast_hours,
                lookback=self.config.fast_lookback,
            ),
            base=self._target(
                dataset,
                index,
                hours=self.config.base_hours,
                lookback=self.config.base_lookback,
            ),
            slow=self._target(
                dataset,
                index,
                hours=self.config.slow_hours,
                lookback=self.config.slow_lookback,
            ),
        )
