from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mars_lite.data.data_utils import TF_TO_MINUTES


@dataclass(frozen=True)
class TrendFamilyConfig:
    fast_lookback: int = 24
    base_lookback: int = 48
    slow_lookback: int = 96
    rebalance_every: int = 24
    base_timeframe: str = "1h"
    momentum_scale: float = 0.10


@dataclass(frozen=True)
class TrendTargets:
    fast: np.ndarray
    base: np.ndarray
    slow: np.ndarray


class TrendFamily:
    """Pure absolute-time trend targets independent of account inventory."""

    def __init__(self, config: TrendFamilyConfig | None = None):
        self.config = config or TrendFamilyConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config
        if (
            min(
                cfg.fast_lookback,
                cfg.base_lookback,
                cfg.slow_lookback,
                cfg.rebalance_every,
            )
            <= 0
        ):
            raise ValueError("lookbacks and rebalance_every must be positive")
        if cfg.base_timeframe not in TF_TO_MINUTES:
            raise ValueError(f"unsupported base_timeframe: {cfg.base_timeframe}")
        if not np.isfinite(cfg.momentum_scale) or cfg.momentum_scale <= 0:
            raise ValueError("momentum_scale must be finite and positive")

    def targets(self, fs, t: int) -> TrendTargets:
        if not 0 <= t < fs.n_bars:
            raise IndexError("t out of range")

        timestamps = np.asarray(fs.timestamps).astype("datetime64[ns]").astype(np.int64)
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")

        close = np.asarray(fs.close, dtype=np.float64)
        if close.ndim != 2 or close.shape[0] != fs.n_bars:
            raise ValueError("close shape must match FeatureSet")
        if not np.all(np.isfinite(close[: t + 1])) or np.any(close[: t + 1] <= 0):
            raise ValueError("close history must be finite and positive")

        bar_ns = int(TF_TO_MINUTES[self.config.base_timeframe]) * 60 * 1_000_000_000
        slot = int(timestamps[t] // bar_ns)
        rebalance_slot = slot - (slot % self.config.rebalance_every)
        rebalance_ns = rebalance_slot * bar_ns
        end = int(np.searchsorted(timestamps, rebalance_ns, side="right") - 1)
        if end < 0:
            return self._zero(fs.n_symbols)

        return TrendTargets(
            fast=self._weights(close, end, self.config.fast_lookback),
            base=self._weights(close, end, self.config.base_lookback),
            slow=self._weights(close, end, self.config.slow_lookback),
        )

    @staticmethod
    def _zero(n_symbols: int) -> TrendTargets:
        zero = np.zeros(n_symbols, dtype=np.float64)
        return TrendTargets(zero.copy(), zero.copy(), zero.copy())

    def _weights(self, close: np.ndarray, end: int, lookback: int) -> np.ndarray:
        if end - lookback < 0:
            return np.zeros(close.shape[1], dtype=np.float64)
        momentum = np.log(close[end] / close[end - lookback])
        raw = np.tanh(momentum / self.config.momentum_scale)
        gross = float(np.abs(raw).sum())
        return raw / gross if gross > 1.0 else raw
