"""Additional causal oscillators not maintained by the canonical FeatureKind bank.

Each indicator is isolated from divergence detection and portfolio policy.
Calculations restart at non-tradable gaps and at rows published after their own
timestamp; warm-up remains NaN. These opt-in
research formulas are not claimed to match TradingView byte-for-byte.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from trade_rl.data.source import RawMarketSeries

EXTRA_INDICATOR_NAMES = frozenset(
    {
        "MOM",
        "ROC",
        "OBV",
        "VWMACD",
        "PPO",
        "AWESOME",
        "ICHIMOKU_TENKAN",
        "ICHIMOKU_KIJUN",
    }
)


def _segments(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    padded = np.concatenate((np.array([False]), mask, np.array([False])))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return tuple((int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2))


def _ema_segment(values: np.ndarray, start: int, stop: int, length: int) -> np.ndarray:
    out = np.full(stop - start, np.nan, dtype=np.float64)
    if stop == start:
        return out
    alpha = 2 / (length + 1)
    out[0] = values[start]
    for j in range(1, stop - start):
        out[j] = alpha * values[start + j] + (1 - alpha) * out[j - 1]
    return out


def _sma_segment(values: np.ndarray, start: int, stop: int, period: int) -> np.ndarray:
    out = np.full(stop - start, np.nan, dtype=np.float64)
    for j in range(period - 1, stop - start):
        out[j] = (
            math.fsum(float(x) for x in values[start + j - period + 1 : start + j + 1])
            / period
        )
    return out


def _vwma_segment(
    close: np.ndarray, volume: np.ndarray, start: int, stop: int, period: int
) -> np.ndarray:
    out = np.full(stop - start, np.nan, dtype=np.float64)
    for j in range(period - 1, stop - start):
        prices = close[start + j - period + 1 : start + j + 1]
        vols = volume[start + j - period + 1 : start + j + 1]
        scale = float(np.max(vols))
        if scale <= 0:
            continue
        w = vols / scale
        denom = math.fsum(float(v) for v in w)
        if denom <= 0:
            continue
        price_scale = float(np.max(np.abs(prices)))
        numerator = math.fsum(float(p * v) for p, v in zip(prices / price_scale, w))
        out[j] = numerator * price_scale / denom
    return out


def build_extra_indicators(
    raw: RawMarketSeries,
    *,
    names: Iterable[str],
    usable: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute selected optional raw-scale oscillators using only past bars.

    ``usable`` defaults to tradable rows published on time
    (``available_at <= timestamp``); a caller may only narrow that mask.
    """
    selected = tuple(names)
    if set(selected) - EXTRA_INDICATOR_NAMES:
        raise ValueError("unsupported extra indicator")
    n = len(raw.close)
    outputs = {name: np.full(n, np.nan, dtype=np.float64) for name in selected}
    if not selected:
        return outputs
    assert raw.available_at is not None
    rows = raw.tradable & (
        raw.available_at.astype("datetime64[ns]")
        <= raw.timestamps.astype("datetime64[ns]")
    )
    if usable is not None:
        mask = np.asarray(usable)
        if mask.dtype != np.dtype(np.bool_) or mask.shape != rows.shape:
            raise ValueError("usable must be a boolean mask shaped like the series")
        rows = rows & mask
    for start, stop in _segments(rows):
        close, volume, high, low = raw.close, raw.volume, raw.high, raw.low
        length = stop - start
        if "MOM" in outputs and length > 10:
            outputs["MOM"][start + 10 : stop] = (
                close[start + 10 : stop] - close[start : stop - 10]
            )
        if "ROC" in outputs and length > 10:
            outputs["ROC"][start + 10 : stop] = 100 * (
                close[start + 10 : stop] / close[start : stop - 10] - 1
            )
        if "OBV" in outputs:
            values = np.zeros(length, dtype=np.float64)
            for j in range(1, length):
                values[j] = (
                    values[j - 1]
                    + np.sign(close[start + j] - close[start + j - 1])
                    * volume[start + j]
                )
            outputs["OBV"][start:stop] = values
        if "PPO" in outputs:
            fast = _ema_segment(close, start, stop, 12)
            slow = _ema_segment(close, start, stop, 26)
            if length >= 26:
                outputs["PPO"][start + 25 : stop] = 100 * (fast[25:] / slow[25:] - 1)
        if "VWMACD" in outputs:
            fast_v = _vwma_segment(close, volume, start, stop, 12)
            slow_v = _vwma_segment(close, volume, start, stop, 26)
            outputs["VWMACD"][start:stop] = fast_v - slow_v
        if "AWESOME" in outputs:
            median = 0.5 * (high + low)
            fast_m = _sma_segment(median, start, stop, 5)
            slow_m = _sma_segment(median, start, stop, 34)
            outputs["AWESOME"][start:stop] = fast_m - slow_m
        for name, period in (
            ("ICHIMOKU_TENKAN", 9),
            ("ICHIMOKU_KIJUN", 26),
        ):
            if name not in outputs:
                continue
            for i in range(start + period - 1, stop):
                outputs[name][i] = 0.5 * (
                    float(np.max(high[i - period + 1 : i + 1]))
                    + float(np.min(low[i - period + 1 : i + 1]))
                )
    for name, values in outputs.items():
        if np.isinf(values).any():
            raise ValueError(f"nonfinite {name} result")
        values.setflags(write=False)
    return outputs


__all__ = ["EXTRA_INDICATOR_NAMES", "build_extra_indicators"]
