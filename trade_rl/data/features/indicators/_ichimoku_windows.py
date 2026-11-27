"""Shared low-level Ichimoku window arithmetic; no alignment dependencies."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from trade_rl.data.contracts import FeatureKind, IchimokuPeriods


def _midpoint(upper: float, lower: float) -> float:
    total = upper + lower
    return 0.5 * total if math.isfinite(total) else 0.5 * upper + 0.5 * lower


def _rolling_midpoints(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Compute trailing midpoints in amortized linear time."""
    result = np.full(len(high), np.nan)
    upper: deque[tuple[int, float]] = deque()
    lower: deque[tuple[int, float]] = deque()
    for i, (hi, lo) in enumerate(zip(high, low, strict=True)):
        high_value, low_value = float(hi), float(lo)
        while upper and upper[0][0] <= i - period:
            upper.popleft()
        while lower and lower[0][0] <= i - period:
            lower.popleft()
        while upper and upper[-1][1] <= high_value:
            upper.pop()
        while lower and lower[-1][1] >= low_value:
            lower.pop()
        upper.append((i, high_value))
        lower.append((i, low_value))
        if i >= period - 1:
            result[i] = _midpoint(upper[0][1], lower[0][1])
    return result


def _configured_ichimoku_events(
    kind: FeatureKind,
    periods: IchimokuPeriods,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    usable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute explicit profile features with one window pass per usable segment.

    A visible-row event binds every source row from the longest required
    trailing window through that visible row, including displacement and
    the current close denominator. Do not bridge unusable intervals.
    """
    n = len(close)
    values = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    starts = np.full(n, -1, dtype=np.int64)
    tenkan, kijun, span_b = (
        periods.tenkan_period,
        periods.kijun_period,
        periods.span_b_period,
    )
    windows: tuple[int, ...]
    if kind is FeatureKind.ICHIMOKU_TENKAN_DISTANCE:
        windows = (tenkan,)
    elif kind is FeatureKind.ICHIMOKU_KIJUN_DISTANCE:
        windows = (kijun,)
    else:
        windows = (tenkan, kijun, span_b)
    required = max(windows)
    displacement = periods.displacement
    if required + displacement > n:
        return values, valid, starts
    padded = np.concatenate(([False], usable, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for left, right in zip(edges[::2], edges[1::2], strict=True):
        start, stop = int(left), int(right)
        first = start + required - 1 + displacement
        if first >= stop:
            continue
        # Price suffix after the final source row is irrelevant to the
        # displaced spans, but still belongs to the full usability window.
        source_stop = stop - displacement
        mids = {
            period: _rolling_midpoints(
                high[start:source_stop], low[start:source_stop], period
            )
            for period in set(windows)
        }
        for t in range(first, stop):
            s = t - displacement - start
            c = float(close[t])
            if len(windows) == 1:
                value = (c - float(mids[windows[0]][s])) / c
            else:
                a = _midpoint(float(mids[tenkan][s]), float(mids[kijun][s]))
                b = float(mids[span_b][s])
                value = (
                    (c - _midpoint(a, b)) / c
                    if kind is FeatureKind.ICHIMOKU_CLOUD_POSITION
                    else abs(a - b) / c
                )
            values[t] = value
            valid[t] = True
            starts[t] = t - displacement - required + 1
    return values, valid, starts
