"""Visible Ichimoku cloud calculated without chart-only future displacement.

A span computed at source bar s is conventionally *drawn* at s+26. At decision
bar t the visible cloud therefore comes from s=t-26, never from a later bar.
The existing core cloud-position feature instead uses unshifted spans for
decision-time analysis; this opt-in series intentionally has different semantics.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from trade_rl.data.source import RawMarketSeries


@dataclass(frozen=True, slots=True)
class VisibleCloud:
    top: np.ndarray
    bottom: np.ndarray
    span_a: np.ndarray
    span_b: np.ndarray
    ready: np.ndarray


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


def visible_ichimoku_cloud(
    raw: RawMarketSeries,
    *,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> VisibleCloud:
    """Compute the visible top/bottom from past native bars in linear time.

    Readiness describes completed native-row arithmetic, not publication
    availability or grid completeness. Those remain caller-owned boundaries.
    """
    periods = (tenkan_period, kijun_period, span_b_period, displacement)
    if any(type(value) is not int or value < 1 for value in periods):
        raise ValueError("cloud periods and displacement must be positive integers")
    n = len(raw.close)
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    ready = np.zeros(n, dtype=np.bool_)
    dependency_period = max(tenkan_period, kijun_period, span_b_period)
    first_visible = dependency_period - 1 + displacement
    if first_visible < n:
        # Each unique trailing window is computed once. The displacement
        # never requires the price suffix beyond the final source row.
        source_count = n - displacement
        midpoints = {
            period: _rolling_midpoints(
                raw.high[:source_count], raw.low[:source_count], period
            )
            for period in {tenkan_period, kijun_period, span_b_period}
        }
        # Prefix counts preserve the complete tradable window through the
        # visible row, including the displacement interval, without rescans.
        blocked = np.empty(n + 1, dtype=np.int64)
        blocked[0] = 0
        np.cumsum(~raw.tradable, dtype=np.int64, out=blocked[1:])
        for t in range(first_visible, n):
            s = t - displacement
            if blocked[t + 1] != blocked[s - dependency_period + 1]:
                continue
            a[t] = _midpoint(
                float(midpoints[tenkan_period][s]), float(midpoints[kijun_period][s])
            )
            b[t] = midpoints[span_b_period][s]
            ready[t] = True
    top = np.maximum(a, b)
    bottom = np.minimum(a, b)
    for array in (a, b, top, bottom, ready):
        array.setflags(write=False)
    return VisibleCloud(top=top, bottom=bottom, span_a=a, span_b=b, ready=ready)


__all__ = ["VisibleCloud", "visible_ichimoku_cloud"]
