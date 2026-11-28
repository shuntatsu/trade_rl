"""Shared trailing price windows and Ichimoku arithmetic; no alignment dependencies."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator

import numpy as np

from trade_rl.data.contracts import FeatureKind, IchimokuPeriods


def _midpoint(upper: float, lower: float) -> float:
    total = upper + lower
    return 0.5 * total if math.isfinite(total) else 0.5 * upper + 0.5 * lower


def _rolling_extrema(
    high: np.ndarray, low: np.ndarray, period: int
) -> Iterator[tuple[int, float, float]]:
    """Yield trailing extrema with each row entering and leaving each deque once."""
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
            yield i, upper[0][1], lower[0][1]


def _rolling_midpoints(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Compute trailing midpoints in amortized linear time."""
    result = np.full(len(high), np.nan)
    for i, upper, lower in _rolling_extrema(high, low, period):
        result[i] = _midpoint(upper, lower)
    return result


def _cloud_value(kind: FeatureKind, a: float, b: float, close: float) -> float:
    """Return one cloud-relative ratio from span A, span B and the visible close.

    Fail closed: a non-positive or non-finite close, a non-finite span or a
    nonrepresentable ratio raises instead of yielding a valid value. The edge
    comparison alone would otherwise map a NaN close or span to 0.0 ("inside").
    """
    if not math.isfinite(close) or close <= 0.0:
        raise ValueError("Ichimoku feature requires a positive finite close")
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("Ichimoku feature must be finite")
    if kind is FeatureKind.ICHIMOKU_CLOUD_POSITION:
        value = (close - _midpoint(a, b)) / close
    elif kind is FeatureKind.ICHIMOKU_CLOUD_THICKNESS:
        value = abs(a - b) / close
    elif kind is FeatureKind.ICHIMOKU_CLOUD_TWIST:
        value = (a - b) / close
    elif kind is FeatureKind.ICHIMOKU_CLOUD_EDGE_DISTANCE:
        top, bottom = max(a, b), min(a, b)
        if close > top:
            value = (close - top) / close
        elif close < bottom:
            value = (close - bottom) / close
        else:
            value = 0.0
    else:
        raise ValueError(f"not an Ichimoku cloud feature kind: {kind}")
    if not math.isfinite(value):
        raise ValueError("Ichimoku feature must be finite")
    return value


def _chikou_value(close: float, lagged: float) -> float:
    """Return (close - lagged) / close for two causal closes; fail closed."""
    for price in (close, lagged):
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError("Ichimoku feature requires a positive finite close")
    value = (close - lagged) / close
    if not math.isfinite(value):
        raise ValueError("Ichimoku feature must be finite")
    return value


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

    Cloud kinds share the span arithmetic (top = max(a, b), bottom =
    min(a, b)): position uses the midpoint, thickness |a - b|, edge distance
    the signed distance to the nearest edge (exactly zero inside, edges
    included) and twist a - b, each divided by the visible close. Chikou
    distance compares the close with the close ``displacement`` rows earlier;
    its window is [t - displacement, t] and uses no midpoint window.
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
    if kind is FeatureKind.ICHIMOKU_CHIKOU_DISTANCE:
        windows = ()
    elif kind is FeatureKind.ICHIMOKU_TENKAN_DISTANCE:
        windows = (tenkan,)
    elif kind is FeatureKind.ICHIMOKU_KIJUN_DISTANCE:
        windows = (kijun,)
    else:
        windows = (tenkan, kijun, span_b)
    required = max(windows, default=1)
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
            if not math.isfinite(c) or c <= 0.0:
                raise ValueError("Ichimoku feature requires a positive finite close")
            if kind is FeatureKind.ICHIMOKU_CHIKOU_DISTANCE:
                value = _chikou_value(c, float(close[t - displacement]))
            elif len(windows) == 1:
                value = (c - float(mids[windows[0]][s])) / c
            else:
                a = _midpoint(float(mids[tenkan][s]), float(mids[kijun][s]))
                b = float(mids[span_b][s])
                value = _cloud_value(kind, a, b, c)
            if not math.isfinite(value):
                raise ValueError("Ichimoku feature must be finite")
            values[t] = value
            valid[t] = True
            starts[t] = t - displacement - required + 1
    return values, valid, starts
