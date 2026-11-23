"""Visible Ichimoku cloud calculated without chart-only future displacement.

A span computed at source bar s is conventionally *drawn* at s+26. At decision
bar t the visible cloud therefore comes from s=t-26, never from a later bar.
The existing core cloud-position feature instead uses unshifted spans for
decision-time analysis; this opt-in series intentionally has different semantics.
"""

from __future__ import annotations

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


def visible_ichimoku_cloud(
    raw: RawMarketSeries,
    *,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> VisibleCloud:
    """Compute the currently visible top/bottom using *past* native bars only."""
    periods = (tenkan_period, kijun_period, span_b_period, displacement)
    if any(type(value) is not int or value < 1 for value in periods):
        raise ValueError("cloud periods and displacement must be positive integers")
    n = len(raw.close)
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    ready = np.zeros(n, dtype=np.bool_)
    first_source = max(tenkan_period, kijun_period, span_b_period) - 1
    for t in range(first_source + displacement, n):
        s = t - displacement
        if not np.all(raw.tradable[s - span_b_period + 1 : t + 1]):
            continue
        tenkan = 0.5 * (
            float(np.max(raw.high[s - tenkan_period + 1 : s + 1]))
            + float(np.min(raw.low[s - tenkan_period + 1 : s + 1]))
        )
        kijun = 0.5 * (
            float(np.max(raw.high[s - kijun_period + 1 : s + 1]))
            + float(np.min(raw.low[s - kijun_period + 1 : s + 1]))
        )
        a[t] = 0.5 * (tenkan + kijun)
        b[t] = 0.5 * (
            float(np.max(raw.high[s - span_b_period + 1 : s + 1]))
            + float(np.min(raw.low[s - span_b_period + 1 : s + 1]))
        )
        ready[t] = True
    top = np.maximum(a, b)
    bottom = np.minimum(a, b)
    for array in (a, b, top, bottom, ready):
        array.setflags(write=False)
    return VisibleCloud(top=top, bottom=bottom, span_a=a, span_b=b, ready=ready)


__all__ = ["VisibleCloud", "visible_ichimoku_cloud"]
