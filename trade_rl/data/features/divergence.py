"""Causal price/indicator divergence events for research (not trading advice).

Detection is not backdated: a pivot at p is available only at p+pivot_period,
and a confirmed divergence is first observable at decision bar t, using the
candidate endpoint t-1. One pivot/indicator/type emits at most once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

DivergenceKind = Literal["bull_regular", "bear_regular", "bull_hidden", "bear_hidden"]
_KINDS: tuple[DivergenceKind, ...] = (
    "bull_regular",
    "bear_regular",
    "bull_hidden",
    "bear_hidden",
)


@dataclass(frozen=True, slots=True)
class DivergenceConfig:
    pivot_period: int = 5
    max_bars: int = 100
    max_pivots: int = 10
    min_indicators: int = 2
    pivot_source: Literal["close", "high_low"] = "close"
    validate_lines: bool = True
    min_price_change_bps: float = 0.0

    def __post_init__(self) -> None:
        for name, lo, hi in (
            ("pivot_period", 1, 50),
            ("max_bars", 6, 1000),
            ("max_pivots", 1, 20),
            ("min_indicators", 1, 20),
        ):
            v = getattr(self, name)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        if self.pivot_source not in ("close", "high_low"):
            raise ValueError("pivot_source must be close or high_low")
        if type(self.validate_lines) is not bool:
            raise TypeError("validate_lines must be a boolean")
        if (
            isinstance(self.min_price_change_bps, bool)
            or not isinstance(self.min_price_change_bps, (int, float))
            or not math.isfinite(self.min_price_change_bps)
            or self.min_price_change_bps < 0
        ):
            raise ValueError("min_price_change_bps must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DivergenceEvent:
    indicator: str
    kind: DivergenceKind
    observed_at: int
    pivot_at: int
    anchor_at: int
    price_change_bps: float
    indicator_change: float


@dataclass(frozen=True, slots=True)
class DivergenceSignals:
    bull_regular: np.ndarray
    bear_regular: np.ndarray
    bull_hidden: np.ndarray
    bear_hidden: np.ndarray
    events: tuple[DivergenceEvent, ...]

    @property
    def bull_total(self) -> np.ndarray:
        return self.bull_regular + self.bull_hidden

    @property
    def bear_total(self) -> np.ndarray:
        return self.bear_regular + self.bear_hidden


def _confirmed_pivot(prices: np.ndarray, t: int, period: int, *, upper: bool) -> bool:
    if t < 2 * period:
        return False
    p = t - period
    segment = prices[p - period : t + 1]
    value = prices[p]
    if upper:
        return bool(
            np.all(value > segment[:period]) and np.all(value > segment[period + 1 :])
        )
    return bool(
        np.all(value < segment[:period]) and np.all(value < segment[period + 1 :])
    )


def _clear_line(
    prices: np.ndarray,
    osc: np.ndarray,
    *,
    p: int,
    anchor: int,
    bullish: bool,
) -> bool:
    if anchor <= p:
        return False
    # A missing indicator breaks continuity; do not bridge outages or warm-up.
    if not np.isfinite(osc[p : anchor + 1]).all():
        return False
    if anchor == p + 1:
        return True
    weight = (np.arange(p + 1, anchor, dtype=np.float64) - p) / (anchor - p)
    price_line = prices[p] + (prices[anchor] - prices[p]) * weight
    osc_line = osc[p] + (osc[anchor] - osc[p]) * weight
    if bullish:
        return bool(
            np.all(prices[p + 1 : anchor] >= price_line)
            and np.all(osc[p + 1 : anchor] >= osc_line)
        )
    return bool(
        np.all(prices[p + 1 : anchor] <= price_line)
        and np.all(osc[p + 1 : anchor] <= osc_line)
    )


def detect_divergences(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    indicators: Mapping[str, np.ndarray],
    config: DivergenceConfig | None = None,
) -> DivergenceSignals:
    """Find four divergence types without peeking beyond each observation bar."""
    cfg = config or DivergenceConfig()
    c, h, low_price = (np.asarray(a, dtype=np.float64) for a in (close, high, low))
    n = len(c)
    if any(a.ndim != 1 or len(a) != n for a in (c, h, low_price)):
        raise ValueError("prices must be equally sized one-dimensional arrays")
    if n == 0 or not all(np.isfinite(a).all() for a in (c, h, low_price)):
        raise ValueError("prices must be nonempty and finite")
    if (
        np.any(c <= 0)
        or np.any(h < c)
        or np.any(low_price > c)
        or np.any(low_price <= 0)
    ):
        raise ValueError("invalid positive OHLC price bars")
    if not indicators:
        raise ValueError("at least one indicator is required")
    src: dict[str, np.ndarray] = {}
    for name, value in indicators.items():
        if not isinstance(name, str) or not name or name in src:
            raise ValueError("invalid indicator name")
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (n,) or np.isinf(arr).any():
            raise ValueError(f"{name} must be length {n} with finite-or-NaN values")
        src[name] = arr
    arrays = {kind: np.zeros(n, dtype=np.int16) for kind in _KINDS}
    events: list[DivergenceEvent] = []
    low_prices = c if cfg.pivot_source == "close" else low_price
    high_prices = c if cfg.pivot_source == "close" else h
    lows: list[int] = []
    highs: list[int] = []
    seen: set[tuple[str, DivergenceKind, int]] = set()

    for t in range(n):
        p = t - cfg.pivot_period
        if _confirmed_pivot(low_prices, t, cfg.pivot_period, upper=False):
            lows.insert(0, p)
            del lows[cfg.max_pivots :]
        if _confirmed_pivot(high_prices, t, cfg.pivot_period, upper=True):
            highs.insert(0, p)
            del highs[cfg.max_pivots :]
        anchor = t - 1  # confirmed at current close, endpoint on prior complete bar
        if anchor <= 0:
            continue
        candidates: list[DivergenceEvent] = []
        for name, osc in src.items():
            if not np.isfinite(osc[t]) or not np.isfinite(osc[anchor]):
                continue
            for kind in _KINDS:
                bullish = kind.startswith("bull")
                if bullish and not (osc[t] > osc[anchor] or c[t] > c[anchor]):
                    continue
                if not bullish and not (osc[t] < osc[anchor] or c[t] < c[anchor]):
                    continue
                level = low_prices if bullish else high_prices
                for old in lows if bullish else highs:
                    if t - old > cfg.max_bars:
                        break
                    if anchor - old < 2 or not np.isfinite(osc[old]):
                        continue
                    if (name, kind, old) in seen:
                        continue
                    price_delta = float((level[anchor] / level[old] - 1) * 10_000)
                    osc_delta = float(osc[anchor] - osc[old])
                    if not math.isfinite(price_delta) or not math.isfinite(osc_delta):
                        raise ValueError("nonfinite divergence delta")
                    if abs(price_delta) < cfg.min_price_change_bps:
                        continue
                    if kind == "bull_regular":
                        matched = price_delta < 0 and osc_delta > 0
                    elif kind == "bear_regular":
                        matched = price_delta > 0 and osc_delta < 0
                    elif kind == "bull_hidden":
                        matched = price_delta > 0 and osc_delta < 0
                    else:
                        matched = price_delta < 0 and osc_delta > 0
                    if not matched:
                        continue
                    if not np.isfinite(osc[old : anchor + 1]).all():
                        continue
                    if cfg.validate_lines and not _clear_line(
                        level, osc, p=old, anchor=anchor, bullish=bullish
                    ):
                        continue
                    candidates.append(
                        DivergenceEvent(
                            indicator=name,
                            kind=kind,
                            observed_at=t,
                            pivot_at=old,
                            anchor_at=anchor,
                            price_change_bps=price_delta,
                            indicator_change=osc_delta,
                        )
                    )
                    break  # nearest eligible pivot, not best after seeing future

        for bullish in (True, False):
            side = [
                event
                for event in candidates
                if event.kind.startswith("bull") == bullish
            ]
            if len({event.indicator for event in side}) < cfg.min_indicators:
                continue
            for event in side:
                arrays[event.kind][t] += 1
                seen.add((event.indicator, event.kind, event.pivot_at))
                events.append(event)

    for series in arrays.values():
        series.setflags(write=False)
    return DivergenceSignals(**arrays, events=tuple(events))


__all__ = [
    "DivergenceConfig",
    "DivergenceEvent",
    "DivergenceSignals",
    "detect_divergences",
]
