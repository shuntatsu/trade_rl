"""Causal, vectorised signal precomputation of the low-turnover family screen.

Information timing (identical to ``trade_rl/data/features/mtf6_signals.py`` and
``ShadowAnchorMtfPolicy``): dataset timestamps are bar *close* times, row ``t`` holds
only bars completed at or before ``timestamps[t]`` and a decision at snapshot index
``t`` may read row ``t``. Every array below is indexed by row; row ``t`` uses closes of
rows ``<= t`` only, so a policy reads row ``max(0, t - signal_lag_bars)``. Windows are
sliding cumulative sums over rows ``<= t``, so the value at row ``t`` is bit-identical
under any change to rows after ``t``.

A value is NaN (unavailable, weight 0) when the leg is not usable at the row (same mask
as ``mtf6_signals``: tradable, active, information available, ``available_at <=``
timestamp, finite positive close), when its history is too short or when any close the
lookback needs is not finite and positive. Spot-halt marks (preceding published close)
are finite and are used as prices, exactly as the dataset carries them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from trade_rl.data.features.mtf6_signals import (
    enforce_pre_fs1_boundary,
    validate_alternating_spot_perp_dataset,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.lowturn_screen import protocol


def _readonly(arr: np.ndarray) -> np.ndarray:
    view = np.ascontiguousarray(arr).view()
    view.setflags(write=False)
    return view


@dataclass(frozen=True, slots=True)
class LowturnSignals:
    """Per-row signals for every leg of the alternating spot/perp layout, shape (T, 2P)."""

    pair_names: tuple[str, ...]
    symbols: tuple[str, ...]
    r30: np.ndarray
    r90: np.ndarray
    r180: np.ndarray
    vol60: np.ndarray
    xs_score: np.ndarray
    sma_fast: np.ndarray
    sma_slow: np.ndarray
    usable: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.usable.shape[0])


#: Row-indexed arrays of :class:`LowturnSignals` (everything but the names).
ROW_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(LowturnSignals) if f.name not in ("pair_names", "symbols")
)


def usable_mask(dataset: MarketDataset) -> np.ndarray:
    """Leg availability per row; the same rule as ``mtf6_signals._availability_mask``."""
    ts_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    usable = np.asarray(dataset.tradable, dtype=bool).copy()
    if dataset.asset_active is not None:
        usable &= np.asarray(dataset.asset_active, dtype=bool)
    if dataset.information_available is not None:
        usable &= np.asarray(dataset.information_available, dtype=bool)
    if dataset.available_at is not None:
        avail_ns = dataset.available_at.astype("datetime64[ns]").astype(np.int64)
        usable &= avail_ns <= ts_ns[:, None]
    close = np.asarray(dataset.close, dtype=np.float64)
    usable &= np.isfinite(close) & (close > 0.0)
    return usable


def _lagged_ratio(close: np.ndarray, recent: int, past: int) -> np.ndarray:
    """``close[t - recent] / close[t - past] - 1`` (NaN until ``t >= past``)."""
    out = np.full(close.shape, np.nan, dtype=np.float64)
    n = close.shape[0]
    if n <= past:
        return out
    numer = close[past - recent : n - recent]
    denom = close[: n - past]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[past:] = numer / denom - 1.0
    return out


def _window_stat(
    values: np.ndarray, window: int, *, squared: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing ``window``-row sum of ``values`` and count of non-finite entries."""
    bad = ~np.isfinite(values)
    clean = np.where(bad, 0.0, values)
    if squared:
        clean = clean * clean
    n = values.shape[0]
    zero = np.zeros((1, values.shape[1]), dtype=np.float64)
    cs = np.concatenate([zero, np.cumsum(clean, axis=0)])
    cb = np.concatenate([zero, np.cumsum(bad.astype(np.float64), axis=0)])
    total = np.full(values.shape, np.nan, dtype=np.float64)
    invalid = np.full(values.shape, np.nan, dtype=np.float64)
    if n >= window:
        total[window - 1 :] = cs[window:] - cs[: n - window + 1]
        invalid[window - 1 :] = cb[window:] - cb[: n - window + 1]
    return total, invalid


def _sma(close: np.ndarray, window: int) -> np.ndarray:
    total, invalid = _window_stat(close, window)
    with np.errstate(invalid="ignore"):
        return np.where(invalid == 0.0, total / window, np.nan)


def _annualized_vol(close: np.ndarray, window: int) -> np.ndarray:
    """Population std of 4h log returns over ``window`` returns, annualised, floored."""
    n = close.shape[0]
    log_ret = np.full(close.shape, np.nan, dtype=np.float64)
    if n > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            log_ret[1:] = np.log(close[1:] / close[:-1])
    s1, bad1 = _window_stat(log_ret, window)
    s2, _ = _window_stat(log_ret, window, squared=True)
    # row 0 has no return, so the first full window ends at row ``window``
    with np.errstate(invalid="ignore"):
        mean = s1 / window
        var = np.maximum(s2 / window - mean * mean, 0.0)
        vol = np.sqrt(var) * math.sqrt(protocol.PERIODS_PER_YEAR)
    valid = (bad1 == 0.0) & np.isfinite(vol)
    return np.where(valid, np.maximum(vol, protocol.VOL_FLOOR), np.nan)


def build_signals(dataset: MarketDataset) -> LowturnSignals:
    """Precompute every signal of the screen from the dataset's close arrays."""
    pair_names = validate_alternating_spot_perp_dataset(dataset)
    enforce_pre_fs1_boundary(dataset)
    close_raw = np.asarray(dataset.close, dtype=np.float64)
    clean = np.where(np.isfinite(close_raw) & (close_raw > 0.0), close_raw, np.nan)
    usable = usable_mask(dataset)

    def gated(values: np.ndarray) -> np.ndarray:
        return np.where(usable, values, np.nan)

    return LowturnSignals(
        pair_names=pair_names,
        symbols=tuple(dataset.symbols),
        r30=_readonly(gated(_lagged_ratio(clean, 0, protocol.R30_BARS))),
        r90=_readonly(gated(_lagged_ratio(clean, 0, protocol.R90_BARS))),
        r180=_readonly(gated(_lagged_ratio(clean, 0, protocol.R180_BARS))),
        vol60=_readonly(gated(_annualized_vol(clean, protocol.VOL_WINDOW_BARS))),
        xs_score=_readonly(
            gated(_lagged_ratio(clean, protocol.XS_RECENT_BARS, protocol.XS_PAST_BARS))
        ),
        sma_fast=_readonly(gated(_sma(clean, protocol.SMA_FAST_BARS))),
        sma_slow=_readonly(gated(_sma(clean, protocol.SMA_SLOW_BARS))),
        usable=_readonly(usable),
    )


def signal_row(index: int, signal_lag_bars: int) -> int:
    """Pack row a decision at snapshot ``index`` reads (``DELAY_1BAR`` lags by one)."""
    if signal_lag_bars < 0:
        raise ValueError("signal_lag_bars must be non-negative")
    return max(0, index - signal_lag_bars)


__all__ = [
    "ROW_FIELDS",
    "LowturnSignals",
    "build_signals",
    "signal_row",
    "usable_mask",
]
