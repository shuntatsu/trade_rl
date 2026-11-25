"""Explicit non-circular overlapping-block percentile intervals.

Unlike the existing circular moving-block hypothesis test, this method
returns common resample indices so callers can pair several return paths.
It contains no study roster, significance gate, or execution authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def _require_plain_native_array(value: np.ndarray, *, field_name: str) -> np.ndarray:
    """Reject masks, subclasses and opaque owners before reading numeric values.

    Exact ndarray/memoryview owner links can be inspected without invoking
    arbitrary user attributes. Owned arrays, native views, and exact raw-byte
    buffers remain supported. Missingness already erased by the caller cannot
    be recovered.
    """
    error = f"{field_name} must be a native array without opaque or masked ownership"
    if type(value) is not np.ndarray:
        raise ValueError(error)
    owner: object = value
    while True:
        if type(owner) is np.ndarray:
            owner = np.ndarray.base.__get__(owner)
        elif type(owner) is memoryview:
            owner = owner.obj
        else:
            break
    if owner is not None and type(owner) is not bytes and type(owner) is not bytearray:
        raise ValueError(error)
    return value


def _integer(value: object, *, minimum: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return resolved


def _real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        resolved = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def moving_block_indices(
    n_values: int, *, block: int, resamples: int, seed: int
) -> np.ndarray:
    """Generate non-wrapping blocks; use the same indices for all paired paths."""
    n_values = _integer(n_values, minimum=2, name="n_values")
    block = min(_integer(block, minimum=1, name="block"), n_values)
    resamples = _integer(resamples, minimum=1, name="resamples")
    seed = _integer(seed, minimum=0, name="seed")
    n_blocks = -(-n_values // block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_values - block + 1, size=(resamples, n_blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(resamples, -1)
    return indices[:, :n_values]


@dataclass(frozen=True)
class BootstrapInterval:
    """Annualized mean and percentile intervals, not a significance decision."""

    mean_daily: float
    annualized_mean: float
    ci_low: float
    ci_high: float
    bonferroni_ci_low: float
    bonferroni_ci_high: float
    alpha: float
    comparisons: int

    def to_mapping(self) -> dict[str, float]:
        """Frozen 95% mapping, refused for a differently configured interval."""
        if self.alpha != 0.05:
            raise ValueError("legacy ci95 field names require alpha=0.05")
        return {
            "mean_daily_excess": self.mean_daily,
            "annualized_mean_excess": self.annualized_mean,
            "ci95_low_annualized": self.ci_low,
            "ci95_high_annualized": self.ci_high,
            "bonferroni_ci_low_annualized": self.bonferroni_ci_low,
            "bonferroni_ci_high_annualized": self.bonferroni_ci_high,
        }


def bootstrap_excess(
    excess_daily: np.ndarray,
    indices: np.ndarray,
    *,
    trading_days_per_year: float,
    alpha: float,
    comparisons: int,
) -> BootstrapInterval:
    """Paired mean daily excess, simple annualization and two-sided percentiles.

    The caller fixes the resampling plan before evaluating any candidate.
    Reject invalid rows instead of silently filling or dropping observations.
    """
    raw = _require_plain_native_array(excess_daily, field_name="excess")
    if raw.dtype.kind not in "iuf":
        raise ValueError("excess must contain real numeric values")
    excess = np.asarray(raw, dtype=np.float64)
    if excess.ndim != 1 or excess.size < 2 or not np.isfinite(excess).all():
        raise ValueError("excess must be a finite one-dimensional daily series")
    idx = _require_plain_native_array(indices, field_name="bootstrap indices")
    if (
        idx.ndim != 2
        or idx.shape[0] < 1
        or idx.shape[1] != excess.size
        or idx.dtype.kind not in "iu"
        or np.any(idx < 0)
        or np.any(idx >= excess.size)
    ):
        raise ValueError("bootstrap indices must be an in-bounds integer matrix")
    scale = _real(trading_days_per_year, name="trading_days_per_year")
    if scale <= 0:
        raise ValueError("trading_days_per_year must be positive and finite")
    alpha = _real(alpha, name="alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    comparisons = _integer(comparisons, minimum=1, name="comparisons")
    try:
        adjusted = alpha / comparisons
    except OverflowError as exc:
        raise ValueError("comparisons are too large") from exc
    if adjusted / 2.0 <= 0:
        raise ValueError("adjusted alpha underflow")
    with np.errstate(over="ignore", invalid="ignore"):
        means = excess[idx].mean(axis=1) * scale
        mean_daily = float(excess.mean())
        annual = mean_daily * scale
    if not np.isfinite(means).all() or not math.isfinite(annual):
        raise ValueError("annualized bootstrap means must be finite")
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    b_low, b_high = np.quantile(means, [adjusted / 2.0, 1.0 - adjusted / 2.0])
    if not np.isfinite([low, high, b_low, b_high]).all():
        raise ValueError("bootstrap interval must be finite")
    return BootstrapInterval(
        mean_daily,
        annual,
        float(low),
        float(high),
        float(b_low),
        float(b_high),
        alpha,
        comparisons,
    )
