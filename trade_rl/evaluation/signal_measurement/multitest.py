"""Multiple-testing corrections and the block-bootstrap interval (Issue #1146, SD1).

Holm (family-wise error) and Benjamini-Hochberg (false discovery rate) as *adjusted
p-values*, so a caller compares them with a level instead of re-deriving a cutoff.
Equality counts as a rejection (``adjusted <= level``), matching the SD0 verdict
rule. Both procedures are order independent: permuting the input permutes the output.

The interval wrapper reuses the repository's ``moving_block_mean_test`` without
changing it, so the protocol's 20-day blocks, 2000 resamples and seed 0 are
arguments of the caller, not constants hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

import numpy as np

from trade_rl.evaluation.comparison.bootstrap import moving_block_mean_test
from trade_rl.evaluation.signal_measurement.ic import SignalMeasurementError


def _p_values(values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] == 0 or array.dtype.kind not in "fiu":
        raise SignalMeasurementError("p-values must be a non-empty 1-D numeric array")
    p = array.astype(np.float64, copy=False)
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise SignalMeasurementError("p-values must be finite and lie in [0, 1]")
    return p


def _level(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SignalMeasurementError(f"{name} must be a real number")
    level = float(value)
    if not 0.0 < level < 1.0:
        raise SignalMeasurementError(f"{name} must lie strictly between 0 and 1")
    return level


def holm_adjusted(p_values: object) -> np.ndarray:
    """Holm step-down adjusted p-values: ``max_{j<=k} (m - j + 1) p_(j)``, capped at 1."""

    p = _p_values(p_values)
    count = p.shape[0]
    order = np.argsort(p, kind="mergesort")
    scaled = (count - np.arange(count)) * p[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    out = np.empty(count, dtype=np.float64)
    out[order] = adjusted_sorted
    return out


def bh_adjusted(p_values: object) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values: ``min_{j>=k} m p_(j) / j``, capped at 1."""

    p = _p_values(p_values)
    count = p.shape[0]
    order = np.argsort(p, kind="mergesort")
    scaled = p[order] * count / (np.arange(count) + 1.0)
    adjusted_sorted = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    out = np.empty(count, dtype=np.float64)
    out[order] = adjusted_sorted
    return out


def holm_rejections(p_values: object, alpha: float) -> np.ndarray:
    """Boolean mask of hypotheses Holm rejects at family-wise level ``alpha``."""

    return holm_adjusted(p_values) <= _level(alpha, "alpha")


def bh_rejections(p_values: object, q: float) -> np.ndarray:
    """Boolean mask of hypotheses Benjamini-Hochberg rejects at FDR level ``q``."""

    return bh_adjusted(p_values) <= _level(q, "q")


@dataclass(frozen=True, slots=True)
class BlockMeanResult:
    """Mean of a day series with its moving-block bootstrap interval and p-value."""

    n_days: int
    mean: float
    lower: float
    upper: float
    p_value_one_sided: float
    block_days: int


def block_mean_interval(
    daily_values: object, *, block_days: int, resamples: int, seed: int
) -> BlockMeanResult:
    """95% moving-block bootstrap interval of a day series' mean (blocks in days).

    The one-sided p-value tests a positive mean. The 95% level is fixed by the shared
    ``moving_block_mean_test``; a different level needs a different test, not a flag.
    """

    array = np.asarray(daily_values)
    if array.ndim != 1 or array.shape[0] == 0 or array.dtype.kind not in "fiu":
        raise SignalMeasurementError("daily_values must be a non-empty 1-D array")
    values = array.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise SignalMeasurementError("daily_values must be finite")
    if (
        not isinstance(block_days, int)
        or isinstance(block_days, bool)
        or block_days <= 0
    ):
        raise SignalMeasurementError("block_days must be a positive integer")
    if block_days > values.shape[0]:
        raise SignalMeasurementError(
            f"block_days ({block_days}) must not exceed the number of days ({values.shape[0]})"
        )
    result = moving_block_mean_test(
        tuple(float(v) for v in values),
        n_bootstrap=resamples,
        seed=seed,
        block_size=block_days,
    )
    return BlockMeanResult(
        n_days=int(values.shape[0]),
        mean=fmean(float(v) for v in values),
        lower=result.lower_ci,
        upper=result.upper_ci,
        p_value_one_sided=result.p_value,
        block_days=result.block_size,
    )


__all__ = [
    "BlockMeanResult",
    "bh_adjusted",
    "bh_rejections",
    "block_mean_interval",
    "holm_adjusted",
    "holm_rejections",
]
