"""Rank and linear information coefficients (Issue #1146, card SD1).

Pure numpy statistics over arrays the caller already holds. Nothing here reads a
dataset, a file or the network, and nothing knows what a feature or a market is.

Conventions fixed by the signal-diagnosis protocol (SD0), restated here as code:

* a row is *eligible* only when its score and its target are both finite; every
  other row is dropped, never imputed;
* the rank IC is the Pearson correlation of average ranks (Spearman, ties share
  the mean rank); the linear IC is the Pearson correlation of the raw values;
* a cross-section (one UTC day) with fewer than two eligible rows or a constant
  score or target has **no** IC. Such a day is skipped and counted, never filled
  with zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

_NS_PER_DAY = 86_400_000_000_000

IcMethod = Literal["rank", "linear"]


class SignalMeasurementError(ValueError):
    """An input to a measurement function is malformed."""


def _vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise SignalMeasurementError(f"{name} must be one-dimensional")
    if array.dtype.kind not in "fiu":
        raise SignalMeasurementError(f"{name} must be numeric")
    return array.astype(np.float64, copy=False)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks where tied values share the mean of the ranks they span."""

    array = _vector(values, "values")
    if not np.isfinite(array).all():
        raise SignalMeasurementError("values must be finite to be ranked")
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start = 0
    count = array.shape[0]
    while start < count:
        stop = start + 1
        while stop < count and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape[0] < 2:
        return None
    centred_left = left - left.mean()
    centred_right = right - right.mean()
    denominator = float(
        np.sqrt(
            np.dot(centred_left, centred_left) * np.dot(centred_right, centred_right)
        )
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    value = float(np.dot(centred_left, centred_right)) / denominator
    return float(min(1.0, max(-1.0, value)))


def _eligible(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    if scores.shape != target.shape:
        raise SignalMeasurementError("scores and target must have the same length")
    return np.isfinite(scores) & np.isfinite(target)


def linear_ic(scores: object, target: object) -> float | None:
    """Pearson correlation on the eligible rows, or ``None`` if undefined."""

    left = _vector(scores, "scores")
    right = _vector(target, "target")
    keep = _eligible(left, right)
    return _pearson(left[keep], right[keep])


def rank_ic(scores: object, target: object) -> float | None:
    """Spearman correlation (average ranks) on the eligible rows, or ``None``."""

    left = _vector(scores, "scores")
    right = _vector(target, "target")
    keep = _eligible(left, right)
    if int(keep.sum()) < 2:
        return None
    return _pearson(average_ranks(left[keep]), average_ranks(right[keep]))


def utc_day_ids(timestamps_ns: object) -> np.ndarray:
    """Whole UTC days since the epoch for integer nanosecond timestamps."""

    array = np.asarray(timestamps_ns)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise SignalMeasurementError("timestamps_ns must be a 1-D integer array")
    return (array.astype(np.int64) // _NS_PER_DAY).astype(np.int64)


@dataclass(frozen=True, slots=True)
class DailyIc:
    """Per-day IC values in ascending day order plus the days that had none."""

    day_ids: tuple[int, ...]
    values: tuple[float, ...]
    skipped_days: int

    def __post_init__(self) -> None:
        if len(self.day_ids) != len(self.values):
            raise SignalMeasurementError("day_ids and values must align")

    @property
    def mean(self) -> float | None:
        return float(np.mean(self.values)) if self.values else None


def _day_groups(day_ids: np.ndarray, length: int) -> list[tuple[int, np.ndarray]]:
    if day_ids.ndim != 1 or day_ids.dtype.kind not in "iu":
        raise SignalMeasurementError("day_ids must be a 1-D integer array")
    if day_ids.shape[0] != length:
        raise SignalMeasurementError("day_ids must align with the rows")
    order = np.argsort(day_ids, kind="stable")
    sorted_days = day_ids[order]
    unique_days, starts = np.unique(sorted_days, return_index=True)
    stops = np.append(starts[1:], sorted_days.shape[0])
    return [
        (int(day), order[start:stop])
        for day, start, stop in zip(unique_days, starts, stops, strict=True)
    ]


def _ic(method: IcMethod, scores: np.ndarray, target: np.ndarray) -> float | None:
    if method == "rank":
        return rank_ic(scores, target)
    if method == "linear":
        return linear_ic(scores, target)
    raise SignalMeasurementError("method must be 'rank' or 'linear'")


def daily_ic(
    day_ids: object, scores: object, target: object, *, method: IcMethod
) -> DailyIc:
    """IC pooled over all rows of each UTC day; undefined days are skipped and counted."""

    days = np.asarray(day_ids)
    left = _vector(scores, "scores")
    right = _vector(target, "target")
    _eligible(left, right)
    kept_days: list[int] = []
    values: list[float] = []
    skipped = 0
    for day, rows in _day_groups(days, left.shape[0]):
        value = _ic(method, left[rows], right[rows])
        if value is None:
            skipped += 1
        else:
            kept_days.append(day)
            values.append(value)
    return DailyIc(tuple(kept_days), tuple(values), skipped)


def daily_rank_ic_increment(
    day_ids: object, forecast: object, baseline: object, target: object
) -> DailyIc:
    """Daily rank IC of the forecast minus that of the baseline on the same rows.

    Both ICs are formed on the rows where the forecast, the baseline and the target
    are all finite, so the two numbers always describe identical rows. A day on
    which either IC is undefined is skipped and counted.
    """

    days = np.asarray(day_ids)
    model = _vector(forecast, "forecast")
    base = _vector(baseline, "baseline")
    truth = _vector(target, "target")
    if not model.shape == base.shape == truth.shape:
        raise SignalMeasurementError("forecast, baseline and target must align")
    common = np.isfinite(model) & np.isfinite(base) & np.isfinite(truth)
    kept_days: list[int] = []
    values: list[float] = []
    skipped = 0
    for day, rows in _day_groups(days, model.shape[0]):
        rows = rows[common[rows]]
        model_ic = rank_ic(model[rows], truth[rows])
        base_ic = rank_ic(base[rows], truth[rows])
        if model_ic is None or base_ic is None:
            skipped += 1
        else:
            kept_days.append(day)
            values.append(model_ic - base_ic)
    return DailyIc(tuple(kept_days), tuple(values), skipped)


__all__ = [
    "DailyIc",
    "IcMethod",
    "SignalMeasurementError",
    "average_ranks",
    "daily_ic",
    "daily_rank_ic_increment",
    "linear_ic",
    "rank_ic",
    "utc_day_ids",
]
