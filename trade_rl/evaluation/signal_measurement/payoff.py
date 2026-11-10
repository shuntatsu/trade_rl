"""Signed after-cost payoff and the drift-excess payoff (Issue #1146, card SD1; layer D4).

Vectorised numpy; a deliberately simple loop reference lives in the tests and shares no
code with this module. The rules restate the signal-diagnosis plan:

* a unit position ``s`` in {-1, 0, +1} is held for the label horizon and earns
  ``s * r - |s| * cost`` with ``cost = round_trip_cost * cost_multiplier``;
* the signal policy takes ``s = sign(forecast)`` only when ``|forecast| > k * cost``
  (strictly greater: equality stays flat); otherwise it stays flat and pays nothing;
* the drift comparator takes ``s = sign(drift forecast)`` and is charged **no** cost,
  a deliberately favourable comparator, so a payoff that merely follows the market
  drift earns nothing extra;
* ``drift_excess = signal payoff - comparator payoff`` on the same row at the same cost
  multiplier. It is a payoff relative to a drift-following policy, **not** a
  beta-adjusted alpha;
* a row with a missing forecast, drift forecast or label is ineligible (``NaN``), never
  flat and never filled.

``round_trip_cost`` is a fraction of notional (14 bp is ``0.0014``) supplied by the
caller; this module carries no protocol constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.evaluation.signal_measurement.ic import (
    SignalMeasurementError,
    _day_groups,
)


def _vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "fiu":
        raise SignalMeasurementError(f"{name} must be a 1-D numeric array")
    return array.astype(np.float64, copy=False)


def _aligned(*arrays: np.ndarray) -> None:
    if len({array.shape for array in arrays}) != 1:
        raise SignalMeasurementError("inputs must have the same length")


def _cost(
    k: object, round_trip_cost: object, cost_multiplier: object
) -> tuple[float, float]:
    if type(k) is not int or k < 1:
        raise SignalMeasurementError("k must be a positive integer")
    if type(cost_multiplier) is not int or cost_multiplier < 1:
        raise SignalMeasurementError("cost_multiplier must be a positive integer")
    if isinstance(round_trip_cost, bool) or not isinstance(
        round_trip_cost, (int, float)
    ):
        raise SignalMeasurementError("round_trip_cost must be a positive finite number")
    try:
        unit_cost = float(round_trip_cost)
        factor = float(cost_multiplier)
        k_factor = float(k)
    except OverflowError as error:
        raise SignalMeasurementError(
            "derived cost and threshold must be finite"
        ) from error
    if not math.isfinite(unit_cost) or unit_cost <= 0.0:
        raise SignalMeasurementError("round_trip_cost must be a positive finite number")
    cost = unit_cost * factor
    threshold = cost * k_factor
    if not math.isfinite(cost) or not math.isfinite(threshold):
        raise SignalMeasurementError("derived cost and threshold must be finite")
    return threshold, cost

def _positions(forecast: np.ndarray, threshold: float) -> np.ndarray:
    """sign(forecast) where ``|forecast| > threshold``, else 0 (NaN stays NaN)."""

    signed = np.sign(forecast)
    return np.where(np.abs(forecast) > threshold, signed, 0.0)


def after_cost_payoff(
    forecast: object,
    realised: object,
    *,
    k: int,
    round_trip_cost: float,
    cost_multiplier: int,
) -> np.ndarray:
    """``s * r - |s| * cost`` per row; ``NaN`` where the forecast or label is missing."""

    model = _vector(forecast, "forecast")
    label = _vector(realised, "realised")
    _aligned(model, label)
    threshold, cost = _cost(k, round_trip_cost, cost_multiplier)
    position = _positions(model, threshold)
    payoff = position * label - np.abs(position) * cost
    eligible = np.isfinite(model) & np.isfinite(label)
    return np.where(eligible, payoff, np.nan)


def drift_sign_payoff(drift_forecast: object, realised: object) -> np.ndarray:
    """The zero-cost comparator ``sign(drift) * r``; ``NaN`` where an input is missing."""

    drift = _vector(drift_forecast, "drift_forecast")
    label = _vector(realised, "realised")
    _aligned(drift, label)
    eligible = np.isfinite(drift) & np.isfinite(label)
    return np.where(eligible, np.sign(drift) * label, np.nan)


def drift_excess_payoff(
    forecast: object,
    drift_forecast: object,
    realised: object,
    *,
    k: int,
    round_trip_cost: float,
    cost_multiplier: int,
) -> np.ndarray:
    """Signal after-cost payoff minus the zero-cost drift-sign payoff, row by row."""

    model = _vector(forecast, "forecast")
    drift = _vector(drift_forecast, "drift_forecast")
    label = _vector(realised, "realised")
    _aligned(model, drift, label)
    signal = after_cost_payoff(
        model,
        label,
        k=k,
        round_trip_cost=round_trip_cost,
        cost_multiplier=cost_multiplier,
    )
    return signal - drift_sign_payoff(drift, label)


@dataclass(frozen=True, slots=True)
class DailyMean:
    """Mean of the eligible rows of each UTC day; days with none are skipped, counted."""

    day_ids: tuple[int, ...]
    values: tuple[float, ...]
    skipped_days: int


def daily_mean(day_ids: object, values: object) -> DailyMean:
    """Average a per-row series over each day's finite rows (the bootstrap's unit)."""

    days = np.asarray(day_ids)
    series = _vector(values, "values")
    kept_days: list[int] = []
    means: list[float] = []
    skipped = 0
    for day, rows in _day_groups(days, series.shape[0]):
        finite = series[rows][np.isfinite(series[rows])]
        if finite.shape[0] == 0:
            skipped += 1
        else:
            kept_days.append(day)
            means.append(float(finite.mean()))
    return DailyMean(tuple(kept_days), tuple(means), skipped)


__all__ = [
    "DailyMean",
    "after_cost_payoff",
    "daily_mean",
    "drift_excess_payoff",
    "drift_sign_payoff",
]
