"""Deliberately simple, independent reference for the signed after-cost payoff (#1146).

Plain Python floats and one ``for`` loop per row; no numpy vectorisation, no shared
helper with the fast implementation. The rules restated here come from the plan, not
from the code under test:

* a position ``s`` in {-1, 0, +1} is held for the label horizon and earns
  ``s * r - |s| * cost`` where ``cost = round_trip_cost * cost_multiplier``;
* the signal policy takes ``s = sign(forecast)`` only when ``|forecast| > k * cost``
  (strictly greater), otherwise it stays flat and pays nothing;
* the drift comparator takes ``s = sign(drift forecast)`` on every row and is charged
  **no** cost (a deliberately favourable comparator);
* the drift excess is the signal payoff minus the comparator payoff on the same row;
* a row with a missing forecast, drift forecast or label is not eligible (``None``).
"""

from __future__ import annotations

import math


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _present(*values: float) -> bool:
    return all(not math.isnan(v) and not math.isinf(v) for v in values)


def slow_after_cost_payoff(
    forecasts: list[float],
    realised: list[float],
    *,
    k: int,
    round_trip_cost: float,
    cost_multiplier: int,
) -> list[float | None]:
    cost = round_trip_cost * cost_multiplier
    out: list[float | None] = []
    for forecast, label in zip(forecasts, realised, strict=True):
        if not _present(forecast, label):
            out.append(None)
            continue
        position = _sign(forecast) if abs(forecast) > k * cost else 0
        out.append(position * label - abs(position) * cost)
    return out


def slow_drift_sign_payoff(
    drift_forecasts: list[float], realised: list[float]
) -> list[float | None]:
    out: list[float | None] = []
    for drift, label in zip(drift_forecasts, realised, strict=True):
        if not _present(drift, label):
            out.append(None)
            continue
        out.append(_sign(drift) * label)
    return out


def slow_drift_excess_payoff(
    forecasts: list[float],
    drift_forecasts: list[float],
    realised: list[float],
    *,
    k: int,
    round_trip_cost: float,
    cost_multiplier: int,
) -> list[float | None]:
    cost = round_trip_cost * cost_multiplier
    out: list[float | None] = []
    for forecast, drift, label in zip(
        forecasts, drift_forecasts, realised, strict=True
    ):
        if not _present(forecast, drift, label):
            out.append(None)
            continue
        position = _sign(forecast) if abs(forecast) > k * cost else 0
        signal = position * label - abs(position) * cost
        comparator = _sign(drift) * label
        out.append(signal - comparator)
    return out


def slow_daily_mean(day_ids: list[int], values: list[float | None]) -> dict[int, float]:
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for day, value in zip(day_ids, values, strict=True):
        if value is None:
            continue
        totals[day] = totals.get(day, 0.0) + value
        counts[day] = counts.get(day, 0) + 1
    return {day: totals[day] / counts[day] for day in sorted(totals)}
