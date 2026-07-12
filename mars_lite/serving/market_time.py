"""Completed-bar selection and timeframe-aware market-data freshness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mars_lite.data.data_utils import TF_TO_MINUTES

SUPPORTED_BASE_TIMEFRAMES = frozenset({"15m", "1h", "4h", "1d"})


@dataclass(frozen=True)
class CompletedBarEndpoint:
    end_exclusive: int
    latest_bar_close: np.datetime64
    data_age_hours: float


def resolve_completed_bar_endpoint(
    timestamps: np.ndarray,
    *,
    base_timeframe: str,
    now_utc: np.datetime64,
) -> CompletedBarEndpoint:
    """Return the last bar whose close is no later than the supplied UTC clock."""

    if base_timeframe not in SUPPORTED_BASE_TIMEFRAMES:
        raise ValueError(f"unsupported base timeframe: {base_timeframe!r}")
    timestamp_values = np.asarray(timestamps, dtype="datetime64[ns]")
    if timestamp_values.ndim != 1 or timestamp_values.size == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional array")
    timestamp_ints = timestamp_values.astype("int64")
    if np.any(timestamp_ints == np.iinfo(np.int64).min):
        raise ValueError("timestamps must not contain NaT")
    if timestamp_values.size > 1 and np.any(np.diff(timestamp_ints) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    now_value = np.asarray(now_utc, dtype="datetime64[ns]")
    if now_value.ndim != 0 or np.isnat(now_value):
        raise ValueError("now_utc must be a valid scalar timestamp")
    now_ns = np.datetime64(now_value, "ns")
    duration = np.timedelta64(TF_TO_MINUTES[base_timeframe], "m")
    bar_closes = timestamp_values + duration
    completed = np.flatnonzero(bar_closes <= now_ns)
    if completed.size == 0:
        raise ValueError("no completed bar is available")

    last_index = int(completed[-1])
    latest_close = np.datetime64(bar_closes[last_index], "ns")
    age_hours = float((now_ns - latest_close) / np.timedelta64(1, "h"))
    if age_hours < 0:
        raise ValueError("completed-bar age cannot be negative")
    return CompletedBarEndpoint(
        end_exclusive=last_index + 1,
        latest_bar_close=latest_close,
        data_age_hours=age_hours,
    )
