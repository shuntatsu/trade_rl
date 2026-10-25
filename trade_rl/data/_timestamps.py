"""Exact datetime64 normalization for internal data boundaries."""

from __future__ import annotations

import math
from datetime import date

import numpy as np

_MIN_FINITE_DATETIME64_NS = -(2**63) + 1
_MAX_DATETIME64_NS = (2**63) - 1
_NANOSECONDS_PER_DAY = 86_400_000_000_000
_EPOCH_DATE = date(1970, 1, 1)
_FIXED_UNIT_NANOSECONDS = {
    "W": (604_800_000_000_000, 1),
    "D": (86_400_000_000_000, 1),
    "h": (3_600_000_000_000, 1),
    "m": (60_000_000_000, 1),
    "s": (1_000_000_000, 1),
    "ms": (1_000_000, 1),
    "us": (1_000, 1),
    "ns": (1, 1),
    "ps": (1, 1_000),
    "fs": (1, 1_000_000),
    "as": (1, 1_000_000_000),
}


def _datetime64_values_to_ns_exact(values: np.ndarray) -> np.ndarray:
    """Normalize datetime64 values with a positive unit multiplier exactly."""

    timestamps = np.asarray(values)
    if timestamps.dtype.kind != "M":
        raise ValueError("timestamps must use a datetime64 dtype")
    if np.isnat(timestamps).any():
        raise ValueError("timestamps must not contain NaT")

    unit, multiplier = np.datetime_data(timestamps.dtype)
    if multiplier <= 0:
        raise ValueError("datetime64 unit multiplier must be positive")
    ticks = timestamps.view(np.dtype(f"{timestamps.dtype.byteorder}i8"))
    if unit in ("Y", "M"):
        normalized_ticks: list[int] = []
        for raw_tick in ticks.reshape(-1):
            offset = int(raw_tick) * multiplier
            if unit == "Y":
                year = 1970 + offset
                month = 1
                if year < 1678 or year > 2262:
                    raise ValueError(
                        "datetime64 value is outside the finite nanosecond range"
                    )
            else:
                year = 1970 + offset // 12
                month = offset % 12 + 1
                if (year, month) < (1677, 10) or (year, month) > (2262, 4):
                    raise ValueError(
                        "datetime64 value is outside the finite nanosecond range"
                    )
            normalized = (date(year, month, 1) - _EPOCH_DATE).days
            normalized_ticks.append(normalized * _NANOSECONDS_PER_DAY)
        nanoseconds = np.asarray(normalized_ticks, dtype=np.int64).reshape(
            timestamps.shape
        )
    else:
        unit_ratio = _FIXED_UNIT_NANOSECONDS.get(unit)
        if unit_ratio is None:
            raise ValueError(
                "datetime64 unit cannot be normalized exactly to nanoseconds"
            )
        numerator, denominator = unit_ratio
        numerator *= multiplier
        common = math.gcd(numerator, denominator)
        numerator //= common
        denominator //= common

        exact_ticks = ticks
        if denominator != 1:
            if np.any(np.remainder(ticks, denominator) != 0):
                raise ValueError("datetime64 value has sub-nanosecond precision")
            exact_ticks = np.floor_divide(ticks, denominator)

        minimum_tick = -((-_MIN_FINITE_DATETIME64_NS) // numerator)
        maximum_tick = _MAX_DATETIME64_NS // numerator
        if np.any(exact_ticks < minimum_tick) or np.any(exact_ticks > maximum_tick):
            raise ValueError("datetime64 value is outside the finite nanosecond range")
        if numerator > _MAX_DATETIME64_NS:
            nanoseconds = np.zeros(timestamps.shape, dtype=np.int64)
        else:
            nanoseconds = np.asarray(exact_ticks, dtype=np.int64) * np.int64(numerator)

    return nanoseconds.view(np.dtype("datetime64[ns]"))
