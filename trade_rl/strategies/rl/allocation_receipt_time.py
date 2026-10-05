"""Exact calendar and nanosecond boundaries for allocation provenance receipts."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import numpy as np

_ISO = re.compile(
    r"(?P<calendar>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?(?P<zone>Z|[+-]\d{2}:\d{2})?"
)
_NS_LIMITS = np.iinfo(np.int64)


def _calendar(value: object, *, precision: int, aware: bool) -> tuple[datetime, int]:
    if not isinstance(value, str) or (match := _ISO.fullmatch(value)) is None:
        raise ValueError("receipt timestamp must use a complete ISO calendar")
    fraction, zone = match["fraction"] or "", match["zone"]
    if len(fraction) > precision:
        raise ValueError("receipt timestamp exceeds its supported precision")
    if aware and zone is None:
        raise ValueError("objective timestamp must declare its UTC offset")
    microseconds = fraction[:6].ljust(6, "0")
    suffix = "+00:00" if zone in (None, "Z") else zone
    try:
        time = datetime.fromisoformat(f"{match['calendar']}.{microseconds}{suffix}")
        time = time.astimezone(UTC)
    except (ValueError, OverflowError) as error:
        raise ValueError("receipt timestamp calendar is invalid") from error
    remainder = int(fraction[6:].ljust(3, "0") or "0")
    return time, remainder


def receipt_datetime_ns(time: datetime) -> np.datetime64:
    """Check original UTC microseconds before any narrowing nanosecond cast."""
    if time.tzinfo is None or time.utcoffset() is None:
        raise ValueError("receipt timestamp must be timezone-aware")
    try:
        coarse = np.datetime64(time.astimezone(UTC).replace(tzinfo=None), "us")
    except (ValueError, OverflowError) as error:
        raise ValueError("receipt timestamp calendar is invalid") from error
    ticks = int(coarse.astype(np.int64)) * 1000
    result = _nanosecond_ticks(ticks)
    if result.astype("datetime64[us]") != coarse:
        raise ValueError("receipt timestamp does not roundtrip in nanoseconds")
    return result


def _nanosecond_ticks(ticks: int) -> np.datetime64:
    if not int(_NS_LIMITS.min) < ticks <= int(_NS_LIMITS.max):
        raise ValueError("receipt timestamp is outside finite nanosecond range")
    return np.datetime64(ticks, "ns")


def parse_objective_datetime(value: object) -> datetime:
    time, _ = _calendar(value, precision=6, aware=True)
    receipt_datetime_ns(time)
    return time


def parse_source_timestamp(value: object) -> np.datetime64:
    """Preserve all declared ns digits; a naive source calendar denotes UTC."""
    time, remainder = _calendar(value, precision=9, aware=False)
    coarse = np.datetime64(time.replace(tzinfo=None), "us")
    # Python integers cannot wrap. Validate the complete fractional timestamp,
    # including the sub-microsecond tail, before constructing datetime64[ns].
    ticks = int(coarse.astype(np.int64)) * 1000 + remainder
    return _nanosecond_ticks(ticks)
