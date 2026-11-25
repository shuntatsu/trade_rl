"""Public lossless UTC scalar time boundary for external research consumers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import numpy as np

from trade_rl._validation import require_aware_datetime
from trade_rl.data._timestamps import (
    _MAX_DATETIME64_NS,
    _MIN_FINITE_DATETIME64_NS,
    _datetime64_values_to_ns_exact,
)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def utc_timestamp_ns(value: datetime | np.datetime64) -> int:
    """Return signed nanoseconds since the UTC Unix epoch, with no rounding.

    Native Python datetimes must have a real UTC offset. Native NumPy
    datetime64 scalars are already UTC-labelled; their original unit and
    multiplier are checked by the shared exact converter. Reject ambiguous
    or unrepresentable inputs, NaT, and sub-nanosecond values. Subclasses
    may carry additional precision or semantics and are not coerced.
    This boundary cannot recover precision lost by the caller beforehand.
    """
    if type(value) is datetime:
        require_aware_datetime(value, field="timestamp")
        try:
            elapsed = value.astimezone(UTC) - _EPOCH
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "timestamp is outside the finite nanosecond range"
            ) from exc
        result = (
            elapsed.days * 86_400 + elapsed.seconds
        ) * 1_000_000_000 + elapsed.microseconds * 1_000
        if not _MIN_FINITE_DATETIME64_NS <= result <= _MAX_DATETIME64_NS:
            raise ValueError("timestamp is outside the finite nanosecond range")
        return result
    if type(value) is np.datetime64:
        scalar = cast(
            np.datetime64, _datetime64_values_to_ns_exact(np.asarray(value))[()]
        )
        return int(scalar.astype(np.int64))
    raise TypeError("timestamp must be a native datetime or datetime64 scalar")


__all__ = ["utc_timestamp_ns"]
