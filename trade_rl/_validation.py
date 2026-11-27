"""Standard-library validation helpers shared by the lean research core."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Final

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")


def require_non_empty(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be non-empty")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def require_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def require_git_sha(value: str, *, field: str = "git_commit") -> str:
    if not isinstance(value, str) or not _GIT_SHA_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def require_aware_datetime(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be timezone-aware")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def require_unique_non_empty(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field} must not be empty")
    normalized = tuple(require_non_empty(value, field=field) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique values")
    return normalized


def _ordered_ns_durations(boundaries_ns: tuple[int, ...]) -> tuple[int, ...]:
    """Validate one finite ordered boundary sequence using Python integers."""
    if (
        type(boundaries_ns) is not tuple
        or len(boundaries_ns) < 2
        or any(type(x) is not int or not -(2**63) < x < 2**63 for x in boundaries_ns)
    ):
        raise ValueError("session boundaries must be finite native nanoseconds")
    durations = tuple(
        after - before
        for before, after in zip(boundaries_ns[:-1], boundaries_ns[1:], strict=True)
    )
    if any(not 0 < duration <= (2**63 - 1) for duration in durations):
        raise ValueError("session intervals must be positive representable nanoseconds")
    return durations


def _require_nominal_session_seconds(
    nominal_seconds: int, durations_ns: tuple[int, ...]
) -> None:
    if (
        type(nominal_seconds) is not int
        or not 0 < nominal_seconds <= (2**63 - 1) // 1_000_000_000
        or any(d < nominal_seconds * 1_000_000_000 for d in durations_ns)
    ):
        raise ValueError(
            "session requires whole-second nominal bars within each interval"
        )


__all__ = [
    "require_aware_datetime",
    "require_git_sha",
    "require_non_empty",
    "require_sha256",
    "require_unique_non_empty",
]
