"""Validation primitives for constrained-policy report evidence."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Final

_SCHEMA_VERSION: Final = "constrained_policy_report_v1"
_BUDGET_TOLERANCE: Final = 1e-12
_DIAGNOSTIC_FIELDS: Final = (
    "raw_estimate",
    "ema_estimate",
    "multiplier_mean",
    "multiplier_max",
    "upper_cap_fraction",
    "lower_bound_fraction",
    "cost_critic_explained_variance",
    "cost_critic_loss",
)


def _require_non_empty(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def _require_sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return value.lower()


def _require_finite(
    value: float,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return resolved


def _optional_finite(
    value: float | None,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _require_finite(
        value,
        field=field,
        minimum=minimum,
        maximum=maximum,
    )


def _require_integer(value: int, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _normalized_digests(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    normalized = tuple(
        _require_sha256(value, field=f"{field}[{index}]")
        for index, value in enumerate(tuple(values))
    )
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique digests")
    return normalized


def _complete_mean(values: tuple[float | None, ...]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(fmean(float(value) for value in values if value is not None))


def _complete_max(values: tuple[float | None, ...]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return max(float(value) for value in values if value is not None)
