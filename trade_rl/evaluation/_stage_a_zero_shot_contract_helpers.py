"""Validation helpers and schema constants for Stage A zero-shot contracts."""

from __future__ import annotations

import math
from typing import Final, Literal

from trade_rl.domain.common import require_non_empty, require_sha256

STAGE_A_CANDIDATE_SCHEMA: Final = "stage_a_zero_shot_candidate_v1"
STAGE_A_EVALUATION_PLAN_SCHEMA: Final = "stage_a_zero_shot_evaluation_plan_v2"
STAGE_A_OBSERVATION_SCHEMA: Final = "stage_a_zero_shot_observation_v2"
STAGE_A_EVIDENCE_SCHEMA: Final = "stage_a_zero_shot_evidence_v2"
MAX_STAGE_A_BOOTSTRAP_RESAMPLES: Final = 1_000_000

StageAEvaluationSplit = Literal["validation", "test"]
_SPLITS: Final = frozenset({"validation", "test"})


def _non_negative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(
    value: int,
    *,
    field: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return value


def _finite(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _fraction(value: float, *, field: str) -> float:
    result = _finite(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return result


def _unique_ints(
    values: tuple[int, ...], *, field: str, minimum_count: int = 1
) -> tuple[int, ...]:
    if len(values) < minimum_count:
        raise ValueError(f"{field} must contain at least {minimum_count} values")
    normalized = tuple(_non_negative_int(value, field=field) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(normalized))


def _unique_strings(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field} must not be empty")
    normalized = tuple(require_non_empty(value, field=field) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(normalized))


def _unique_digests(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    normalized = _unique_strings(values, field=field)
    for value in normalized:
        require_sha256(value, field=field)
    return normalized


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _require_fields(
    payload: dict[str, object], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} field closure mismatch")


