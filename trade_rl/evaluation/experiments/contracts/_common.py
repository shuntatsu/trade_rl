"""Shared validation helpers for immutable experiment contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime

from trade_rl._validation import require_aware_datetime, require_sha256
from trade_rl.evaluation.experiments.errors import ContractViolationError


def contract_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractViolationError(f"{field} must be non-empty")
    return value


def contract_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ContractViolationError(f"{field} must be a lowercase SHA-256 digest")
    try:
        return require_sha256(value, field=field)
    except ValueError as error:
        raise ContractViolationError(str(error)) from error


def contract_aware_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ContractViolationError(f"{field} must be timezone-aware")
    try:
        return require_aware_datetime(value, field=field)
    except ValueError as error:
        raise ContractViolationError(str(error)) from error


def contract_non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractViolationError(f"{field} must be a non-negative integer")
    return value


def contract_positive_int(
    value: object,
    *,
    field: str,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractViolationError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ContractViolationError(f"{field} must be <= {maximum}")
    return value


def contract_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolationError(f"{field} must be finite")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ContractViolationError(f"{field} must be finite")
    return resolved


def contract_unique_texts(values: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ContractViolationError(f"{field} must be a non-empty tuple")
    result = tuple(contract_text(value, field=field) for value in values)
    if len(set(result)) != len(result):
        raise ContractViolationError(f"{field} must contain unique values")
    return result


def contract_sha256_tuple(
    values: object,
    *,
    field: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ContractViolationError(f"{field} must be a tuple")
    if not allow_empty and not values:
        raise ContractViolationError(f"{field} must not be empty")
    result = tuple(contract_sha256(value, field=field) for value in values)
    if len(set(result)) != len(result):
        raise ContractViolationError(f"{field} must contain unique values")
    return result


def contract_non_negative_int_tuple(
    values: object,
    *,
    field: str,
    minimum_items: int = 1,
) -> tuple[int, ...]:
    if not isinstance(values, tuple) or len(values) < minimum_items:
        raise ContractViolationError(
            f"{field} must contain at least {minimum_items} values"
        )
    result = tuple(contract_non_negative_int(value, field=field) for value in values)
    if len(set(result)) != len(result):
        raise ContractViolationError(f"{field} must contain unique values")
    if result != tuple(sorted(result)):
        raise ContractViolationError(f"{field} must use canonical ascending order")
    return result


def contract_unique_enum_tuple(
    values: object,
    *,
    field: str,
    expected_type: type,
) -> tuple[object, ...]:
    if not isinstance(values, tuple) or not values:
        raise ContractViolationError(f"{field} must be a non-empty tuple")
    if any(not isinstance(value, expected_type) for value in values):
        raise ContractViolationError(f"{field} contains an unsupported value")
    if len(set(values)) != len(values):
        raise ContractViolationError(f"{field} must contain unique values")
    return tuple(values)


def payload_tuple_strings(values: Iterable[str]) -> list[str]:
    return list(values)


__all__ = [
    "contract_aware_datetime",
    "contract_finite",
    "contract_non_negative_int",
    "contract_non_negative_int_tuple",
    "contract_positive_int",
    "contract_sha256",
    "contract_sha256_tuple",
    "contract_text",
    "contract_unique_enum_tuple",
    "contract_unique_texts",
    "payload_tuple_strings",
]
