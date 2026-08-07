"""Standard-library canonical JSON conversion for content identities."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias, cast

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _datetime_value(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime values must be timezone-aware")
    normalized = value.astimezone(UTC).isoformat()
    return normalized.removesuffix("+00:00") + "Z"


def _mapping_value(value: Mapping[object, object]) -> dict[str, JsonValue]:
    converted: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("canonical JSON mapping keys must be strings")
        converted[key] = to_json_value(item)
    return converted


def _dataclass_value(value: object) -> dict[str, JsonValue]:
    dataclass_value = cast(Any, value)
    return {
        field.name: to_json_value(getattr(dataclass_value, field.name))
        for field in fields(dataclass_value)
    }


def to_json_value(value: object) -> JsonValue:
    """Convert a supported object into a deterministic JSON value tree."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON floats must be finite")
        return value
    if isinstance(value, datetime):
        return _datetime_value(value)
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_value(value)
    if isinstance(value, Mapping):
        return _mapping_value(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [to_json_value(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode a supported value as stable UTF-8 canonical JSON bytes."""

    normalized = to_json_value(value)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


__all__ = ["JsonScalar", "JsonValue", "canonical_json_bytes", "to_json_value"]
