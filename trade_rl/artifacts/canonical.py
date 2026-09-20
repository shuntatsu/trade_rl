"""Standard-library canonical JSON conversion for content identities."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic, NoReturn, Self, SupportsIndex, TypeAlias, TypeVar, cast

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenDict(dict[_K, _V], Generic[_K, _V]):
    """Read-compatible dict whose contents cannot change after construction."""

    @staticmethod
    def _reject() -> NoReturn:
        raise TypeError("frozen mapping does not support mutation")

    def __setitem__(self, key: _K, value: _V) -> NoReturn:
        del key, value
        self._reject()

    def __delitem__(self, key: _K) -> NoReturn:
        del key
        self._reject()

    def clear(self) -> NoReturn:
        self._reject()

    def pop(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        self._reject()

    def popitem(self) -> NoReturn:
        self._reject()

    def setdefault(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        self._reject()

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        self._reject()

    def __ior__(self, other: object) -> Self:
        del other
        self._reject()


class FrozenList(list[_V], Generic[_V]):
    """Read-compatible list whose contents cannot change after construction."""

    @staticmethod
    def _reject() -> NoReturn:
        raise TypeError("frozen sequence does not support mutation")

    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        del key, value
        self._reject()

    def __delitem__(self, key: Any) -> NoReturn:
        del key
        self._reject()

    def __iadd__(self, other: Iterable[_V]) -> Self:
        del other
        self._reject()

    def __imul__(self, value: SupportsIndex) -> Self:
        del value
        self._reject()

    def append(self, value: _V) -> NoReturn:
        del value
        self._reject()

    def clear(self) -> NoReturn:
        self._reject()

    def extend(self, values: Any) -> NoReturn:
        del values
        self._reject()

    def insert(self, index: SupportsIndex, value: _V) -> NoReturn:
        del index, value
        self._reject()

    def pop(self, index: SupportsIndex = -1) -> NoReturn:
        del index
        self._reject()

    def remove(self, value: _V) -> NoReturn:
        del value
        self._reject()

    def reverse(self) -> NoReturn:
        self._reject()

    def sort(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        self._reject()


FrozenJsonValue: TypeAlias = JsonScalar | FrozenList[object] | FrozenDict[str, object]


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


def freeze_json_value(value: object) -> FrozenJsonValue:
    """Canonicalize one JSON-compatible value into deep read-only containers."""

    normalized = to_json_value(value)

    def freeze(item: JsonValue) -> FrozenJsonValue:
        if isinstance(item, dict):
            return FrozenDict({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return FrozenList(freeze(child) for child in item)
        return item

    return freeze(normalized)


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


__all__ = [
    "FrozenDict",
    "FrozenJsonValue",
    "FrozenList",
    "JsonScalar",
    "JsonValue",
    "canonical_json_bytes",
    "freeze_json_value",
    "to_json_value",
]
