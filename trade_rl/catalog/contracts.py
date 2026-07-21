"""Framework-independent contracts for reusable research artifact catalogs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, Protocol, TypeAlias

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


class ArtifactKind(StrEnum):
    RAW_MARKET_DATA = "raw_market_data"
    MARKET_DATASET = "market_dataset"
    NATIVE_FEATURES = "native_features"
    ALIGNED_FEATURES = "aligned_features"
    SEQUENCE_PLANE = "sequence_plane"
    NORMALIZER = "normalizer"
    ORACLE_TEACHER = "oracle_teacher"
    CHECKPOINT = "checkpoint"
    MODEL = "model"
    RESEARCH_RUN = "research_run"


class ArtifactStatus(StrEnum):
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


def _require_digest(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_non_empty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _freeze_json(value: object, *, field_name: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain finite JSON values")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} must contain JSON object string keys")
            frozen[key] = _freeze_json(item, field_name=field_name)
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field_name=field_name) for item in value)
    raise ValueError(f"{field_name} must contain only JSON values")


def thaw_json(value: object) -> object:
    """Return ordinary JSON-compatible containers for database adapters."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_bytes(
    value: Mapping[str, JsonValue] | Mapping[str, object],
) -> bytes:
    frozen = _freeze_json(value, field_name="JSON payload")
    if not isinstance(frozen, Mapping):
        raise ValueError("JSON payload must be an object")
    return json.dumps(
        thaw_json(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def cache_key_digest(cache_key: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(cache_key)).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRegistration:
    artifact_digest: str
    artifact_kind: ArtifactKind
    schema_version: str
    cache_key: Mapping[str, JsonValue] | Mapping[str, object]
    metadata: Mapping[str, JsonValue] | Mapping[str, object]
    location: str
    size_bytes: int
    dataset_id: str | None = None
    status: ArtifactStatus = ArtifactStatus.READY
    cache_key_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require_digest(self.artifact_digest, field_name="artifact_digest")
        if self.dataset_id is not None:
            _require_digest(self.dataset_id, field_name="dataset_id")
        _require_non_empty(self.schema_version, field_name="schema_version")
        _require_non_empty(self.location, field_name="location")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("size_bytes must be a non-negative integer")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        frozen_cache = _freeze_json(self.cache_key, field_name="cache_key JSON")
        frozen_metadata = _freeze_json(self.metadata, field_name="metadata JSON")
        if not isinstance(frozen_cache, Mapping) or not isinstance(
            frozen_metadata, Mapping
        ):
            raise ValueError("cache_key and metadata JSON must be objects")
        object.__setattr__(self, "cache_key", frozen_cache)
        object.__setattr__(self, "metadata", frozen_metadata)
        object.__setattr__(
            self,
            "cache_key_digest",
            hashlib.sha256(canonical_json_bytes(frozen_cache)).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    registration: ArtifactRegistration
    created_at: datetime
    last_seen_at: datetime

    def __post_init__(self) -> None:
        for name, value in (
            ("created_at", self.created_at),
            ("last_seen_at", self.last_seen_at),
        ):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must include a timezone")


@dataclass(frozen=True, slots=True)
class ArtifactQuery:
    artifact_kind: ArtifactKind | None = None
    dataset_id: str | None = None
    status: ArtifactStatus | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if self.dataset_id is not None:
            _require_digest(self.dataset_id, field_name="dataset_id")
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or not 1 <= self.limit <= 10_000
        ):
            raise ValueError("limit must be an integer within [1, 10000]")


class ArtifactCatalog(Protocol):
    def migrate(self) -> tuple[int, ...]: ...

    def health(self) -> Mapping[str, object]: ...

    def register(self, registration: ArtifactRegistration) -> ArtifactRecord: ...

    def find(
        self, artifact_kind: ArtifactKind, cache_key: Mapping[str, object]
    ) -> ArtifactRecord | None: ...

    def list(
        self, query: ArtifactQuery = ArtifactQuery()
    ) -> tuple[ArtifactRecord, ...]: ...

    def add_dependency(
        self, parent_digest: str, child_digest: str, role: str
    ) -> None: ...


__all__ = [
    "ArtifactCatalog",
    "ArtifactKind",
    "ArtifactQuery",
    "ArtifactRecord",
    "ArtifactRegistration",
    "ArtifactStatus",
    "JsonValue",
    "cache_key_digest",
    "canonical_json_bytes",
    "thaw_json",
]
