"""Canonical, secret-free manifest for Universal U3-U6 runtime inputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_CACHE_ID,
    UNIVERSAL_202411_202607_TABLES,
    PostgresMarketTableSet,
)


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value.astimezone(UTC)


def _portable_relative_path(value: str | Path, *, field: str) -> Path:
    text = str(value).replace("\\", "/").strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a portable relative path")
    return Path(*path.parts)


def _symbols(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class UniversalRuntimeManifest:
    """All static identities needed to recompose one concrete Universal runtime."""

    cache_id: str
    tables: PostgresMarketTableSet
    research_start: datetime
    research_end: datetime
    instrument_artifact_relpath: Path
    dataset_artifact_relpath: Path
    normalizer_artifact_relpath: Path
    train_symbols: tuple[str, ...]
    validation_symbols: tuple[str, ...]
    test_symbols: tuple[str, ...]
    fold_train_range: tuple[int, int]
    shared_complete_row_count: int
    catalog_digest: str
    partition_digest: str
    split_manifest_digest: str
    feature_schema_digest: str
    statistics_digest: str
    metadata_evidence_digest: str
    source_manifest_digest: str
    dataset_digests: tuple[tuple[str, str], ...]
    schema_version: str = "universal_runtime_manifest_v1"
    manifest_digest: str = ""

    def __post_init__(self) -> None:
        if self.cache_id != UNIVERSAL_202411_202607_CACHE_ID:
            raise ValueError("runtime manifest cache identity mismatch")
        if self.tables != UNIVERSAL_202411_202607_TABLES:
            raise ValueError("runtime manifest table identity mismatch")
        start = _aware_utc(self.research_start, field="runtime research_start")
        end = _aware_utc(self.research_end, field="runtime research_end")
        if end <= start:
            raise ValueError("runtime research range is invalid")
        object.__setattr__(self, "research_start", start)
        object.__setattr__(self, "research_end", end)
        for field_name in (
            "instrument_artifact_relpath",
            "dataset_artifact_relpath",
            "normalizer_artifact_relpath",
        ):
            object.__setattr__(
                self,
                field_name,
                _portable_relative_path(getattr(self, field_name), field=field_name),
            )
        train = _symbols(self.train_symbols, field="runtime train_symbols")
        validation = _symbols(
            self.validation_symbols, field="runtime validation_symbols"
        )
        test = _symbols(self.test_symbols, field="runtime test_symbols")
        if (len(train), len(validation), len(test)) != (9, 3, 3):
            raise ValueError("runtime manifest partition must be 9/3/3")
        if len(set((*train, *validation, *test))) != 15:
            raise ValueError("runtime manifest partition overlaps")
        object.__setattr__(self, "train_symbols", train)
        object.__setattr__(self, "validation_symbols", validation)
        object.__setattr__(self, "test_symbols", test)
        if (
            isinstance(self.shared_complete_row_count, bool)
            or not isinstance(self.shared_complete_row_count, int)
            or self.shared_complete_row_count <= 0
            or self.fold_train_range != (0, self.shared_complete_row_count)
        ):
            raise ValueError("runtime manifest train range is not maximal shared closure")
        for item in (
            "catalog_digest",
            "partition_digest",
            "split_manifest_digest",
            "feature_schema_digest",
            "statistics_digest",
            "metadata_evidence_digest",
            "source_manifest_digest",
        ):
            require_sha256(getattr(self, item), field=f"runtime {item}")
        datasets = tuple(self.dataset_digests)
        if tuple(symbol for symbol, _ in datasets) != train:
            raise ValueError("runtime dataset digests must follow train_symbols")
        for symbol, digest in datasets:
            if not symbol:
                raise ValueError("runtime dataset symbol must not be empty")
            require_sha256(digest, field=f"runtime dataset digest {symbol}")
        object.__setattr__(self, "dataset_digests", datasets)
        if self.schema_version != "universal_runtime_manifest_v1":
            raise ValueError("runtime manifest schema mismatch")
        expected = content_digest(self.digest_payload())
        if self.manifest_digest and self.manifest_digest != expected:
            raise ValueError("runtime manifest digest mismatch")
        object.__setattr__(self, "manifest_digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "cache_id": self.cache_id,
            "catalog_digest": self.catalog_digest,
            "dataset_artifact_relpath": self.dataset_artifact_relpath.as_posix(),
            "dataset_digests": [list(item) for item in self.dataset_digests],
            "feature_schema_digest": self.feature_schema_digest,
            "fold_train_range": list(self.fold_train_range),
            "instrument_artifact_relpath": self.instrument_artifact_relpath.as_posix(),
            "metadata_evidence_digest": self.metadata_evidence_digest,
            "normalizer_artifact_relpath": self.normalizer_artifact_relpath.as_posix(),
            "partition_digest": self.partition_digest,
            "research_end": self.research_end.isoformat(),
            "research_start": self.research_start.isoformat(),
            "schema_version": self.schema_version,
            "shared_complete_row_count": self.shared_complete_row_count,
            "source_manifest_digest": self.source_manifest_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "statistics_digest": self.statistics_digest,
            "tables": asdict(self.tables),
            "test_symbols": list(self.test_symbols),
            "train_symbols": list(self.train_symbols),
            "validation_symbols": list(self.validation_symbols),
        }

    def to_json_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_json_dict(cls, value: object) -> UniversalRuntimeManifest:
        if not isinstance(value, dict):
            raise ValueError("runtime manifest must be an object")
        expected = {item.name for item in fields(cls)}
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise ValueError("runtime manifest has unknown or missing fields")
        tables = value["tables"]
        if not isinstance(tables, dict):
            raise ValueError("runtime manifest tables must be an object")
        try:
            return cls(
                cache_id=str(value["cache_id"]),
                tables=PostgresMarketTableSet(**tables),
                research_start=datetime.fromisoformat(
                    str(value["research_start"]).replace("Z", "+00:00")
                ),
                research_end=datetime.fromisoformat(
                    str(value["research_end"]).replace("Z", "+00:00")
                ),
                instrument_artifact_relpath=Path(
                    str(value["instrument_artifact_relpath"])
                ),
                dataset_artifact_relpath=Path(
                    str(value["dataset_artifact_relpath"])
                ),
                normalizer_artifact_relpath=Path(
                    str(value["normalizer_artifact_relpath"])
                ),
                train_symbols=_symbols(value["train_symbols"], field="train_symbols"),
                validation_symbols=_symbols(
                    value["validation_symbols"], field="validation_symbols"
                ),
                test_symbols=_symbols(value["test_symbols"], field="test_symbols"),
                fold_train_range=tuple(value["fold_train_range"]),
                shared_complete_row_count=value["shared_complete_row_count"],
                catalog_digest=str(value["catalog_digest"]),
                partition_digest=str(value["partition_digest"]),
                split_manifest_digest=str(value["split_manifest_digest"]),
                feature_schema_digest=str(value["feature_schema_digest"]),
                statistics_digest=str(value["statistics_digest"]),
                metadata_evidence_digest=str(value["metadata_evidence_digest"]),
                source_manifest_digest=str(value["source_manifest_digest"]),
                dataset_digests=tuple(
                    (str(item[0]), str(item[1]))
                    for item in value["dataset_digests"]
                ),
                schema_version=str(value["schema_version"]),
                manifest_digest=str(value["manifest_digest"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError("runtime manifest field types are invalid") from error


def write_universal_runtime_manifest(
    path: str | Path, manifest: UniversalRuntimeManifest
) -> Path:
    output = Path(path)
    payload = canonical_json_bytes(manifest.to_json_dict()) + b"\n"
    if output.exists():
        if output.is_file() and output.read_bytes() == payload:
            return output
        raise FileExistsError("runtime manifest already exists with different content")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return output


def load_universal_runtime_manifest(path: str | Path) -> UniversalRuntimeManifest:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("runtime manifest JSON is invalid") from error
    return UniversalRuntimeManifest.from_json_dict(payload)


__all__ = [
    "UniversalRuntimeManifest",
    "load_universal_runtime_manifest",
    "write_universal_runtime_manifest",
]
