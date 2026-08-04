from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.catalog.contracts import (
    ArtifactKind,
    ArtifactRecord,
    ArtifactRegistration,
)
from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex


class _Catalog:
    def __init__(self) -> None:
        self.records: dict[tuple[ArtifactKind, str], ArtifactRecord] = {}
        self.registered: ArtifactRegistration | None = None

    def find(self, artifact_kind: ArtifactKind, cache_key: object) -> ArtifactRecord | None:
        assert isinstance(cache_key, dict)
        return self.records.get((artifact_kind, str(cache_key["identity"])))

    def register(self, registration: ArtifactRegistration) -> ArtifactRecord:
        self.registered = registration
        now = datetime.now(UTC)
        record = ArtifactRecord(
            registration=registration,
            created_at=now,
            last_seen_at=now,
        )
        self.records[(registration.artifact_kind, str(registration.cache_key["identity"]))] = record
        return record


def _record(path: Path) -> ArtifactRecord:
    registration = ArtifactRegistration(
        artifact_digest="a" * 64,
        artifact_kind=ArtifactKind.ORACLE_TEACHER,
        schema_version="episode_supervised_teacher_artifact_v1",
        dataset_id="b" * 64,
        cache_key={"identity": "teacher-1"},
        metadata={"sample_count": 10},
        location=str(path),
        size_bytes=4,
    )
    now = datetime.now(UTC)
    return ArtifactRecord(registration=registration, created_at=now, last_seen_at=now)


def test_resolves_ready_directory_and_registers_file_backed_artifact(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    artifact = root / "teacher"
    artifact.mkdir(parents=True)
    (artifact / "arrays.npz").write_bytes(b"data")
    catalog = _Catalog()
    catalog.records[(ArtifactKind.ORACLE_TEACHER, "teacher-1")] = _record(artifact)
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    assert index.resolve(ArtifactKind.ORACLE_TEACHER, {"identity": "teacher-1"}) == artifact
    index.register_directory(
        artifact_digest="a" * 64,
        artifact_kind=ArtifactKind.ORACLE_TEACHER,
        schema_version="episode_supervised_teacher_artifact_v1",
        dataset_id="b" * 64,
        cache_key={"identity": "teacher-1"},
        metadata={"sample_count": 10},
        location=artifact,
    )

    assert catalog.registered is not None
    assert catalog.registered.size_bytes == 4


def test_new_registration_namespaces_payload_digest_by_cache_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    artifact = root / "teacher"
    artifact.mkdir(parents=True)
    (artifact / "arrays.npz").write_bytes(b"data")
    catalog = _Catalog()
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    index.register_directory(
        artifact_digest="a" * 64,
        artifact_kind=ArtifactKind.ORACLE_TEACHER,
        schema_version="episode_supervised_teacher_artifact_v1",
        dataset_id="b" * 64,
        cache_key={"identity": "teacher-2"},
        metadata={"sample_count": 10},
        location=artifact,
    )

    assert catalog.registered is not None
    assert catalog.registered.artifact_digest != "a" * 64
    assert catalog.registered.metadata["payload_digest"] == "a" * 64


def test_refreshes_legacy_registration_without_mutating_metadata(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    artifact = root / "teacher"
    artifact.mkdir(parents=True)
    (artifact / "arrays.npz").write_bytes(b"data")
    catalog = _Catalog()
    legacy = _record(artifact)
    catalog.records[(ArtifactKind.ORACLE_TEACHER, "teacher-1")] = legacy
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    index.register_directory(
        artifact_digest="a" * 64,
        artifact_kind=ArtifactKind.ORACLE_TEACHER,
        schema_version="episode_supervised_teacher_artifact_v1",
        dataset_id="b" * 64,
        cache_key={"identity": "teacher-1"},
        metadata={"sample_count": 10, "new_runtime_field": True},
        location=artifact,
    )

    assert catalog.registered == legacy.registration


def test_rejects_catalog_location_outside_durable_root(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    catalog = _Catalog()
    catalog.records[(ArtifactKind.ORACLE_TEACHER, "teacher-1")] = _record(outside)
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="escapes storage root"):
        index.resolve(ArtifactKind.ORACLE_TEACHER, {"identity": "teacher-1"})
