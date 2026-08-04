"""PostgreSQL index for large reusable artifacts stored on durable volumes."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.catalog.contracts import (
    ArtifactCatalog,
    ArtifactKind,
    ArtifactRegistration,
    ArtifactStatus,
)
from trade_rl.catalog.postgres import PostgresArtifactCatalog


def teacher_cache_identity(
    *,
    dataset_id: str,
    train_range: tuple[int, int],
    environment_digest: str,
    action_spec_digest: str,
    teacher_config_digest: str,
) -> dict[str, object]:
    return {
        "action_spec_digest": action_spec_digest,
        "dataset_id": dataset_id,
        "environment_digest": environment_digest,
        "schema_version": "teacher_cache_identity_v1",
        "teacher_config_digest": teacher_config_digest,
        "train_range": train_range,
    }


class ReusableArtifactIndex:
    """Resolve trusted file-backed artifacts through the shared SQL catalog."""

    def __init__(self, catalog: ArtifactCatalog, *, storage_root: Path) -> None:
        self.catalog = catalog
        self.storage_root = storage_root.resolve()

    @classmethod
    def from_environment(cls, *, storage_root: Path) -> ReusableArtifactIndex | None:
        enabled = os.environ.get("TRADE_RL_REUSABLE_ARTIFACT_CATALOG", "").lower()
        if enabled not in {"true", "1", "yes"}:
            return None
        database_url = os.environ.get("TRADE_RL_DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeError(
                "reusable artifact catalog requires TRADE_RL_DATABASE_URL"
            )
        catalog = PostgresArtifactCatalog(database_url)
        catalog.migrate()
        return cls(catalog, storage_root=storage_root)

    def _trusted_path(self, value: str) -> Path:
        path = Path(value).resolve()
        try:
            path.relative_to(self.storage_root)
        except ValueError as error:
            raise ValueError("catalog artifact location escapes storage root") from error
        return path

    def resolve(
        self,
        artifact_kind: ArtifactKind,
        cache_key: Mapping[str, object],
    ) -> Path | None:
        record = self.catalog.find(artifact_kind, cache_key)
        if record is None or record.registration.status is not ArtifactStatus.READY:
            return None
        path = self._trusted_path(record.registration.location)
        return path if path.is_dir() else None

    def register_directory(
        self,
        *,
        artifact_digest: str,
        artifact_kind: ArtifactKind,
        schema_version: str,
        dataset_id: str | None,
        cache_key: Mapping[str, object],
        metadata: Mapping[str, object],
        location: Path,
    ) -> None:
        path = self._trusted_path(str(location))
        if not path.is_dir():
            raise FileNotFoundError(f"reusable artifact directory is absent: {path}")
        size_bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        existing = self.catalog.find(artifact_kind, cache_key)
        if existing is not None:
            registration = existing.registration
            recorded_payload_digest = str(
                registration.metadata.get(
                    "payload_digest",
                    registration.artifact_digest,
                )
            )
            if recorded_payload_digest != artifact_digest:
                raise ValueError(
                    "catalog cache identity resolves to different artifact content"
                )
            if registration.schema_version != schema_version:
                raise ValueError(
                    "catalog cache identity resolves to different artifact schema"
                )
            if registration.dataset_id != dataset_id:
                raise ValueError(
                    "catalog cache identity resolves to different dataset identity"
                )
            if self._trusted_path(registration.location) != path:
                raise ValueError(
                    "catalog cache identity resolves to different artifact location"
                )
            if registration.size_bytes != size_bytes:
                raise ValueError(
                    "catalog cache identity resolves to different artifact size"
                )
            # Preserve the immutable legacy/new registration byte-for-byte and
            # let the repository refresh only last_seen_at.
            self.catalog.register(registration)
            return
        catalog_digest = content_digest(
            {
                "artifact_kind": artifact_kind.value,
                "cache_key": cache_key,
                "payload_digest": artifact_digest,
                "schema_version": "reusable_artifact_catalog_entry_v1",
            }
        )
        self.catalog.register(
            ArtifactRegistration(
                artifact_digest=catalog_digest,
                artifact_kind=artifact_kind,
                schema_version=schema_version,
                dataset_id=dataset_id,
                cache_key=cache_key,
                metadata={**metadata, "payload_digest": artifact_digest},
                location=str(path),
                size_bytes=size_bytes,
            )
        )


def backfill_teacher_cache(index: ReusableArtifactIndex) -> int:
    """Validate and index completed Teacher directories already on the volume."""

    from trade_rl.learning.episode_teacher_artifact import (
        EPISODE_TEACHER_ARTIFACT_SCHEMA,
        load_episode_teacher_artifact,
    )
    from trade_rl.learning.teacher_artifact import load_teacher_artifact

    registered = 0
    for path in sorted(index.storage_root.iterdir()):
        manifest_path = path / "manifest.json"
        if (
            not path.is_dir()
            or path.name.startswith(".")
            or not manifest_path.is_file()
        ):
            continue
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("teacher cache manifest must be an object")
        schema_version = str(raw.get("schema_version", ""))
        if schema_version == EPISODE_TEACHER_ARTIFACT_SCHEMA:
            manifest, _ = load_episode_teacher_artifact(path)
            metadata = {
                "episode_count": manifest.episode_count,
                "sample_count": manifest.sample_count,
            }
        else:
            manifest, _ = load_teacher_artifact(path)
            metadata = {"sample_count": manifest.sample_count}
        cache_key = teacher_cache_identity(
            dataset_id=manifest.dataset_id,
            train_range=(manifest.train_start, manifest.train_stop),
            environment_digest=manifest.environment_digest,
            action_spec_digest=manifest.action_spec_digest,
            teacher_config_digest=manifest.teacher_config_digest,
        )
        index.register_directory(
            artifact_digest=manifest.artifact_digest,
            artifact_kind=ArtifactKind.ORACLE_TEACHER,
            schema_version=manifest.schema_version,
            dataset_id=manifest.dataset_id,
            cache_key=cache_key,
            metadata=metadata,
            location=path,
        )
        registered += 1
    return registered


__all__ = [
    "ReusableArtifactIndex",
    "backfill_teacher_cache",
    "teacher_cache_identity",
]
