"""PostgreSQL index for large reusable artifacts stored on durable volumes."""

from __future__ import annotations

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
            raise ValueError(
                "catalog artifact location escapes storage root"
            ) from error
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
        size_bytes = sum(
            item.stat().st_size for item in path.rglob("*") if item.is_file()
        )
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


__all__ = ["ReusableArtifactIndex"]
