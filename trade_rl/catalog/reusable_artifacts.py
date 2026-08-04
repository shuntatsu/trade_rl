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
from trade_rl.learning.oracle_bellman_contracts import OracleSolverProvenance
from trade_rl.learning.oracle_market_tape import ORACLE_MARKET_TAPE_SCHEMA


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


def teacher_cache_identity_v2(
    *,
    dataset_id: str,
    train_range: tuple[int, int],
    environment_digest: str,
    action_spec_digest: str,
    teacher_config_digest: str,
    solver_provenance: OracleSolverProvenance,
) -> dict[str, object]:
    """Return stable solver-aware cache identity for newly generated teachers."""

    if not isinstance(solver_provenance, OracleSolverProvenance):
        raise ValueError("solver_provenance must be OracleSolverProvenance")
    return {
        "action_spec_digest": action_spec_digest,
        "compile_chunk_size": solver_provenance.compile_chunk_size,
        "compile_mode": solver_provenance.compile_mode,
        "dataset_id": dataset_id,
        "environment_digest": environment_digest,
        "episode_batch_size": solver_provenance.episode_batch_size,
        "fallback_reason": solver_provenance.fallback_reason,
        "market_tape_digest": solver_provenance.market_tape_digest,
        "market_tape_schema": ORACLE_MARKET_TAPE_SCHEMA,
        "numeric_dtype": solver_provenance.numeric_dtype,
        "oom_retry_performed": solver_provenance.oom_retry_performed,
        "schema_version": "teacher_cache_identity_v2",
        "solver_backend": solver_provenance.backend,
        "solver_contract": solver_provenance.solver_contract,
        "target_state_block_size": solver_provenance.target_state_block_size,
        "teacher_config_digest": teacher_config_digest,
        "tie_break_contract": solver_provenance.tie_break_contract,
        "tie_tolerance": solver_provenance.tie_tolerance,
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


def backfill_teacher_cache(index: ReusableArtifactIndex) -> int:
    """Validate and index completed Teacher directories already on the volume."""

    from trade_rl.learning.episode_teacher_artifact import (
        EPISODE_TEACHER_ARTIFACT_SCHEMA,
        EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
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
        if schema_version in {
            EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
            EPISODE_TEACHER_ARTIFACT_SCHEMA,
        }:
            episode_manifest, _ = load_episode_teacher_artifact(path)
            artifact_digest = episode_manifest.artifact_digest
            manifest_schema = episode_manifest.schema_version
            dataset_id = episode_manifest.dataset_id
            train_range = (episode_manifest.train_start, episode_manifest.train_stop)
            environment_digest = episode_manifest.environment_digest
            action_spec_digest = episode_manifest.action_spec_digest
            teacher_config_digest = episode_manifest.teacher_config_digest
            solver_provenance = episode_manifest.solver_provenance
            metadata: dict[str, object] = {
                "episode_count": episode_manifest.episode_count,
                "sample_count": episode_manifest.sample_count,
            }
            if solver_provenance is not None:
                metadata["solver_provenance"] = solver_provenance.serialized_payload()
        else:
            teacher_manifest, _ = load_teacher_artifact(path)
            artifact_digest = teacher_manifest.artifact_digest
            manifest_schema = teacher_manifest.schema_version
            dataset_id = teacher_manifest.dataset_id
            train_range = (teacher_manifest.train_start, teacher_manifest.train_stop)
            environment_digest = teacher_manifest.environment_digest
            action_spec_digest = teacher_manifest.action_spec_digest
            teacher_config_digest = teacher_manifest.teacher_config_digest
            solver_provenance = None
            metadata = {"sample_count": teacher_manifest.sample_count}
        cache_key = (
            teacher_cache_identity(
                dataset_id=dataset_id,
                train_range=train_range,
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_config_digest,
            )
            if solver_provenance is None
            else teacher_cache_identity_v2(
                dataset_id=dataset_id,
                train_range=train_range,
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_config_digest,
                solver_provenance=solver_provenance,
            )
        )
        index.register_directory(
            artifact_digest=artifact_digest,
            artifact_kind=ArtifactKind.ORACLE_TEACHER,
            schema_version=manifest_schema,
            dataset_id=dataset_id,
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
    "teacher_cache_identity_v2",
]
