"""Oracle teacher cache identities and artifact backfill orchestration."""

from __future__ import annotations

import json

from trade_rl.catalog.contracts import ArtifactKind
from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex
from trade_rl.learning.episode_teacher_artifact import (
    EPISODE_TEACHER_ARTIFACT_SCHEMA,
    EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
    load_episode_teacher_artifact,
)
from trade_rl.learning.oracle_bellman_contracts import OracleSolverProvenance
from trade_rl.learning.oracle_market_tape import ORACLE_MARKET_TAPE_SCHEMA
from trade_rl.learning.teacher_artifact import load_teacher_artifact


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
    """Return stable numerical identity independent of runtime execution details."""

    if not isinstance(solver_provenance, OracleSolverProvenance):
        raise ValueError("solver_provenance must be OracleSolverProvenance")
    return {
        "action_spec_digest": action_spec_digest,
        "dataset_id": dataset_id,
        "environment_digest": environment_digest,
        "market_tape_schema": ORACLE_MARKET_TAPE_SCHEMA,
        "schema_version": "teacher_cache_identity_v2",
        "solver_identity": {
            **solver_provenance.identity_payload(),
            "digest": solver_provenance.digest,
        },
        "teacher_config_digest": teacher_config_digest,
        "train_range": train_range,
    }


def backfill_teacher_cache(index: ReusableArtifactIndex) -> int:
    """Validate and index completed Teacher directories already on the volume."""

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
    "backfill_teacher_cache",
    "teacher_cache_identity",
    "teacher_cache_identity_v2",
]
