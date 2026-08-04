from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_teacher_artifact import (
    EPISODE_TEACHER_ARTIFACT_SCHEMA,
    EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
    EPISODE_TEACHER_ARTIFACT_SCHEMA_V2,
    EpisodeSupervisedPolicyDataset,
    load_episode_teacher_artifact,
    write_episode_teacher_artifact,
)
from trade_rl.learning.oracle_bellman_contracts import (
    OracleSolverConfig,
    OracleSolverProvenance,
)


def _dataset(
    *, provenance: OracleSolverProvenance | None
) -> EpisodeSupervisedPolicyDataset:
    return EpisodeSupervisedPolicyDataset(
        observations=np.zeros((2, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
        dataset_id="a" * 64,
        train_start=1,
        train_stop=4,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
        decision_indices=np.array([1, 2], dtype=np.int64),
        episode_ids=np.array([0, 0], dtype=np.int64),
        solver_provenance=provenance,
    )


def _provenance() -> OracleSolverProvenance:
    return replace(
        OracleSolverProvenance.numpy_reference(
            config=OracleSolverConfig(),
            market_tape_digest="e" * 64,
        ),
        solver_wall_time_seconds=0.25,
        peak_host_memory_bytes=4096,
        digest="",
    )


def test_legacy_artifact_round_trip_does_not_fabricate_solver_provenance(
    tmp_path: Path,
) -> None:
    write_episode_teacher_artifact(tmp_path, _dataset(provenance=None))

    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest, dataset = load_episode_teacher_artifact(tmp_path)

    assert raw["schema_version"] == EPISODE_TEACHER_ARTIFACT_SCHEMA_V1
    assert "solver_provenance" not in raw
    assert "solver_provenance_digest" not in raw
    assert manifest.solver_provenance is None
    assert manifest.solver_provenance_digest is None
    assert dataset.solver_provenance is None


def test_v3_artifact_round_trip_preserves_complete_solver_provenance(
    tmp_path: Path,
) -> None:
    provenance = _provenance()
    write_episode_teacher_artifact(tmp_path, _dataset(provenance=provenance))

    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest, dataset = load_episode_teacher_artifact(tmp_path)

    assert raw["schema_version"] == EPISODE_TEACHER_ARTIFACT_SCHEMA
    assert raw["solver_provenance"]["backend"] == "numpy"
    assert raw["solver_provenance"]["solver_wall_time_seconds"] == 0.25
    assert len(raw["solver_provenance_digest"]) == 64
    assert manifest.solver_provenance == provenance
    assert manifest.solver_provenance_digest == raw["solver_provenance_digest"]
    assert dataset.solver_provenance == provenance


def test_runtime_evidence_changes_integrity_digest_not_artifact_identity(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_provenance = _provenance()
    second_provenance = replace(
        first_provenance,
        solver_wall_time_seconds=9.0,
        peak_host_memory_bytes=8192,
        digest="",
    )

    first_digest = write_episode_teacher_artifact(
        first_root,
        _dataset(provenance=first_provenance),
    )
    second_digest = write_episode_teacher_artifact(
        second_root,
        _dataset(provenance=second_provenance),
    )
    first_raw = json.loads(
        (first_root / "manifest.json").read_text(encoding="utf-8")
    )
    second_raw = json.loads(
        (second_root / "manifest.json").read_text(encoding="utf-8")
    )

    assert first_provenance.digest == second_provenance.digest
    assert first_digest == second_digest
    assert (
        first_raw["solver_provenance_digest"]
        != second_raw["solver_provenance_digest"]
    )


def test_legacy_v2_artifact_without_runtime_digest_remains_readable(
    tmp_path: Path,
) -> None:
    provenance = _provenance()
    write_episode_teacher_artifact(tmp_path, _dataset(provenance=provenance))
    manifest_path = tmp_path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["schema_version"] = EPISODE_TEACHER_ARTIFACT_SCHEMA_V2
    raw.pop("solver_provenance_digest")
    identity_payload = {
        key: value
        for key, value in raw.items()
        if key not in {"artifact_digest", "solver_provenance"}
    }
    raw["artifact_digest"] = content_digest(identity_payload)
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    manifest, dataset = load_episode_teacher_artifact(tmp_path)

    assert manifest.schema_version == EPISODE_TEACHER_ARTIFACT_SCHEMA_V2
    assert manifest.solver_provenance == provenance
    assert manifest.solver_provenance_digest is None
    assert dataset.solver_provenance == provenance


def test_v3_artifact_rejects_tampered_solver_provenance(tmp_path: Path) -> None:
    write_episode_teacher_artifact(tmp_path, _dataset(provenance=_provenance()))
    manifest_path = tmp_path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["solver_provenance"]["solver_wall_time_seconds"] = 9.0
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        load_episode_teacher_artifact(tmp_path)
