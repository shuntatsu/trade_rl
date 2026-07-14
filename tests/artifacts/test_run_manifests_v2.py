from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    WalkForwardRunManifest,
    load_walk_forward_run_manifest,
    validate_training_run_directory,
    validate_walk_forward_run_directory,
    write_training_run_manifest,
    write_walk_forward_run_manifest,
)

NOW = datetime(2026, 7, 14, tzinfo=UTC)


def _write(root: Path, name: str, payload: bytes = b"x") -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_walk_forward_manifest_has_distinct_semantics(tmp_path: Path) -> None:
    _write(tmp_path, "walk-forward.json")
    manifest = WalkForwardRunManifest.build(
        root=tmp_path,
        run_id="wf-001",
        dataset_id="a" * 64,
        environment_digest="b" * 64,
        evaluation_digest="c" * 64,
        workflow_config_digest="d" * 64,
        policy_set_digest="e" * 64,
        provenance_digest="f" * 64,
        fold_count=3,
        artifact_paths=("walk-forward.json",),
        created_at=NOW,
    )

    write_walk_forward_run_manifest(tmp_path, manifest)

    assert load_walk_forward_run_manifest(tmp_path) == manifest
    assert validate_walk_forward_run_directory(tmp_path) == manifest
    assert manifest.schema_version == "walk_forward_run_v1"


def test_training_loader_rejects_walk_forward_manifest(tmp_path: Path) -> None:
    _write(tmp_path, "walk-forward.json")
    manifest = WalkForwardRunManifest.build(
        root=tmp_path,
        run_id="wf-001",
        dataset_id="a" * 64,
        environment_digest="b" * 64,
        evaluation_digest="c" * 64,
        workflow_config_digest="d" * 64,
        policy_set_digest="e" * 64,
        provenance_digest="f" * 64,
        fold_count=1,
        artifact_paths=("walk-forward.json",),
        created_at=NOW,
    )
    write_walk_forward_run_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="training run schema"):
        validate_training_run_directory(tmp_path)


def test_run_validation_rejects_undeclared_file_and_symlink(tmp_path: Path) -> None:
    _write(tmp_path, "ensemble.json")
    manifest = TrainingRunManifest.build(
        root=tmp_path,
        run_id="run-001",
        dataset_id="a" * 64,
        environment_digest="b" * 64,
        ensemble_digest="c" * 64,
        training_config_digest="d" * 64,
        provenance_digest="e" * 64,
        artifact_paths=("ensemble.json",),
        created_at=NOW,
    )
    write_training_run_manifest(tmp_path, manifest)
    _write(tmp_path, "undeclared.bin")

    with pytest.raises(ValueError, match="undeclared"):
        validate_training_run_directory(tmp_path)

    (tmp_path / "undeclared.bin").unlink()
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    link = tmp_path / "linked.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="symlink|undeclared"):
        validate_training_run_directory(tmp_path)
