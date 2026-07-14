from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    load_training_run_manifest,
    validate_training_run_directory,
    write_training_run_manifest,
)


def _build(root: Path) -> TrainingRunManifest:
    member = root / "members" / "member-000" / "policy.zip"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"checkpoint")
    ensemble = root / "ensemble.json"
    ensemble.write_text('{"schema":"ensemble"}', encoding="utf-8")
    return TrainingRunManifest.build(
        root=root,
        run_id="run-001",
        dataset_id="a" * 64,
        environment_digest="b" * 64,
        ensemble_digest="c" * 64,
        training_config_digest="d" * 64,
        provenance_digest="e" * 64,
        artifact_paths=(
            "ensemble.json",
            "members/member-000/policy.zip",
        ),
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )


def test_training_run_manifest_round_trip_and_validation(tmp_path: Path) -> None:
    manifest = _build(tmp_path)

    path = write_training_run_manifest(tmp_path, manifest)
    loaded = load_training_run_manifest(tmp_path)
    validated = validate_training_run_directory(tmp_path)

    assert path == tmp_path / "run.json"
    assert (
        path.read_bytes()
        == write_training_run_manifest(tmp_path, manifest).read_bytes()
    )
    assert loaded == manifest
    assert validated == manifest
    assert tuple(item.path for item in manifest.files) == (
        "ensemble.json",
        "members/member-000/policy.zip",
    )


def test_training_run_validation_rejects_changed_file(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    write_training_run_manifest(tmp_path, manifest)
    (tmp_path / "members" / "member-000" / "policy.zip").write_bytes(b"changed")

    with pytest.raises(ValueError, match="digest|size"):
        validate_training_run_directory(tmp_path)


def test_training_run_manifest_rejects_unsafe_paths(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="relative|unsafe"):
        TrainingRunManifest.build(
            root=tmp_path,
            run_id="run-001",
            dataset_id="a" * 64,
            environment_digest="b" * 64,
            ensemble_digest="c" * 64,
            training_config_digest="d" * 64,
            provenance_digest="e" * 64,
            artifact_paths=("../outside.bin",),
            created_at=datetime(2026, 7, 14, tzinfo=UTC),
        )


def test_training_run_validation_rejects_missing_file(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    write_training_run_manifest(tmp_path, manifest)
    (tmp_path / "ensemble.json").unlink()

    with pytest.raises(ValueError, match="missing"):
        validate_training_run_directory(tmp_path)
