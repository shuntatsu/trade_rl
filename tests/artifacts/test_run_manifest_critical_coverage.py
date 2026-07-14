from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import trade_rl.artifacts.run_manifest as run_manifest

NOW = datetime(2026, 7, 14, tzinfo=UTC)
SHA = "a" * 64


def _run_file(path: str = "artifact.bin", *, size: int = 1) -> run_manifest.RunFile:
    return run_manifest.RunFile(path=path, digest=SHA, size_bytes=size)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "run.json"])
def test_run_file_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        _run_file(path)


@pytest.mark.parametrize("size", [True, -1, 1.5])
def test_run_file_rejects_invalid_sizes(size: object) -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        run_manifest.RunFile(path="a", digest=SHA, size_bytes=size)  # type: ignore[arg-type]


def _common_kwargs() -> dict[str, object]:
    return {
        "digest": SHA,
        "run_id": "run-001",
        "dataset_id": "b" * 64,
        "environment_digest": "c" * 64,
        "provenance_digest": "d" * 64,
        "files": (_run_file(),),
        "created_at": NOW,
        "production_status": "NO-GO",
    }


def test_common_run_manifest_validation_rejects_invalid_contracts() -> None:
    base = _common_kwargs()
    cases = (
        ({"run_id": ".."}, "run_id"),
        ({"files": ()}, "declare"),
        ({"files": (_run_file("b"), _run_file("a"))}, "ordering"),
        ({"files": (_run_file("a"), _run_file("a"))}, "unique"),
        ({"created_at": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"production_status": "GO"}, "NO-GO"),
    )
    for override, message in cases:
        with pytest.raises(ValueError, match=message):
            run_manifest._validate_common(**{**base, **override})  # type: ignore[arg-type]


def _write(root: Path, name: str, payload: bytes = b"x") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _training_manifest(root: Path) -> run_manifest.TrainingRunManifest:
    _write(root, "artifact.bin")
    return run_manifest.TrainingRunManifest.build(
        root=root,
        run_id="run-001",
        dataset_id="a" * 64,
        environment_digest="b" * 64,
        ensemble_digest="c" * 64,
        training_config_digest="d" * 64,
        provenance_digest="e" * 64,
        artifact_paths=("artifact.bin",),
        created_at=NOW,
    )


def _walk_forward_manifest(root: Path) -> run_manifest.WalkForwardRunManifest:
    _write(root, "walk-forward.json")
    return run_manifest.WalkForwardRunManifest.build(
        root=root,
        run_id="wf-001",
        dataset_id="a" * 64,
        environment_digest="b" * 64,
        evaluation_digest="c" * 64,
        workflow_config_digest="d" * 64,
        policy_set_digest="e" * 64,
        provenance_digest="f" * 64,
        fold_count=2,
        artifact_paths=("walk-forward.json",),
        created_at=NOW,
    )


def test_manifest_dataclasses_reject_semantic_and_digest_tampering(tmp_path: Path) -> None:
    training = _training_manifest(tmp_path / "training")
    with pytest.raises(ValueError, match="ensemble_digest"):
        replace(training, ensemble_digest="bad")
    with pytest.raises(ValueError, match="training_config_digest"):
        replace(training, training_config_digest="bad")
    with pytest.raises(ValueError, match="schema"):
        replace(training, schema_version="training_run_v0")
    with pytest.raises(ValueError, match="digest"):
        replace(training, digest="f" * 64)

    walk_forward = _walk_forward_manifest(tmp_path / "walk-forward")
    with pytest.raises(ValueError, match="fold_count"):
        replace(walk_forward, fold_count=True)
    with pytest.raises(ValueError, match="fold_count"):
        replace(walk_forward, fold_count=0)
    with pytest.raises(ValueError, match="schema"):
        replace(walk_forward, schema_version="walk_forward_run_v0")
    with pytest.raises(ValueError, match="digest"):
        replace(walk_forward, digest="0" * 64)


def test_build_files_rejects_missing_and_symlink_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_manifest._build_files(tmp_path, ("missing.bin",))

    target = _write(tmp_path, "target.bin")
    link = tmp_path / "link.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="symlink"):
        run_manifest._build_files(tmp_path, ("link.bin",))


def test_run_manifest_parsers_reject_malformed_payloads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        run_manifest._mapping([], field="value")
    with pytest.raises(ValueError, match="string"):
        run_manifest._string(1, field="value")
    with pytest.raises(ValueError, match="integer"):
        run_manifest._integer(True, field="value")
    with pytest.raises(ValueError, match="list"):
        run_manifest._parse_files({"files": {}})
    with pytest.raises(FileNotFoundError, match="missing"):
        run_manifest._read_payload(tmp_path / "missing")

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "run.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        run_manifest.load_training_run_manifest(malformed)

    wrong_schema = tmp_path / "wrong-schema"
    wrong_schema.mkdir()
    (wrong_schema / "run.json").write_text(
        json.dumps({"schema_version": "unknown"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="training run schema"):
        run_manifest.load_training_run_manifest(wrong_schema)
    with pytest.raises(ValueError, match="walk-forward run schema"):
        run_manifest.load_walk_forward_run_manifest(wrong_schema)


def test_run_directory_detects_missing_size_and_digest_tampering(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    missing_manifest = _training_manifest(missing_root)
    run_manifest.write_training_run_manifest(missing_root, missing_manifest)
    (missing_root / "artifact.bin").unlink()
    with pytest.raises(ValueError, match="missing"):
        run_manifest.validate_training_run_directory(missing_root)

    size_root = tmp_path / "size"
    size_manifest = _training_manifest(size_root)
    run_manifest.write_training_run_manifest(size_root, size_manifest)
    (size_root / "artifact.bin").write_bytes(b"longer")
    with pytest.raises(ValueError, match="size"):
        run_manifest.validate_training_run_directory(size_root)

    digest_root = tmp_path / "digest"
    digest_manifest = _training_manifest(digest_root)
    run_manifest.write_training_run_manifest(digest_root, digest_manifest)
    (digest_root / "artifact.bin").write_bytes(b"y")
    with pytest.raises(ValueError, match="digest"):
        run_manifest.validate_training_run_directory(digest_root)
