from __future__ import annotations

import hashlib
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

import trade_rl.workflows.release_packaging as release_packaging
from tests.serving.conftest import _rebuild_v3_training_run
from tests.serving.test_package import (
    PUBLIC_KEY,
    _confirmation,
    _training_run,
)
from trade_rl.artifacts.run_manifest import RunFile, TrainingRunManifest
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.workflows.release_packaging import package_selected_training_run


def _complete_training_run(
    root: Path,
    **kwargs: Any,
) -> TrainingRunManifest:
    return _rebuild_v3_training_run(_training_run, root, **kwargs)


def _package(
    tmp_path: Path,
    training_root: Path,
    training: TrainingRunManifest,
) -> None:
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)
    package_selected_training_run(
        training_root=training_root,
        confirmation_path=confirmation_path,
        output_root=tmp_path / "bundle",
        signal_digest="a" * 64,
        selection_digest="b" * 64,
        trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
        trusted_now=training.completed_at + timedelta(days=30),
    )


def _run_file(path: str, content: bytes, *, size_delta: int = 0) -> RunFile:
    return RunFile(
        path=path,
        digest=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content) + size_delta,
    )


def test_sequence_publication_requires_structured_policy_loader(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "training"
    training = _complete_training_run(
        training_root,
        run_kind="research_selected_final",
        observation_schema=SEQUENCE_OBSERVATION_SCHEMA,
        architecture_digest="9" * 64,
        include_structured_loader=False,
    )

    with pytest.raises(ValueError, match="structured policy loader"):
        _package(tmp_path, training_root, training)

    assert not (tmp_path / "bundle").exists()


def test_sequence_publication_rejects_loader_action_size_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_root = tmp_path / "training"
    training = _complete_training_run(
        training_root,
        run_kind="research_selected_final",
        observation_schema=SEQUENCE_OBSERVATION_SCHEMA,
        architecture_digest="9" * 64,
        include_structured_loader=True,
    )
    monkeypatch.setattr(
        "trade_rl.serving.policy_loader.load_structured_policy_loader_manifest",
        lambda _: {"architecture_digest": "9" * 64, "action_size": 999},
    )

    with pytest.raises(ValueError, match="action size"):
        _package(tmp_path, training_root, training)

    assert not (tmp_path / "bundle").exists()


def test_publication_rejects_artifact_changed_after_manifest_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_root = tmp_path / "training"
    training = _complete_training_run(
        training_root,
        run_kind="research_selected_final",
    )
    original_validate = release_packaging.validate_training_run_directory

    def validate_then_mutate(path: Path) -> TrainingRunManifest:
        manifest = original_validate(path)
        (path / "policy.zip").write_bytes(b"changed-after-validation")
        return manifest

    monkeypatch.setattr(
        release_packaging,
        "validate_training_run_directory",
        validate_then_mutate,
    )

    with pytest.raises(ValueError, match="source artifact identity changed"):
        _package(tmp_path, training_root, training)

    assert not (tmp_path / "bundle").exists()


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges")
def test_verified_copy_rejects_symlink_source(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training_root.mkdir()
    outside = tmp_path / "outside.bin"
    content = b"outside"
    outside.write_bytes(content)
    (training_root / "artifact.bin").symlink_to(outside)

    with pytest.raises(ValueError, match="source artifact identity changed"):
        release_packaging._copy_verified_run_file(
            training_root=training_root,
            stage=tmp_path / "stage",
            item=_run_file("artifact.bin", content),
        )


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges")
def test_verified_copy_rejects_parent_symlink_escape(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    content = b"outside"
    (outside / "artifact.bin").write_bytes(content)
    (training_root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes training root"):
        release_packaging._copy_verified_run_file(
            training_root=training_root,
            stage=tmp_path / "stage",
            item=_run_file("linked/artifact.bin", content),
        )


def test_verified_copy_rejects_missing_source(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training_root.mkdir()

    with pytest.raises(ValueError, match="source artifact identity changed"):
        release_packaging._copy_verified_run_file(
            training_root=training_root,
            stage=tmp_path / "stage",
            item=_run_file("missing.bin", b"missing"),
        )


def test_verified_copy_rejects_source_size_mismatch(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    training_root.mkdir()
    content = b"artifact"
    (training_root / "artifact.bin").write_bytes(content)

    with pytest.raises(ValueError, match="source artifact identity changed"):
        release_packaging._copy_verified_run_file(
            training_root=training_root,
            stage=tmp_path / "stage",
            item=_run_file("artifact.bin", content, size_delta=1),
        )


def test_verified_copy_rejects_destination_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_root = tmp_path / "training"
    training_root.mkdir()
    content = b"artifact"
    (training_root / "artifact.bin").write_bytes(content)
    monkeypatch.setattr(release_packaging, "_file_digest", lambda _: "f" * 64)

    with pytest.raises(ValueError, match="destination artifact identity mismatch"):
        release_packaging._copy_verified_run_file(
            training_root=training_root,
            stage=tmp_path / "stage",
            item=_run_file("artifact.bin", content),
        )

    assert not (tmp_path / "stage" / "artifact.bin").exists()
    assert not (tmp_path / "stage" / ".artifact.bin.tmp").exists()
