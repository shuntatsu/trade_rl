from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

import trade_rl.workflows.release_packaging as release_packaging
from tests.serving.test_package import (
    PUBLIC_KEY,
    _confirmation,
    _training_run,
)
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.workflows.release_packaging import package_selected_training_run


def _package(tmp_path: Path, training_root: Path, training: object) -> None:
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)  # type: ignore[arg-type]
    package_selected_training_run(
        training_root=training_root,
        confirmation_path=confirmation_path,
        output_root=tmp_path / "bundle",
        signal_digest="a" * 64,
        selection_digest="b" * 64,
        trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
        trusted_now=training.completed_at + timedelta(days=30),  # type: ignore[attr-defined]
    )


def test_sequence_publication_requires_structured_policy_loader(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "training"
    training = _training_run(
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
    training = _training_run(
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
    training = _training_run(training_root, run_kind="research_selected_final")
    original_validate = release_packaging.validate_training_run_directory

    def validate_then_mutate(path: Path):
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
