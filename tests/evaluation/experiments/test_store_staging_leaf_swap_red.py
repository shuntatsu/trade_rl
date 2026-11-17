"""Test-only RED: replacing the staging leaf must not redirect builder writes.

The symlink, staging directory, and outside marker live entirely in pytest's
disposable tmp_path. Historical Study and market artifacts are never accessed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments import store as store_module
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    InvalidExperimentStateError,
    PublishedArtifactCleanupError,
)
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(
    os.name != "posix"
    or not getattr(os, "O_NOFOLLOW", 0)
    or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX directory-fd publication",
)
def test_directory_builder_refuses_replaced_staging_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attacker-writable staging name is not a pinned directory capability."""
    store = StudyStore(tmp_path / "study")
    outside = tmp_path / "outside-stage-leaf"
    outside.mkdir()
    detached_stage = tmp_path / "original-stage-leaf"
    real_write_text = Path.write_text
    injected = False

    def replace_leaf_before_builder_write(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        nonlocal injected
        if path.name == "stage-leaf-marker.txt" and not injected:
            stages = list(store.root.glob(".synthetic.staging-*"))
            assert len(stages) == 1
            original_stage = stages[0]
            assert original_stage.is_dir() and not original_stage.is_symlink()
            original_stage.rename(detached_stage)
            original_stage.symlink_to(outside, target_is_directory=True)
            injected = True
        return real_write_text(
            path, data, encoding=encoding, errors=errors, newline=newline
        )

    monkeypatch.setattr(Path, "write_text", replace_leaf_before_builder_write)

    def build(stage: Path) -> None:
        (stage / "stage-leaf-marker.txt").write_text("synthetic", encoding="utf-8")

    with pytest.raises((ArtifactIntegrityError, ValueError, OSError)):
        store.publish_directory_once("evidence/synthetic", build)

    assert injected, "the concurrent stage-leaf substitution was not exercised"
    assert not (outside / "stage-leaf-marker.txt").exists(), (
        "builder wrote outside Study through a replaced staging leaf"
    )
    assert not (store.root / "evidence" / "synthetic").exists()


@pytest.mark.skipif(
    os.name != "posix"
    or not getattr(os, "O_NOFOLLOW", 0)
    or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX directory-fd publication",
)
def test_directory_publish_rejects_stage_swap_after_inode_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checked staging inode must remain the inode atomically published."""
    store = StudyStore(tmp_path / "study")
    detached = tmp_path / "validated-stage-detached"
    real_rename = store_module.atomic_rename_directory
    injected = False

    def replace_stage_at_rename(source: str | Path, target: str | Path) -> Path:
        nonlocal injected
        stage = Path(source)
        if not injected:
            assert stage.name.startswith(".synthetic.staging-")
            original_name = store.root / stage.name
            assert original_name.is_dir()
            assert not original_name.is_symlink()
            genuine_path = original_name / "genuine.txt"
            assert genuine_path.read_text(encoding="utf-8") == "genuine"
            original_name.rename(detached)
            original_name.mkdir()
            forged_path = original_name / "forged.txt"
            forged_path.write_text("forged", encoding="utf-8")
            injected = True
        return real_rename(source, target)

    monkeypatch.setattr(
        store_module, "atomic_rename_directory", replace_stage_at_rename
    )

    def build(stage: Path) -> None:
        good_path = stage / "genuine.txt"
        good_path.write_text("genuine", encoding="utf-8")

    permitted_failures = (
        ArtifactIntegrityError,
        InvalidExperimentStateError,
        PublishedArtifactCleanupError,
        ValueError,
        OSError,
    )
    with pytest.raises(permitted_failures):
        store.publish_directory_once("evidence/synthetic", build)

    assert injected, "the post-inode-check substitution was not exercised"
    assert (detached / "genuine.txt").read_text(encoding="utf-8") == "genuine"
    published = store.root / "evidence" / "synthetic"
    assert not published.exists(), "an unvalidated staging inode was published"
    assert not (published / "forged.txt").exists()
