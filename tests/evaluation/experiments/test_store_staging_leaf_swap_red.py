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


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX directory-fd publication",
)
def test_directory_swap_with_denied_rollback_reports_unsafe_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed rollback must never pretend the unverified artifact is safe."""
    store = StudyStore(tmp_path / "study")
    detached = tmp_path / "verified-detached"
    real_rename = store_module.atomic_rename_directory
    calls = 0

    def swap_then_block(source: str | Path, target: str | Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            stage_name = store.root / Path(source).name
            stage_name.rename(detached)
            stage_name.mkdir()
            (stage_name / "forged.txt").write_text("NOT VALIDATED", encoding="utf-8")
            return real_rename(source, target)
        raise PermissionError("synthetic rollback refused")

    monkeypatch.setattr(store_module, "atomic_rename_directory", swap_then_block)

    def build(stage: Path) -> None:
        (stage / "genuine.txt").write_text("GENUINE", encoding="utf-8")

    with pytest.raises(PublishedArtifactCleanupError, match="do not retry"):
        store.publish_directory_once("evidence/synthetic", build)

    assert calls == 2
    assert (detached / "genuine.txt").read_text(encoding="utf-8") == "GENUINE"
    # The exception signals that manual quarantine is still required.
    assert (store.root / "evidence" / "synthetic" / "forged.txt").read_text(
        encoding="utf-8"
    ) == "NOT VALIDATED"


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires Linux process file descriptors for deterministic injection",
)
def test_json_publish_never_links_replaced_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After staged fsync, publish the checked file inode or refuse publication.

    Both the replacement and the validated original stay under disposable tmp_path.
    The injection is at the write/fsync boundary, independent of the final link
    implementation; the expected result forbids publishing substituted bytes.
    """
    store = StudyStore(tmp_path / "study")
    detached = tmp_path / "validated-json-detached"
    real_fsync = os.fsync
    swapped = False

    def swap_after_staging_fsync(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        if swapped:
            return
        try:
            stage = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            return
        if not stage.name.startswith(".synthetic.json.staging-"):
            return
        assert stage.read_bytes() == b'{"validated":true}'
        stage.rename(detached)
        stage.write_bytes(b'{"forged":true}')
        swapped = True

    monkeypatch.setattr(store_module.os, "fsync", swap_after_staging_fsync)
    target = store.root / "evidence" / "synthetic.json"
    # Source-name replacement after fsync must preserve foreign files and
    # never classify a committed link as a safe-to-retry failed write.
    with pytest.raises(PublishedArtifactCleanupError, match="do not retry"):
        store.publish_json_once("evidence/synthetic.json", {"validated": True})
    assert swapped, "the staging source-name swap was not exercised"
    assert store.read_json("evidence/synthetic.json") == {"validated": True}
    assert target.stat().st_ino == detached.stat().st_ino
    replacements = list((store.root / "evidence").glob(".synthetic.json.staging-*"))
    assert len(replacements) == 1
    assert replacements[0].read_bytes() == b'{"forged":true}'
