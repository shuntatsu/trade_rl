"""Synthetic regressions for file-stage cleanup after a source-name substitution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments import store as store_module
from trade_rl.evaluation.experiments.errors import (
    InvalidExperimentStateError,
    PublishedArtifactCleanupError,
)
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires proc file descriptors for deterministic file-stage swap",
)
@pytest.mark.parametrize("conflicting_target", (False, True))
def test_json_stage_replacement_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conflicting_target: bool
) -> None:
    store = StudyStore(tmp_path / "study")
    detached = tmp_path / "verified-staging-detached"
    source_name: Path | None = None
    real_fsync = os.fsync

    def replace_after_verified_fsync(fd: int) -> None:
        nonlocal source_name
        real_fsync(fd)
        if source_name is not None:
            return
        stage = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if not stage.name.startswith(".synthetic.json.staging-"):
            return
        stage.rename(detached)
        stage.write_bytes(b"UNRELATED_REPLACEMENT")
        source_name = stage
        if conflicting_target:
            target = store.root / "evidence" / "synthetic.json"
            target.write_bytes(b"EXISTING_TARGET")

    monkeypatch.setattr(store_module.os, "fsync", replace_after_verified_fsync)
    if conflicting_target:
        with pytest.raises(InvalidExperimentStateError, match="already exists"):
            store.publish_json_once("evidence/synthetic.json", {"validated": True})
    else:
        with pytest.raises(PublishedArtifactCleanupError, match="cleanup failed"):
            store.publish_json_once("evidence/synthetic.json", {"validated": True})
        assert store.read_json("evidence/synthetic.json") == {"validated": True}
        assert (store.root / "evidence" / "synthetic.json").stat().st_ino == (
            detached.stat().st_ino
        ), "the published JSON must be the original fsynced file inode"

    assert source_name is not None
    assert detached.read_bytes() == b'{"validated":true}'
    assert source_name.read_bytes() == b"UNRELATED_REPLACEMENT", (
        "JSON cleanup deleted a file it did not create"
    )
    if conflicting_target:
        assert (store.root / "evidence" / "synthetic.json").read_bytes() == (
            b"EXISTING_TARGET"
        )


def test_json_staging_name_collision_does_not_delete_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StudyStore(tmp_path / "study")
    (store.root / "evidence").mkdir()
    collision = store.root / "evidence" / ".synthetic.json.staging-fixed"
    collision.write_bytes(b"PREEXISTING")
    monkeypatch.setattr(store_module, "_temporary_name_suffix", lambda: "fixed")

    with pytest.raises(FileExistsError):
        store.publish_json_once("evidence/synthetic.json", {"validated": True})

    assert collision.read_bytes() == b"PREEXISTING", (
        "failed exclusive open must not clean up a file it never created"
    )
    assert not (store.root / "evidence" / "synthetic.json").exists()


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires Linux source-FD-backed publication",
)
def test_post_link_file_close_failure_is_nonretryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error closing an already-linked stage must not invite a retry."""
    store = StudyStore(tmp_path / "study")
    actual_open = Path.open

    class FailingClose:
        def __init__(self, handle: object) -> None:
            self.handle = handle

        def __enter__(self) -> object:
            return self.handle.__enter__()  # type: ignore[attr-defined]

        def __exit__(self, *args: object) -> None:
            self.handle.__exit__(*args)  # type: ignore[attr-defined]
            raise OSError("injected post-publication close failure")

    def open_stage(path: Path, *args: object, **kwargs: object) -> object:
        result = actual_open(path, *args, **kwargs)
        if path.name.startswith(".synthetic.json.staging-"):
            return FailingClose(result)
        return result

    monkeypatch.setattr(Path, "open", open_stage)
    with pytest.raises(PublishedArtifactCleanupError, match="do not retry"):
        store.publish_json_once("evidence/synthetic.json", {"validated": True})

    assert store.read_json("evidence/synthetic.json") == {"validated": True}


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires Linux source-FD-backed publication",
)
def test_post_link_parent_fd_close_failure_is_nonretryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closing failure after linkat succeeds must not report a retryable write."""
    store = StudyStore(tmp_path / "study")
    actual_link = os.link
    actual_close = os.close
    linked = False
    injected = False

    def observe_link(*args: object, **kwargs: object) -> None:
        nonlocal linked
        actual_link(*args, **kwargs)
        linked = True

    def fail_close_after_link(fd: int) -> None:
        nonlocal injected
        actual_close(fd)
        if linked and not injected:
            injected = True
            raise OSError("injected post-link parent FD close failure")

    monkeypatch.setattr(store_module.os, "link", observe_link)
    monkeypatch.setattr(store_module.os, "close", fail_close_after_link)
    with pytest.raises(PublishedArtifactCleanupError, match="do not retry"):
        store.publish_json_once("evidence/synthetic.json", {"validated": True})

    assert linked and injected
    assert store.read_json("evidence/synthetic.json") == {"validated": True}
