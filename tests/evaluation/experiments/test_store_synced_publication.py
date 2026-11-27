"""Explicit file/dir fsync order and nonretryable post-publication failure."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments import store as store_module
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    InvalidExperimentStateError,
    PublishedArtifactCleanupError,
)
from trade_rl.evaluation.experiments.store import StudyStore

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc/self/fd").is_dir(),
    reason="requires Linux descriptor-relative fsync publication",
)


def _root(tmp_path: Path) -> StudyStore:
    root = tmp_path / "study"
    root.mkdir()
    return StudyStore(root)


def _builder(stage: Path) -> None:
    (stage / "nested").mkdir()
    (stage / "nested" / "a.bin").write_bytes(b"a\x00")
    (stage / "receipt.json").write_bytes(b'{"blocked":true}')


def test_explicit_sync_flushes_pinned_tree_then_committed_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _root(tmp_path)
    events = []
    sync = os.fsync
    rename = store_module.atomic_rename_directory

    def observed_sync(fd: int) -> None:
        name = Path(os.readlink(f"/proc/self/fd/{fd}")).name
        events.append(("sync", name))
        if name == "study":
            with pytest.raises(ArtifactIntegrityError, match="publication"):
                store.read_json("attempt/receipt.json")
        sync(fd)

    def observed_rename(source: str | Path, target: str | Path) -> Path:
        events.append(("rename", Path(target).name))
        return rename(source, target)

    monkeypatch.setattr(store_module.os, "fsync", observed_sync)
    monkeypatch.setattr(store_module, "atomic_rename_directory", observed_rename)
    published = store.publish_directory_once(
        "attempt", _builder, sync_before_return=True
    )
    assert published == store.root / "attempt"
    assert (published / "receipt.json").read_bytes() == b'{"blocked":true}'
    assert (published / "nested" / "a.bin").read_bytes() == b"a\x00"
    marker = events.index(("rename", "attempt"))
    stage = [
        event
        for event in events
        if event[0] == "sync" and ".attempt.staging-" in event[1]
    ]
    assert len(stage) == 1
    assert (
        events.index(("sync", "a.bin"))
        < events.index(("sync", "nested"))
        < events.index(stage[0])
        < marker
    )
    assert events.index(("sync", "receipt.json")) < events.index(stage[0])
    assert events[marker + 1 :] == [("sync", "study")]
    with pytest.raises(InvalidExperimentStateError, match="already exists"):
        store.publish_directory_once("attempt", _builder, sync_before_return=True)


@pytest.mark.parametrize("where", ["a.bin", "nested", "staging"])
def test_sync_failure_before_rename_leaves_no_final_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    store = _root(tmp_path)
    sync = os.fsync
    seen = []

    def fail(fd: int) -> None:
        name = Path(os.readlink(f"/proc/self/fd/{fd}")).name
        if name == where or (where == "staging" and ".attempt.staging-" in name):
            seen.append(name)
            raise OSError(errno.EIO, "injected prepublication fsync")
        sync(fd)

    def forbidden(*args, **kwargs):
        pytest.fail("failed fsync reached publication rename")

    monkeypatch.setattr(store_module.os, "fsync", fail)
    monkeypatch.setattr(store_module, "atomic_rename_directory", forbidden)
    with pytest.raises(OSError, match="injected prepublication fsync"):
        store.publish_directory_once("attempt", _builder, sync_before_return=True)
    assert len(seen) == 1
    assert not (store.root / "attempt").exists()
    assert not list(store.root.glob(".attempt.staging-*"))


def test_post_rename_sync_failure_keeps_committed_bytes_and_forbids_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _root(tmp_path)
    sync = os.fsync
    target = store.root / "attempt"
    seen = []

    def fail(fd: int) -> None:
        if Path(os.readlink(f"/proc/self/fd/{fd}")) == store.root:
            assert target.is_dir()
            seen.append("committed")
            raise OSError(errno.EIO, "injected committed parent fsync")
        sync(fd)

    monkeypatch.setattr(store_module.os, "fsync", fail)
    with pytest.raises(PublishedArtifactCleanupError, match="do not retry") as error:
        store.publish_directory_once("attempt", _builder, sync_before_return=True)
    assert isinstance(error.value.__cause__, OSError)
    assert seen == ["committed"]
    assert (target / "receipt.json").read_bytes() == b'{"blocked":true}'
    assert not list(store.root.glob(".attempt.staging-*"))
    with pytest.raises(InvalidExperimentStateError, match="already exists"):
        store.publish_directory_once("attempt", _builder, sync_before_return=True)


def test_default_does_not_add_syncs_and_scope_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _root(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected sync or builder")

    monkeypatch.setattr(store_module.os, "fsync", forbidden)
    store.publish_directory_once("legacy/nested", _builder)
    assert (store.root / "legacy/nested/receipt.json").is_file()
    for value in ("yes", 1, None):
        with pytest.raises(ValueError, match="boolean"):
            store.publish_directory_once("bad", forbidden, sync_before_return=value)
    with pytest.raises(ValueError, match="direct child"):
        store.publish_directory_once("bad/nested", forbidden, sync_before_return=True)
    assert not (store.root / "bad").exists()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute root"):
        StudyStore("study").publish_directory_once(
            "bad", forbidden, sync_before_return=True
        )
    monkeypatch.setattr(store_module.sys, "platform", "win32")
    with pytest.raises(ValueError, match="Linux"):
        store.publish_directory_once("unsupported", forbidden, sync_before_return=True)
    assert not (store.root / "unsupported").exists()


def test_sync_does_not_waive_inventory_race_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _root(tmp_path)
    rename = store_module.atomic_rename_directory

    def altered(source: str | Path, target: str | Path) -> Path:
        (Path(source) / "receipt.json").write_bytes(b"forged")
        return rename(source, target)

    monkeypatch.setattr(store_module, "atomic_rename_directory", altered)
    with pytest.raises(ArtifactIntegrityError, match="contents changed"):
        store.publish_directory_once("attempt", _builder, sync_before_return=True)
    assert not (store.root / "attempt").exists()
    (rejected,) = store.root.glob(".attempt.rejected-*")
    assert (rejected / "receipt.json").read_bytes() == b"forged"


def test_transient_nested_builder_root_is_not_a_synced_publication_root(
    tmp_path: Path,
) -> None:
    store = _root(tmp_path)

    def outer_builder(stage: Path) -> None:
        inner = StudyStore(stage)
        with pytest.raises(ValueError, match="stable unlinked root"):
            inner.publish_directory_once("inner", _builder, sync_before_return=True)
        assert not (stage / "inner").exists()

    store.publish_directory_once("outer", outer_builder)


@pytest.mark.parametrize("phase", ["before_rename", "after_rename"])
def test_process_exit_at_rename_boundary_is_not_success(
    tmp_path: Path, phase: str
) -> None:
    """Actual abrupt process exit, not a power-loss/filesystem durability oracle."""
    store = _root(tmp_path)
    script = """
import os
import sys
from pathlib import Path
from trade_rl.evaluation.experiments import store as store_module
from trade_rl.evaluation.experiments.store import StudyStore

root, phase = Path(sys.argv[1]), sys.argv[2]
actual = store_module.atomic_rename_directory

def exit_at_boundary(source, target):
    if phase == "after_rename":
        actual(source, target)
    os._exit(83)

store_module.atomic_rename_directory = exit_at_boundary
StudyStore(root).publish_directory_once(
    "attempt", lambda stage: (stage / "receipt.json").write_bytes(b'{"blocked":true}'),
    sync_before_return=True,
)
raise AssertionError("interrupted publication returned success")
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(store.root), phase],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 83, child.stderr
    target = store.root / "attempt"
    if phase == "before_rename":
        assert not target.exists()
        (stage,) = store.root.glob(".attempt.staging-*")
        assert (stage / "receipt.json").read_bytes() == b'{"blocked":true}'
        # The abandoned staging directory is not a completed receipt.
    else:
        assert (target / "receipt.json").read_bytes() == b'{"blocked":true}'
        assert not list(store.root.glob(".attempt.staging-*"))
        with pytest.raises(InvalidExperimentStateError, match="already exists"):
            store.publish_directory_once("attempt", _builder, sync_before_return=True)
