"""Deterministic read-side path-swap regression for Study JSON artifacts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import pytest

from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink privileges")
def test_study_read_refuses_symlink_inserted_after_regular_file_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No later name lookup may follow a path swapped after a file check."""
    store = StudyStore(tmp_path / "study")
    target = store.publish_json_once("plan.json", {"trusted": True})
    outside = tmp_path / "outside.json"
    outside.write_text('{"trusted": false}\n', encoding="utf-8")
    original_is_file = Path.is_file
    swapped = False

    def swap_after_check(path: Path) -> bool:
        nonlocal swapped
        checked = original_is_file(path)
        if path == target and not swapped:
            assert checked
            target.unlink()
            target.symlink_to(outside)
            swapped = True
        return checked

    monkeypatch.setattr(Path, "is_file", swap_after_check)
    with pytest.raises(ArtifactIntegrityError):
        store.read_json("plan.json")


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink privileges")
def test_study_read_refuses_existing_symlink(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")
    outside = tmp_path / "outside.json"
    outside.write_text('{"trusted": false}\n', encoding="utf-8")
    (store.root / "plan.json").symlink_to(outside)

    with pytest.raises(ArtifactIntegrityError, match="regular file"):
        store.read_json("plan.json")


@pytest.mark.skipif(os.name != "posix", reason="rename race is POSIX specific")
def test_study_read_refuses_late_regular_file_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncooperative writer must not substitute a second valid JSON inode."""
    store = StudyStore(tmp_path / "study")
    target = store.publish_json_once("plan.json", {"trusted": True})
    original = target.read_bytes()
    foreign = tmp_path / "foreign.json"
    foreign.write_text('{"trusted":false}', encoding="utf-8")
    foreign_bytes = foreign.read_bytes()
    detached = tmp_path / "detached.json"
    real_is_file = Path.is_file
    swapped = False

    def swap_after_check(path: Path) -> bool:
        nonlocal swapped
        checked = real_is_file(path)
        if path == target and checked and not swapped:
            target.rename(detached)
            foreign.rename(target)
            swapped = True
        return checked

    monkeypatch.setattr(Path, "is_file", swap_after_check)
    with pytest.raises(ArtifactIntegrityError, match="changed during read"):
        store.read_json("plan.json")

    assert swapped
    assert detached.read_bytes() == original
    assert target.read_bytes() == foreign_bytes


@pytest.mark.skipif(os.name != "posix", reason="read/write race is POSIX specific")
def test_study_read_rejects_in_place_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-cooperating writer cannot alter a published Study read in flight."""
    from trade_rl.evaluation.experiments import store as store_module

    store = StudyStore(tmp_path / "study")
    target = store.publish_json_once("plan.json", {"trusted": True})
    real_open = store_module.open_regular_binary
    raced = False

    class MutatingRead:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            nonlocal raced
            raw = self.handle.read(size)
            if not raced:
                target.write_text('{"trusted":false}', encoding="utf-8")
                raced = True
            return raw

    @contextmanager
    def changing_open(path: Path, *, field: str) -> Iterator[MutatingRead]:
        with real_open(path, field=field) as handle:
            yield MutatingRead(handle)

    monkeypatch.setattr(store_module, "open_regular_binary", changing_open)
    with pytest.raises(ArtifactIntegrityError, match="changed during read"):
        store.read_json("plan.json")

    assert raced
    assert target.read_bytes() == b'{"trusted":false}'


@pytest.mark.skipif(os.name != "posix", reason="unlink race is POSIX specific")
def test_study_read_maps_unlinked_leaf_during_read_to_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late unlink is an integrity failure, not a malformed JSON verdict."""
    from trade_rl.evaluation.experiments import store as store_module

    store = StudyStore(tmp_path / "study")
    target = store.publish_json_once("plan.json", {"trusted": True})
    real_open = store_module.open_regular_binary
    removed = False

    class UnlinkDuringRead:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def fileno(self) -> int:
            return self.handle.fileno()

        def read(self, size: int = -1) -> bytes:
            nonlocal removed
            raw = self.handle.read(size)
            if not removed:
                target.unlink()
                removed = True
            return raw

    @contextmanager
    def changing_open(path: Path, *, field: str) -> Iterator[UnlinkDuringRead]:
        with real_open(path, field=field) as handle:
            yield UnlinkDuringRead(handle)

    monkeypatch.setattr(store_module, "open_regular_binary", changing_open)
    with pytest.raises(
        ArtifactIntegrityError, match="Study artifact changed during read"
    ):
        store.read_json("plan.json")

    assert removed
    assert not target.exists()
