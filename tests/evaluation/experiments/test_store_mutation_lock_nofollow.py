"""Deterministic synthetic Study mutation-lock path substitution regressions."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from trade_rl.artifacts.verified_file import _trusted_process_directory_fd
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore

_POSIX_NOFOLLOW = os.name == "posix" and hasattr(os, "O_NOFOLLOW")


@pytest.mark.skipif(not _POSIX_NOFOLLOW, reason="requires POSIX O_NOFOLLOW")
def test_mutation_lock_refuses_symlink_inserted_after_leaf_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StudyStore(tmp_path / "study")
    lock_path = store.root / ".mutation.lock"
    outside = tmp_path / "outside.json"
    original = b"outside must stay unchanged"
    outside.write_bytes(original)
    real_is_symlink = Path.is_symlink
    swapped = False

    def swap_after_check(path: Path) -> bool:
        nonlocal swapped
        checked = real_is_symlink(path)
        if path == lock_path and not swapped:
            assert not checked
            lock_path.symlink_to(outside)
            swapped = True
        return checked

    monkeypatch.setattr(Path, "is_symlink", swap_after_check)
    with pytest.raises(ArtifactIntegrityError, match="Study mutation lock"):
        with store.mutation_lock():
            pytest.fail("a symlink substituted after the check acquired the lock")

    assert swapped
    assert outside.read_bytes() == original


@pytest.mark.skipif(not _POSIX_NOFOLLOW, reason="requires POSIX O_NOFOLLOW")
def test_mutation_lock_refuses_root_swapped_after_directory_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "study"
    store = StudyStore(root)
    old_root = tmp_path / "previous-study"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_is_dir = Path.is_dir
    swapped = False

    def swap_after_check(path: Path) -> bool:
        nonlocal swapped
        checked = real_is_dir(path)
        if path == root and not swapped:
            assert checked
            root.rename(old_root)
            root.symlink_to(outside, target_is_directory=True)
            swapped = True
        return checked

    monkeypatch.setattr(Path, "is_dir", swap_after_check)
    with pytest.raises(ArtifactIntegrityError, match="Study mutation lock"):
        with store.mutation_lock():
            pytest.fail("a substituted Study root redirected the lock")

    assert swapped
    assert not (outside / ".mutation.lock").exists()


@pytest.mark.skipif(
    os.name != "posix"
    or not getattr(os, "O_NOFOLLOW", 0)
    or not getattr(os, "O_DIRECTORY", 0)
    or os.open not in os.supports_dir_fd
    or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX no-follow process directory FDs",
)
def test_mutation_lock_accepts_registered_stage_fd_alias(tmp_path: Path) -> None:
    """The owning thread may lock a legitimate FD-pinned nested Study."""
    study = tmp_path / "study"
    study.mkdir()
    descriptor = os.open(study, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        alias = Path("/proc/self/fd") / str(descriptor)
        assert alias.is_symlink() and alias.is_dir()
        with _trusted_process_directory_fd(descriptor):
            nested = StudyStore(alias)
            with nested.mutation_lock():
                assert (alias / ".mutation.lock").is_file()
        assert (study / ".mutation.lock").is_file()
    finally:
        os.close(descriptor)


@pytest.mark.skipif(
    os.name != "posix"
    or not getattr(os, "O_NOFOLLOW", 0)
    or not getattr(os, "O_DIRECTORY", 0)
    or os.open not in os.supports_dir_fd
    or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX no-follow process directory FDs",
)
def test_mutation_lock_refuses_unregistered_process_fd_alias(tmp_path: Path) -> None:
    """An arbitrary process FD alias must not create an outside lock file."""
    store = StudyStore(tmp_path / "study")
    outside = tmp_path / "outside"
    outside.mkdir()
    descriptor = os.open(outside, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        alias = Path("/proc/self/fd") / str(descriptor)
        assert alias.is_symlink() and alias.is_dir()
        store.root = alias
        with pytest.raises(ArtifactIntegrityError, match="Study root"):
            with store.mutation_lock():
                pytest.fail("untrusted process FD alias acquired the Study lock")
        assert not (outside / ".mutation.lock").exists()
    finally:
        os.close(descriptor)


def test_mutation_lock_refuses_nonregular_lock_file(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")
    (store.root / ".mutation.lock").mkdir()

    with pytest.raises(ArtifactIntegrityError, match="Study mutation lock"):
        with store.mutation_lock():
            pytest.fail("a directory was accepted as a mutation lock")


def test_mutation_lock_still_creates_regular_reentrant_lock(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")
    lock_path = store.root / ".mutation.lock"

    with store.mutation_lock():
        assert lock_path.exists()
        assert stat.S_ISREG(lock_path.stat().st_mode)
        with store.mutation_lock():
            assert lock_path.is_file()

    with store.mutation_lock():
        assert lock_path.is_file()
