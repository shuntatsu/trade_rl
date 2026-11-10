from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import trade_rl.artifacts.atomic_write as atomic_write


def test_exclusive_file_rename_preserves_existing_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "staging.json"
    target = tmp_path / "plan.json"
    source.write_bytes(b"writer")
    target.write_bytes(b"competitor")

    with pytest.raises(FileExistsError):
        atomic_write.atomic_rename_file(source, target)

    assert source.read_bytes() == b"writer"
    assert target.read_bytes() == b"competitor"

    target.unlink()
    assert atomic_write.atomic_rename_file(source, target) == target
    assert target.read_bytes() == b"writer"
    assert not source.exists()


def test_exclusive_file_rename_fails_closed_when_native_rename_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "staging.json"
    target = tmp_path / "plan.json"
    source.write_bytes(b"writer")
    monkeypatch.setattr(atomic_write, "_IS_WINDOWS", False)
    monkeypatch.setattr(
        atomic_write, "sys", SimpleNamespace(platform="unsupported-platform")
    )

    with pytest.raises(OSError, match="unsupported"):
        atomic_write.atomic_rename_file(source, target)

    assert source.read_bytes() == b"writer"
    assert not target.exists()


def test_exclusive_file_rename_bounds_windows_transient_lock_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "staging.json"
    target = tmp_path / "plan.json"
    source.write_bytes(b"writer")
    monkeypatch.setattr(atomic_write, "_IS_WINDOWS", True)
    original_rename = Path.rename
    attempts = 0
    delays: list[float] = []

    def rename_after_one_lock(staging: Path, destination: str | Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporary Windows file lock")
        return original_rename(staging, destination)

    monkeypatch.setattr(Path, "rename", rename_after_one_lock)
    monkeypatch.setattr(atomic_write.time, "sleep", delays.append)

    assert atomic_write.atomic_rename_file(source, target) == target
    assert target.read_bytes() == b"writer"
    assert not source.exists()
    assert attempts == 2
    assert delays == list(atomic_write._DIRECTORY_RENAME_RETRY_DELAYS[:1])
