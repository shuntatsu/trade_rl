from __future__ import annotations

import errno
from pathlib import Path
from types import SimpleNamespace

import pytest

import trade_rl.artifacts.atomic_write as atomic_write


def test_atomic_directory_publish_retries_transient_windows_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "published"
    staging.mkdir()
    (staging / "receipt.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(atomic_write, "_IS_WINDOWS", True)

    original_rename = type(staging).rename
    attempts = 0
    delays: list[float] = []

    def rename_with_transient_lock(source: Path, destination: str | Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporary Windows directory lock")
        return original_rename(source, destination)

    monkeypatch.setattr(type(staging), "rename", rename_with_transient_lock)
    monkeypatch.setattr(atomic_write.time, "sleep", delays.append)

    result = atomic_write.atomic_rename_directory(staging, target)

    assert result == target
    assert attempts == 2
    assert len(delays) == 1
    assert not staging.exists()
    assert (target / "receipt.json").read_text(encoding="utf-8") == "{}"


def test_atomic_directory_publish_does_not_retry_when_target_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "published"
    staging.mkdir()
    monkeypatch.setattr(atomic_write, "_IS_WINDOWS", True)

    attempts = 0

    def publish_competing_target(source: Path, destination: str | Path) -> Path:
        nonlocal attempts
        attempts += 1
        Path(destination).mkdir()
        raise PermissionError("temporary Windows directory lock")

    monkeypatch.setattr(type(staging), "rename", publish_competing_target)

    with pytest.raises(PermissionError, match="temporary Windows"):
        atomic_write.atomic_rename_directory(staging, target)

    assert attempts == 1
    assert staging.is_dir()
    assert target.is_dir()


def test_atomic_directory_publish_bounds_windows_lock_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "published"
    staging.mkdir()
    monkeypatch.setattr(atomic_write, "_IS_WINDOWS", True)

    attempts = 0
    delays: list[float] = []

    def always_denied(source: Path, destination: str | Path) -> Path:
        nonlocal attempts
        attempts += 1
        raise PermissionError("persistent Windows directory lock")

    monkeypatch.setattr(type(staging), "rename", always_denied)
    monkeypatch.setattr(atomic_write.time, "sleep", delays.append)

    with pytest.raises(PermissionError, match="persistent Windows"):
        atomic_write.atomic_rename_directory(staging, target)

    assert attempts == len(atomic_write._DIRECTORY_RENAME_RETRY_DELAYS) + 1
    assert delays == list(atomic_write._DIRECTORY_RENAME_RETRY_DELAYS)
    assert staging.is_dir()
    assert not target.exists()


def test_atomic_directory_publish_does_not_retry_non_windows_permission_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "published"
    staging.mkdir()
    monkeypatch.setattr(atomic_write, "_IS_WINDOWS", False)

    attempts = 0

    def denied_rename(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError("permanent permission error")

    monkeypatch.setattr(atomic_write, "_rename_directory_no_replace", denied_rename)

    with pytest.raises(PermissionError, match="permanent permission"):
        atomic_write.atomic_rename_directory(staging, target)

    assert attempts == 1
    assert staging.is_dir()
    assert not target.exists()


def test_atomic_directory_publish_never_replaces_racing_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "published"
    staging.mkdir()
    (staging / "receipt.json").write_text("original", encoding="utf-8")
    original_exists = Path.exists
    injected = False

    def insert_competing_directory_during_check(path: Path) -> bool:
        nonlocal injected
        if path == target and not injected:
            injected = True
            target.mkdir()
            return False  # Simulate the existence check immediately before the race.
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", insert_competing_directory_during_check)
    with pytest.raises(FileExistsError):
        atomic_write.atomic_rename_directory(staging, target)

    assert injected
    assert target.is_dir() and not tuple(target.iterdir())
    assert (staging / "receipt.json").read_text(encoding="utf-8") == "original"


def test_atomic_directory_publish_fails_closed_when_native_rename_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "published"
    staging.mkdir()
    monkeypatch.setattr(atomic_write, "_IS_WINDOWS", False)
    monkeypatch.setattr(atomic_write, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(atomic_write.ctypes, "CDLL", lambda *args, **kwargs: object())

    with pytest.raises(OSError, match="renameat2 is unavailable"):
        atomic_write.atomic_rename_directory(staging, target)
    assert staging.is_dir()
    assert not target.exists()


def test_darwin_directory_publish_uses_exclusive_native_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "published"
    staging.mkdir()
    calls: list[tuple[object, ...]] = []

    class NativeRename:
        argtypes: object = None
        restype: object = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            return -1

    rename = NativeRename()
    monkeypatch.setattr(atomic_write, "_IS_WINDOWS", False)
    monkeypatch.setattr(atomic_write, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(
        atomic_write.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(renamex_np=rename),
    )
    monkeypatch.setattr(atomic_write.ctypes, "get_errno", lambda: errno.EEXIST)

    with pytest.raises(FileExistsError):
        atomic_write.atomic_rename_directory(staging, target)
    assert calls == [(str(staging).encode(), str(target).encode(), 0x00000004)]
    assert staging.is_dir()


def test_atomic_directory_publish_rejects_embedded_nul_in_target(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "receipt.json").write_text("keep", encoding="utf-8")
    misleading_target = tmp_path / "other\x00suffix"
    truncated_target = tmp_path / "other"

    with pytest.raises(ValueError, match="NUL"):
        atomic_write.atomic_rename_directory(staging, misleading_target)

    assert (staging / "receipt.json").read_text(encoding="utf-8") == "keep"
    assert not truncated_target.exists()
