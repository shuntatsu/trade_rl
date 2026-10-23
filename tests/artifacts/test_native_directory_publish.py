from __future__ import annotations

import ctypes
import errno
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

import trade_rl.artifacts.atomic_write as atomic_write


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize(
    "error_number",
    [errno.EEXIST, errno.EXDEV, errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, 0],
)
def test_native_failure_preserves_errno_and_paths_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, error_number: int
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "payload").write_bytes(b"preserve source")
    calls: list[tuple[object, ...]] = []

    def native_failure(*args: object) -> int:
        calls.append(args)
        ctypes.set_errno(error_number)
        return -1

    def load_library(name: None, *, use_errno: bool) -> SimpleNamespace:
        assert name is None and use_errno
        return SimpleNamespace(renameat2=native_failure, renamex_np=native_failure)

    def forbidden_rename(*args: object) -> None:
        pytest.fail("native errors must not fall back to ordinary rename")

    monkeypatch.setattr(atomic_write, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(atomic_write.ctypes, "CDLL", load_library)
    monkeypatch.setattr(Path, "rename", forbidden_rename)

    with pytest.raises(OSError) as caught:
        atomic_write.atomic_rename_directory(source, target)

    assert caught.value.errno == (error_number or errno.EIO)
    assert caught.value.filename == str(source)
    assert caught.value.filename2 == str(target)
    assert calls == [
        (-100, os.fsencode(source), -100, os.fsencode(target), 1)
        if platform == "linux"
        else (os.fsencode(source), os.fsencode(target), 4)
    ]
    assert (source / "payload").read_bytes() == b"preserve source"
    assert not target.exists()


@pytest.mark.parametrize("failure", ["missing-symbol", "load-error", "platform"])
def test_unavailable_native_operation_preserves_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "payload").write_bytes(b"complete")

    def load_library(*args: object, **kwargs: object) -> SimpleNamespace:
        if failure == "load-error":
            raise OSError(errno.ENOENT, "library unavailable")
        if failure == "platform":
            pytest.fail("unsupported platform must not load a native library")
        return SimpleNamespace()

    def forbidden_rename(*args: object) -> None:
        pytest.fail("unavailable exclusive rename must not use ordinary rename")

    monkeypatch.setattr(
        atomic_write,
        "sys",
        SimpleNamespace(platform="freebsd14" if failure == "platform" else "linux"),
    )
    monkeypatch.setattr(atomic_write.ctypes, "CDLL", load_library)
    monkeypatch.setattr(Path, "rename", forbidden_rename)
    with pytest.raises(OSError) as caught:
        atomic_write.atomic_rename_directory(source, target)
    assert caught.value.errno == errno.ENOTSUP
    assert (source / "payload").read_bytes() == b"complete"
    assert not target.exists()


@pytest.mark.parametrize("invalid_part", ["source", "target"])
def test_embedded_nul_cannot_publish_to_truncated_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_part: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_bytes(b"complete")
    prefix = tmp_path / "target"

    def forbidden_load(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid path must be rejected before native conversion")

    monkeypatch.setattr(atomic_write.ctypes, "CDLL", forbidden_load)
    with pytest.raises(ValueError, match="null byte"):
        atomic_write._rename_directory_noreplace(
            Path(str(source) + "\0suffix") if invalid_part == "source" else source,
            Path(str(prefix) + "\0suffix") if invalid_part == "target" else prefix,
        )
    assert (source / "payload").read_bytes() == b"complete"
    assert not prefix.exists()


@pytest.mark.parametrize(
    "entry", ["empty-directory", "nonempty-directory", "file", "symlink", "dangling"]
)
def test_real_native_collision_preserves_existing_entry(
    tmp_path: Path, entry: str
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "payload").write_bytes(b"loser")
    referent = tmp_path / "referent"
    if entry in {"symlink", "dangling"}:
        if entry == "symlink":
            referent.mkdir()
            (referent / "winner").write_bytes(b"preserve")
        try:
            target.symlink_to(referent, target_is_directory=True)
        except OSError as error:
            if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                pytest.skip("Windows symlink creation privilege is unavailable")
            raise
    elif entry == "file":
        target.write_bytes(b"preserve")
    else:
        target.mkdir()
        if entry == "nonempty-directory":
            (target / "winner").write_bytes(b"preserve")
    before = target.lstat()
    link_before = target.readlink() if target.is_symlink() else None
    with pytest.raises(OSError):
        atomic_write._rename_directory_noreplace(source, target)
    after = target.lstat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert (source / "payload").read_bytes() == b"loser"
    if entry == "file":
        assert target.read_bytes() == b"preserve"
    elif entry == "nonempty-directory":
        assert (target / "winner").read_bytes() == b"preserve"
    elif entry == "empty-directory":
        assert list(target.iterdir()) == []
    else:
        assert target.is_symlink() and target.readlink() == link_before
        if entry == "symlink":
            assert (referent / "winner").read_bytes() == b"preserve"
        else:
            assert not referent.exists()


def test_real_native_publication_supports_unicode_paths(tmp_path: Path) -> None:
    source = tmp_path / "準備済み"
    target = tmp_path / "公開済み"
    source.mkdir()
    (source / "台帳.json").write_bytes(b"complete")
    assert atomic_write.atomic_rename_directory(source, target) == target
    assert not source.exists()
    assert (target / "台帳.json").read_bytes() == b"complete"


def test_real_native_publication_has_one_winner_after_both_prechecks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = [tmp_path / "first", tmp_path / "second"]
    target = tmp_path / "published"
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "payload").write_bytes(bytes([index]))
    barrier = Barrier(2)
    original_is_symlink = Path.is_symlink

    def synchronize_negative_target_check(path: Path) -> bool:
        observed = original_is_symlink(path)
        if path == target:
            assert observed is False
            barrier.wait(timeout=10)
        return observed

    with monkeypatch.context() as patch:
        patch.setattr(Path, "is_symlink", synchronize_negative_target_check)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(atomic_write.atomic_rename_directory, source, target)
                for source in sources
            ]
    winners = [index for index, future in enumerate(futures) if not future.exception()]
    assert len(winners) == 1
    winner = winners[0]
    loser = 1 - winner
    assert isinstance(futures[loser].exception(), FileExistsError)
    assert futures[winner].result() == target
    assert not sources[winner].exists()
    assert (target / "payload").read_bytes() == bytes([winner])
    assert (sources[loser] / "payload").read_bytes() == bytes([loser])
