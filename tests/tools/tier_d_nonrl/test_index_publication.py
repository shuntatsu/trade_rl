"""Index replacement retries identical bytes without replaying any native work."""

import os
from pathlib import Path

import pytest

from tools.tier_d_nonrl import tier_d_records as records


def test_transient_windows_index_lock_retries_the_same_closed_staging_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "attempt.json"
    records.write_json(path, {"original": True})
    original = path.read_bytes()
    replace = records.os.replace
    calls, delays = [], []
    monkeypatch.setattr(records, "_WINDOWS", True, raising=False)

    def locked_once(source, target):
        calls.append((Path(source), Path(source).read_bytes()))
        assert path.read_bytes() == original
        if len(calls) == 1:
            raise PermissionError("manufactured Windows sharing lock")
        replace(source, target)

    monkeypatch.setattr(records.os, "replace", locked_once)
    monkeypatch.setattr(records, "_pause_index_retry", delays.append, raising=False)
    records.write_json(path, {"replacement": True}, replace=True)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert delays == [0.01]
    assert records.read_json(path) == {"replacement": True}
    assert not tuple(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("windows,attempts", [(True, 4), (False, 1)])
def test_persistent_index_lock_is_bounded_and_preserves_previous_bytes(
    tmp_path, monkeypatch, windows, attempts
):
    path = tmp_path / "attempt.json"
    records.write_json(path, {"retained": True})
    original = path.read_bytes()
    calls, delays = [], []
    monkeypatch.setattr(records, "_WINDOWS", windows, raising=False)
    monkeypatch.setattr(records, "_pause_index_retry", delays.append, raising=False)

    def locked(source, target):
        calls.append(Path(source))
        assert path.read_bytes() == original
        raise PermissionError("manufactured persistent index lock")

    monkeypatch.setattr(records.os, "replace", locked)
    with pytest.raises(PermissionError, match="persistent"):
        records.write_json(path, {"replacement": True}, replace=True)
    assert len(calls) == attempts and len(set(calls)) == 1
    assert delays == ([0.01, 0.05, 0.15] if windows else [])
    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="controlled Windows sharing lock")
def test_actual_windows_sharing_lock_releases_before_same_index_retry(
    tmp_path, monkeypatch
):
    import ctypes

    path = tmp_path / "attempt.json"
    records.write_json(path, {"retained": True})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create.restype = ctypes.c_void_p
    close = kernel.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    close.restype = ctypes.c_int
    # Permit reads/writes, but hold back FILE_SHARE_DELETE on our own test file.
    handle = create(str(path), 0x80000000, 3, None, 3, 128, None)
    assert handle not in (None, ctypes.c_void_p(-1).value)
    released, errors = False, []
    replace = records.os.replace

    def observe(source, target):
        try:
            replace(source, target)
        except PermissionError as error:
            errors.append(error.winerror)
            raise

    def release(delay):
        nonlocal released
        assert delay == 0.01
        assert close(handle)
        released = True

    monkeypatch.setattr(records.os, "replace", observe)
    monkeypatch.setattr(records, "_pause_index_retry", release)
    try:
        records.write_json(path, {"replacement": True}, replace=True)
        assert errors and errors[0] in {5, 32, 33}
        assert released and records.read_json(path) == {"replacement": True}
    finally:
        if not released:
            assert close(handle)
