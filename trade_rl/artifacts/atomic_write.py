"""Atomic file replacement and exclusive directory publication primitives."""

from __future__ import annotations

import base64
import ctypes
import errno
import os
import sys
import time
import uuid
from pathlib import Path

_IS_WINDOWS = os.name == "nt"
_DIRECTORY_RENAME_RETRY_DELAYS = (0.01, 0.05, 0.15)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_name_suffix() -> str:
    """Return a filename-safe encoding of a full 128-bit UUID."""

    return base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("ascii").rstrip("=")


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Atomically replace *path* with fully flushed bytes and clean temporary files."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp-{os.getpid()}-{_temporary_name_suffix()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Commit once using a native operation that refuses every target entry."""

    if "\0" in str(source) or "\0" in str(target):
        raise ValueError("embedded null byte in directory publication path")
    if sys.platform == "win32":
        source.rename(target)
        return
    if sys.platform not in {"linux", "darwin"}:
        raise OSError(
            errno.ENOTSUP,
            "exclusive directory rename is unavailable on this platform",
            str(source),
            None,
            str(target),
        )
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    try:
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renameat2 if sys.platform == "linux" else library.renamex_np
    except (AttributeError, OSError) as error:
        raise OSError(
            errno.ENOTSUP,
            "native exclusive directory rename is unavailable",
            str(source),
            None,
            str(target),
        ) from error
    rename.restype = ctypes.c_int
    if sys.platform == "linux":
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        ctypes.set_errno(0)
        # Linux UAPI: AT_FDCWD=-100, RENAME_NOREPLACE=1.
        result = rename(-100, source_bytes, -100, target_bytes, 1)
    else:
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        ctypes.set_errno(0)
        # Darwin renamex_np: RENAME_EXCL=4.
        result = rename(source_bytes, target_bytes, 4)
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(
            error_number, os.strerror(error_number), str(source), None, str(target)
        )


def atomic_rename_directory(source: str | Path, target: str | Path) -> Path:
    """Exclusively publish a staged directory, retrying transient Windows locks."""

    staging = Path(source)
    output = Path(target)
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("directory publication source must be a regular directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"directory publication target already exists: {output}")
    delays = _DIRECTORY_RENAME_RETRY_DELAYS if _IS_WINDOWS else ()
    for attempt in range(len(delays) + 1):
        try:
            _rename_directory_noreplace(staging, output)
            return output
        except PermissionError:
            if attempt >= len(delays):
                raise
            if (
                staging.is_symlink()
                or not staging.is_dir()
                or output.exists()
                or output.is_symlink()
            ):
                raise
            time.sleep(delays[attempt])
    raise AssertionError("directory rename retry loop exited unexpectedly")


__all__ = ["atomic_rename_directory", "atomic_write_bytes"]
