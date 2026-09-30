"""Small atomic file-write primitive shared by content-addressed artifacts."""

from __future__ import annotations

import base64
import os
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


def atomic_rename_directory(source: str | Path, target: str | Path) -> Path:
    """Atomically publish a staged directory, retrying transient Windows locks."""

    staging = Path(source)
    output = Path(target)
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("directory publication source must be a regular directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"directory publication target already exists: {output}")
    delays = _DIRECTORY_RENAME_RETRY_DELAYS if _IS_WINDOWS else ()
    for attempt in range(len(delays) + 1):
        try:
            staging.rename(output)
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
