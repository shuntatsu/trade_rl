"""Small atomic file-write primitive shared by content-addressed artifacts."""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path


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


__all__ = ["atomic_write_bytes"]
