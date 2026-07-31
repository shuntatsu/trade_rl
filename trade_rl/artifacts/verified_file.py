"""Regular-file and verified private-copy boundaries for unsafe deserializers."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from trade_rl.domain.common import require_sha256


def file_digest(path: Path, *, field: str = "artifact file") -> str:
    digest = hashlib.sha256()
    with open_regular_binary(path, field=field) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def open_regular_binary(path: Path, *, field: str) -> Iterator[BinaryIO]:
    """Open a non-symlink regular file without following a final symlink."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{field} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def read_verified_bytes(
    path: Path,
    *,
    expected_digest: str,
    expected_size_bytes: int,
    field: str,
) -> bytes:
    """Read one exact regular-file byte sequence and verify size and SHA-256."""

    require_sha256(expected_digest, field=f"{field}.expected_digest")
    if (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes < 0
    ):
        raise ValueError(f"{field} expected size must be non-negative")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    with open_regular_binary(path, field=field) as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != expected_size_bytes:
        raise ValueError(f"{field} size mismatch")
    if digest.hexdigest() != expected_digest:
        raise ValueError(f"{field} digest mismatch")
    return b"".join(chunks)


@contextmanager
def verified_private_copy(
    path: Path,
    *,
    expected_digest: str,
    field: str,
    filename: str,
    expected_size_bytes: int | None = None,
) -> Iterator[Path]:
    """Copy verified bytes privately and yield only the immutable copy path."""

    require_sha256(expected_digest, field=f"{field}.expected_digest")
    if expected_size_bytes is not None and (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes < 0
    ):
        raise ValueError(f"{field} expected size must be non-negative")
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise ValueError(f"{field} private filename must be a basename")

    with tempfile.TemporaryDirectory(prefix="trade-rl-verified-") as temporary:
        target = Path(temporary) / safe_name
        copied = 0
        digest = hashlib.sha256()
        with (
            open_regular_binary(path, field=field) as source,
            target.open("xb") as destination,
        ):
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if expected_size_bytes is not None and copied != expected_size_bytes:
            raise ValueError(f"{field} size changed during verified copy")
        if digest.hexdigest() != expected_digest:
            raise ValueError(f"{field} changed during verified copy")
        yield target


__all__ = [
    "file_digest",
    "open_regular_binary",
    "read_verified_bytes",
    "verified_private_copy",
]
