"""Regular-file and verified private-copy boundaries for unsafe deserializers."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from trade_rl._validation import require_sha256


def file_digest_and_size(
    path: Path,
    *,
    field: str = "artifact file",
) -> tuple[str, int]:
    """Return SHA-256 and byte size from one opened regular-file snapshot."""

    digest = hashlib.sha256()
    size = 0
    with open_regular_binary(path, field=field) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def file_digest(path: Path, *, field: str = "artifact file") -> str:
    return file_digest_and_size(path, field=field)[0]


def _open_regular_file_descriptor(source: Path, *, field: str) -> int:
    """Open with a no-follow directory chain where descriptor-relative I/O exists."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if (
        os.name == "posix"
        and nofollow
        and directory_flag
        and os.open in os.supports_dir_fd
    ):
        directory_flags = (
            getattr(os, "O_PATH", os.O_RDONLY) | directory_flag | nofollow
        )
        parent_fd = os.open(source.anchor, directory_flags)
        try:
            for component in source.parts[1:-1]:
                try:
                    next_fd = os.open(
                        component,
                        directory_flags,
                        dir_fd=parent_fd,
                    )
                except OSError as error:
                    if error.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise ValueError(
                            f"{field} parent must not contain a symlink or non-directory"
                        ) from error
                    raise
                os.close(parent_fd)
                parent_fd = next_fd
            # Even a concurrent replacement of an ancestor cannot redirect the
            # opened descriptor chain outside the inspected directory objects.
            return os.open(
                source.name,
                os.O_RDONLY | nofollow,
                dir_fd=parent_fd,
            )
        finally:
            os.close(parent_fd)

    # Portable fallback for platforms without POSIX dir_fd/O_NOFOLLOW. Static
    # ancestors (including Windows directory junctions) must still fail closed;
    # this fallback does not claim atomic protection against concurrent swaps.
    for ancestor in source.parents:
        if ancestor.is_symlink() or ancestor.is_junction():
            raise ValueError(f"{field} parent must not contain a symlink or junction")
    return os.open(source, os.O_RDONLY | nofollow)


@contextmanager
def open_regular_binary(path: Path, *, field: str) -> Iterator[BinaryIO]:
    """Read a regular file without traversing symlinked directory ancestors."""

    source = Path(path)
    if ".." in source.parts:
        raise ValueError(f"{field} path must not contain parent traversal")
    source = source.absolute()
    if source.is_symlink() or source.is_junction():
        raise ValueError(f"{field} must not be a symlink or junction")
    descriptor = _open_regular_file_descriptor(source, field=field)
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
    "file_digest_and_size",
    "open_regular_binary",
    "read_verified_bytes",
    "verified_private_copy",
]
