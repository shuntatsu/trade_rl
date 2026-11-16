"""Regular-file and verified private-copy boundaries for unsafe deserializers."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import BinaryIO

from trade_rl._validation import require_sha256


# Only this process's active Study publication contexts may name an open
# directory capability through /proc/self/fd or /dev/fd. A caller-provided
# alias to any other process FD must remain an untrusted symlink ancestor.
_TRUSTED_DIRECTORY_FDS: ContextVar[frozenset[tuple[int, int, int]]] = ContextVar(
    "trade_rl_trusted_publication_directory_fds", default=frozenset()
)


@contextmanager
def _trusted_process_directory_fd(descriptor: int) -> Iterator[None]:
    """Temporarily authorize one held directory inode for nested Study paths."""

    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode):
        raise ValueError("trusted publication descriptor is not a directory")
    identity = (descriptor, status.st_dev, status.st_ino)
    token = _TRUSTED_DIRECTORY_FDS.set(
        _TRUSTED_DIRECTORY_FDS.get() | frozenset((identity,))
    )
    try:
        yield
    finally:
        _TRUSTED_DIRECTORY_FDS.reset(token)


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


def _open_directory_descriptor_nofollow(source: Path) -> int:
    """Open a POSIX directory chain without following untrusted symlinks.

    /proc/self/fd/N and /dev/fd/N may name a trusted descriptor capability
    only when the enclosing Study publisher registered its exact open inode.
    Unregistered process-FD paths remain forbidden symlinked ancestors.
    All remaining children and ordinary paths use O_NOFOLLOW. The caller owns
    and must close the returned fd.
    """

    if (
        os.name != "posix"
        or not getattr(os, "O_NOFOLLOW", 0)
        or not getattr(os, "O_DIRECTORY", 0)
        or os.open not in os.supports_dir_fd
    ):
        raise OSError(errno.ENOTSUP, "directory-descriptor no-follow is unsupported")

    path = Path(os.path.abspath(source))
    if ".." in path.parts:
        raise OSError(errno.ENOTDIR, "directory parent traversal is forbidden")
    components = path.parts
    flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | os.O_NOFOLLOW
    capability_index: int | None = None
    if components[:4] == ("/", "proc", "self", "fd"):
        capability_index = 4
    elif components[:3] == ("/", "dev", "fd"):
        capability_index = 3

    if (
        capability_index is not None
        and len(components) > capability_index
        and components[capability_index].isascii()
        and components[capability_index].isdecimal()
    ):
        # The proc/dev FD alias is a privileged internal capability only for
        # the exact inode registered by this thread's active Study publisher.
        # An arbitrary caller-supplied number must not bypass ancestor nofollow.
        source_fd = int(components[capability_index])
        descriptor = os.dup(source_fd)
        status = os.fstat(descriptor)
        identity = (source_fd, status.st_dev, status.st_ino)
        if not stat.S_ISDIR(status.st_mode):
            os.close(descriptor)
            raise OSError(errno.ENOTDIR, "process descriptor is not a directory")
        if identity not in _TRUSTED_DIRECTORY_FDS.get():
            os.close(descriptor)
            raise OSError(errno.ELOOP, "untrusted process directory FD alias")
        start = capability_index + 1
    else:
        descriptor = os.open(path.anchor, flags)
        start = 1

    try:
        for component in components[start:]:
            next_fd = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_regular_file_descriptor(source: Path, *, field: str) -> int:
    """Open with a no-follow directory chain where descriptor-relative I/O exists."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    # A read-only FIFO open waits for a writer before fstat can reject it.
    # Nonblocking applies only to POSIX leaf opens; regular reads are unaffected.
    leaf_flags = os.O_RDONLY | nofollow
    if os.name == "posix":
        leaf_flags |= getattr(os, "O_NONBLOCK", 0)
    if (
        os.name == "posix"
        and nofollow
        and directory_flag
        and os.open in os.supports_dir_fd
    ):
        try:
            parent_fd = _open_directory_descriptor_nofollow(source.parent)
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError(
                    f"{field} parent must not contain a symlink or non-directory"
                ) from error
            raise
        try:
            # The verified parent is a held capability, not a rechecked path.
            return os.open(source.name, leaf_flags, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)

    # Portable fallback for platforms without POSIX dir_fd/O_NOFOLLOW. Static
    # ancestors (including Windows directory junctions) must still fail closed;
    # this fallback does not claim atomic protection against concurrent swaps.
    for ancestor in source.parents:
        if ancestor.is_symlink() or ancestor.is_junction():
            raise ValueError(f"{field} parent must not contain a symlink or junction")
    return os.open(source, leaf_flags)


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
            # Read at most one byte past the declared boundary so an oversized
            # source is refused before consuming or buffering the excess.
            chunk = handle.read(min(1024 * 1024, expected_size_bytes - size + 1))
            if not chunk:
                break
            if size + len(chunk) > expected_size_bytes:
                raise ValueError(f"{field} size mismatch")
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
                read_limit = 1024 * 1024
                if expected_size_bytes is not None:
                    read_limit = min(read_limit, expected_size_bytes - copied + 1)
                chunk = source.read(read_limit)
                if not chunk:
                    break
                if (
                    expected_size_bytes is not None
                    and copied + len(chunk) > expected_size_bytes
                ):
                    raise ValueError(f"{field} size changed during verified copy")
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
