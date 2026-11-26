"""Explicit POSIX single-use stage and comment markers for prospective protocols.

No historical ledger is read or migrated. A claim is a local reservation, never
proof of research permission. The caller supplies an independently pinned root.
"""

from __future__ import annotations

import errno
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import _open_directory_descriptor_nofollow


def _positive(value: int) -> None:
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError("expected positive bounded integer identity")


def _digest(value: str, length: int) -> None:
    if (
        type(value) is not str
        or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value)
        or not value.strip("0")
    ):
        raise ValueError("expected nonzero lowercase hexadecimal identity")


def _root_path(path: Path) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.anchor != "/"
        or ".." in path.parts
        or path == Path("/")
        or path.parts[:4] == ("/", "proc", "self", "fd")
        or path.parts[:3] == ("/", "dev", "fd")
    ):
        raise ValueError(
            "stage ledger requires an explicit real absolute POSIX directory"
        )


@contextmanager
def _open_root(path: Path) -> Iterator[int]:
    _root_path(path)
    if (
        os.name != "posix"
        or not getattr(os, "O_NOFOLLOW", 0)
        or not getattr(os, "O_DIRECTORY", 0)
        or os.open not in os.supports_dir_fd
    ):
        raise OSError(errno.ENOTSUP, "strict stage ledger is unsupported")
    chain_fd = _open_directory_descriptor_nofollow(path)
    try:
        fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=chain_fd)
        try:
            first = os.fstat(chain_fd)
            second = os.fstat(fd)
            if (first.st_dev, first.st_ino) != (second.st_dev, second.st_ino):
                raise ValueError("stage ledger directory identity changed")
            yield fd
        finally:
            os.close(fd)
    finally:
        os.close(chain_fd)


@dataclass(frozen=True, slots=True)
class StageLedgerRoot:
    """Locally observed root identity; the trusted operator must pin it."""

    path: Path
    device: int
    inode: int

    def __post_init__(self) -> None:
        _root_path(self.path)
        if (
            type(self.device) is not int
            or self.device < 0
            or type(self.inode) is not int
            or self.inode < 1
        ):
            raise ValueError("invalid ledger directory identity")


def inspect_stage_ledger_root(path: Path) -> StageLedgerRoot:
    """Observe an existing directory without creating it or any marker."""
    with _open_root(path) as fd:
        status = os.fstat(fd)
        return StageLedgerRoot(path, status.st_dev, status.st_ino)


def _check_identity(pin: StageLedgerRoot, fd: int) -> None:
    status = os.fstat(fd)
    if not stat.S_ISDIR(status.st_mode) or (status.st_dev, status.st_ino) != (
        pin.device,
        pin.inode,
    ):
        raise ValueError("stage ledger directory identity differs from its pin")


@dataclass(frozen=True, slots=True)
class StageClaim:
    stage: str
    protocol_digest: str
    commit: str
    repository_id: int
    comment_id: int
    claimed_at: datetime

    def __post_init__(self) -> None:
        if type(self.stage) is not str or not re.fullmatch(
            r"[A-Z][A-Z0-9_]{0,63}", self.stage
        ):
            raise ValueError("expected canonical stage identifier")
        _digest(self.protocol_digest, 64)
        _digest(self.commit, 40)
        _positive(self.repository_id)
        _positive(self.comment_id)
        if type(self.claimed_at) is not datetime:
            raise ValueError("claim timestamp must be an aware datetime")
        offset = self.claimed_at.utcoffset()
        if offset is None:
            raise ValueError("claim timestamp must be an aware datetime")
        # Sample even a stateful tzinfo once; never consult host local time.
        try:
            normalized = (self.claimed_at.replace(tzinfo=None) - offset).replace(
                tzinfo=UTC
            )
        except OverflowError as exc:
            raise ValueError("claim timestamp cannot be normalized to UTC") from exc
        object.__setattr__(self, "claimed_at", normalized)


@dataclass(frozen=True, slots=True)
class StageClaimReservation:
    stage_file: str
    comment_file: str


def _create_marker(directory: int, name: str, payload: bytes) -> None:
    # Exclusive creation consumes this name. Even incomplete marker bytes
    # remain a refusal after failure. Never unlink or retry a consumed marker.
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory,
    )
    try:
        with os.fdopen(fd, "wb") as output:
            fd = -1
            if output.write(payload) != len(payload):
                raise OSError("short stage marker write")
            output.flush()
            os.fsync(output.fileno())
        os.fsync(directory)
    finally:
        if fd >= 0:
            os.close(fd)


def reserve_stage_claim(
    pin: StageLedgerRoot, claim: StageClaim
) -> StageClaimReservation:
    """Reserve one stage and one comment across digest namespaces in one pinned root.

    Call only after separate admission, with a root that the trusted operator
    preserves for the entire protocol family. This POSIX-local primitive
    requires exclusive creation and directory fsync support. A partial failure
    may consume the comment or stage; no automatic reset is allowed. This
    does not coordinate distinct roots, machines, untrusted filesystem users,
    historical v1 ledgers or economic authorization.
    """
    if type(pin) is not StageLedgerRoot or type(claim) is not StageClaim:
        raise ValueError("expected typed root and claim")
    stage_file = f"stage-{claim.stage}.claim"
    comment_file = f"comment-{claim.repository_id}-{claim.comment_id}.claim"
    payload = canonical_json_bytes(
        {
            "schema_version": "single_use_stage_claim_v1",
            "stage": claim.stage,
            "protocol_digest": claim.protocol_digest,
            "commit": claim.commit,
            "repository_id": claim.repository_id,
            "comment_id": claim.comment_id,
            "claimed_at": claim.claimed_at.isoformat(),
        }
    )
    with _open_root(pin.path) as directory:
        _check_identity(pin, directory)
        # Fast refusal before consuming a fresh comment when the stage is
        # already present. O_EXCL remains the actual concurrent arbiter.
        try:
            os.stat(stage_file, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("stage has already been reserved")
        _create_marker(directory, comment_file, payload)
        _create_marker(directory, stage_file, payload)
        with _open_root(pin.path) as rechecked:
            _check_identity(pin, rechecked)
    return StageClaimReservation(stage_file, comment_file)


__all__ = [
    "StageClaim",
    "StageClaimReservation",
    "StageLedgerRoot",
    "inspect_stage_ledger_root",
    "reserve_stage_claim",
]
