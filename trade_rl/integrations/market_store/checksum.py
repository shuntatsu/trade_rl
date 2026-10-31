"""Verification of a Binance Vision archive against its official ``.CHECKSUM`` file.

The official file has the single line ``<sha256 hex>  <archive file name>``. The
archive is accepted only when that line is well formed, names this exact file and its
digest equals the SHA-256 of the downloaded bytes. Anything else fails closed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final

_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")


class ChecksumError(RuntimeError):
    """The archive could not be proven against its official checksum."""


@dataclass(frozen=True, slots=True)
class ChecksumEvidence:
    archive_sha256: str
    checksum_line_sha256: str
    checksum_file_sha256: str
    size_bytes: int


def parse_checksum(text: str, *, filename: str) -> str | None:
    """The SHA-256 named for ``filename``, or None when the file is not exactly that."""

    tokens = text.split()
    if len(tokens) != 2 or _SHA256.match(tokens[0]) is None:
        return None
    if tokens[1].lstrip("*") != filename:
        return None
    return tokens[0]


def verify_archive(
    archive: bytes, checksum_file: bytes, *, filename: str
) -> ChecksumEvidence:
    """Return the evidence, or raise :class:`ChecksumError` (never a partial pass)."""

    if not archive:
        raise ChecksumError(f"{filename}: archive is empty")
    expected = parse_checksum(
        checksum_file.decode("utf-8", errors="replace"), filename=filename
    )
    if expected is None:
        raise ChecksumError(f"{filename}: official checksum file is malformed")
    actual = hashlib.sha256(archive).hexdigest()
    if actual != expected:
        raise ChecksumError(
            f"{filename}: archive sha256 {actual} does not match the official "
            f"checksum {expected}"
        )
    return ChecksumEvidence(
        archive_sha256=actual,
        checksum_line_sha256=expected,
        checksum_file_sha256=hashlib.sha256(checksum_file).hexdigest(),
        size_bytes=len(archive),
    )


__all__ = ["ChecksumError", "ChecksumEvidence", "parse_checksum", "verify_archive"]
