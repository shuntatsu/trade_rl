"""Validate immutable training-image provenance before packaging it."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIRTY_VALUES = {"true", "false"}


def provenance_marker_bytes(
    *,
    git_commit: str,
    git_dirty: str,
    source_tree_digest: str,
    lockfile_digest: str,
    runtime_manifest_digest: str = "",
) -> bytes:
    """Return the canonical marker or reject malformed provenance."""

    if _SHA1.fullmatch(git_commit) is None:
        raise ValueError("git commit must be 40 lowercase hexadecimal characters")
    if git_dirty not in _DIRTY_VALUES:
        raise ValueError("git dirty must be true or false")
    if _SHA256.fullmatch(source_tree_digest) is None:
        raise ValueError(
            "source tree digest must be 64 lowercase hexadecimal characters"
        )
    if _SHA256.fullmatch(lockfile_digest) is None:
        raise ValueError("lockfile digest must be 64 lowercase hexadecimal characters")
    if runtime_manifest_digest and _SHA256.fullmatch(runtime_manifest_digest) is None:
        raise ValueError(
            "runtime manifest digest must be empty or 64 lowercase hexadecimal characters"
        )
    return (
        f"{git_commit}:{git_dirty}:{source_tree_digest}:"
        f"{lockfile_digest}:{runtime_manifest_digest}\n"
    ).encode("ascii")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-dirty", required=True)
    parser.add_argument("--source-tree-digest", required=True)
    parser.add_argument("--lockfile-digest", required=True)
    parser.add_argument("--runtime-manifest-digest", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        marker = provenance_marker_bytes(
            git_commit=arguments.git_commit,
            git_dirty=arguments.git_dirty,
            source_tree_digest=arguments.source_tree_digest,
            lockfile_digest=arguments.lockfile_digest,
            runtime_manifest_digest=arguments.runtime_manifest_digest,
        )
    except ValueError as exc:
        parser.error(str(exc))
    arguments.output.write_bytes(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
