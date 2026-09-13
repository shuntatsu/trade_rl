"""Recovery helpers for Experiment 0003 fresh-runner verification.

The original pre-upload binding used an absolute-path-sensitive shell tree hash.
This module adds a canonical tree digest that depends only on relative paths and
file bytes, so the same immutable candidate tree has the same digest after a
fresh-runner re-download into a different temporary root.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _file_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def canonical_tree_digest(root: Path) -> str:
    """Hash immutable file relative paths and bytes without depending on root."""

    if not root.is_dir():
        raise RuntimeError(f"immutable tree missing: {root}")
    files = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".mutation.lock"
    )
    if not files:
        raise RuntimeError("immutable tree has no immutable files")

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(_file_sha256(path))
    return digest.hexdigest()


__all__ = ["canonical_tree_digest"]
