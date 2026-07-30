"""State-aware atomic replacement for small pointer artifacts."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path


class AtomicReplaceDurabilityError(OSError):
    """Raised after replacement when directory durability could not be confirmed."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"pointer replacement durability is uncertain: {path}")
        self.path = Path(path)
        self.replaced = True


@dataclass(frozen=True, slots=True)
class AtomicReplaceResult:
    path: Path
    replaced: bool
    durable: bool


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_replace_bytes(path: Path, payload: bytes) -> AtomicReplaceResult:
    """Replace one file and distinguish pre-replace from post-replace failure."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            _fsync_directory(target.parent)
        except OSError as error:
            raise AtomicReplaceDurabilityError(target) from error
        return AtomicReplaceResult(path=target, replaced=True, durable=True)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "AtomicReplaceDurabilityError",
    "AtomicReplaceResult",
    "atomic_replace_bytes",
]
