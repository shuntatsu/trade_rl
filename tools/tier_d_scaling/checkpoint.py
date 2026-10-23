"""Atomically publish a cProfile snapshot without stopping its profiler."""

from __future__ import annotations

import cProfile
import marshal
import os
import tempfile
from pathlib import Path


def publish_checkpoint(profiler: cProfile.Profile, path: Path) -> None:
    """Write the current profile snapshot to a new file in its parent directory."""
    profiler.snapshot_stats()
    staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            staging = Path(stream.name)
            marshal.dump(profiler.stats, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, path)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
