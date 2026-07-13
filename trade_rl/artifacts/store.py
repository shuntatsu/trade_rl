"""Staged, validated, and atomically published research-run artifacts."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes

_RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validated_run_id(run_id: str) -> str:
    if run_id in {".", ".."} or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    return run_id


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


class ArtifactStore:
    """Filesystem store with isolated failures and an atomic latest pointer."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.staging_root = root / ".staging"
        self.runs_root = root / "runs"
        self.failed_root = root / "failed"
        for path in (self.root, self.staging_root, self.runs_root, self.failed_root):
            path.mkdir(parents=True, exist_ok=True)

    def stage_run(self, run_id: str) -> Path:
        """Create and return a new run-specific staging directory."""

        resolved_id = _validated_run_id(run_id)
        stage = self.staging_root / resolved_id
        stage.mkdir(parents=False, exist_ok=False)
        return stage

    def publish_run(
        self,
        run_id: str,
        *,
        validate: Callable[[Path], bool],
    ) -> Path:
        """Validate, publish, and atomically repoint the latest-run identity."""

        resolved_id = _validated_run_id(run_id)
        stage = self.staging_root / resolved_id
        if not stage.is_dir():
            raise FileNotFoundError(f"staged run does not exist: {resolved_id}")
        if not validate(stage):
            raise ValueError(f"artifact validation failed for run {resolved_id}")

        published = self.runs_root / resolved_id
        if published.exists():
            raise FileExistsError(f"published run already exists: {resolved_id}")
        os.replace(stage, published)
        _fsync_directory(self.runs_root)

        pointer = {
            "path": published.relative_to(self.root).as_posix(),
            "run_id": resolved_id,
        }
        _atomic_write(self.root / "latest.json", canonical_json_bytes(pointer))
        return published

    def mark_failed(self, run_id: str) -> Path:
        """Move a partial staged run into the isolated failed-run namespace."""

        resolved_id = _validated_run_id(run_id)
        stage = self.staging_root / resolved_id
        if not stage.is_dir():
            raise FileNotFoundError(f"staged run does not exist: {resolved_id}")
        failed = self.failed_root / resolved_id
        if failed.exists():
            raise FileExistsError(f"failed run already exists: {resolved_id}")
        os.replace(stage, failed)
        _fsync_directory(self.failed_root)
        return failed
