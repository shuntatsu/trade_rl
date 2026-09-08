"""Append-only filesystem mechanics for controlled experiment Studies."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, cast

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    InvalidExperimentStateError,
)

_LOCK_NAME = ".mutation.lock"


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        locking = getattr(msvcrt, "locking")
        lock_mode = getattr(msvcrt, "LK_LOCK")
        locking(handle.fileno(), lock_mode, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        locking = getattr(msvcrt, "locking")
        unlock_mode = getattr(msvcrt, "LK_UNLCK")
        locking(handle.fileno(), unlock_mode, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class StudyStore:
    """Own append-only publication, safe paths, and the Study mutation lock."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise ArtifactIntegrityError("Study root must be a regular directory")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ArtifactIntegrityError("Study root must not be a symlink")
        self._thread_lock = threading.RLock()
        self._lock_depth = 0

    def _relative_path(self, relative: str | Path) -> Path:
        value = Path(relative)
        if value.is_absolute():
            raise ValueError("Study path must be relative")
        if not value.parts or value == Path("."):
            raise ValueError("Study path must be a non-empty relative path")
        if ".." in value.parts:
            raise ValueError("Study path must not contain parent traversal")
        if value.parts[0] == _LOCK_NAME:
            raise ValueError("Study path is reserved for the mutation lock")
        return value

    def _checked_target(self, relative: str | Path) -> Path:
        value = self._relative_path(relative)
        current = self.root
        for part in value.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ArtifactIntegrityError("Study path parent must not be a symlink")
            if current.exists() and not current.is_dir():
                raise ArtifactIntegrityError(
                    "Study path parent must be a regular directory"
                )
        return self.root / value

    def _mkdir_checked_parents(self, target: Path) -> None:
        relative_parent = target.parent.relative_to(self.root)
        current = self.root
        for part in relative_parent.parts:
            if part == ".":
                continue
            current = current / part
            if current.is_symlink():
                raise ArtifactIntegrityError("Study path parent must not be a symlink")
            if current.exists():
                if not current.is_dir():
                    raise ArtifactIntegrityError(
                        "Study path parent must be a regular directory"
                    )
                continue
            current.mkdir()

    @contextmanager
    def mutation_lock(self) -> Iterator[None]:
        """Serialize Study mutation across processes and support same-store nesting."""

        with self._thread_lock:
            if self._lock_depth > 0:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return

            if self.root.is_symlink() or not self.root.is_dir():
                raise ArtifactIntegrityError("Study root must be a regular directory")
            lock_path = self.root / _LOCK_NAME
            if lock_path.is_symlink():
                raise ArtifactIntegrityError(
                    "Study mutation lock must not be a symlink"
                )
            with lock_path.open("a+b") as handle:
                _lock_file(handle)
                self._lock_depth = 1
                try:
                    yield
                finally:
                    self._lock_depth = 0
                    _unlock_file(handle)

    def publish_json_once(self, relative: str | Path, value: object) -> Path:
        """Atomically publish canonical JSON without replacing existing evidence."""

        target = self._checked_target(relative)
        self._mkdir_checked_parents(target)
        if target.exists() or target.is_symlink():
            raise InvalidExperimentStateError(
                f"Study artifact already exists: {relative}"
            )

        staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
        try:
            with staging.open("xb") as handle:
                handle.write(canonical_json_bytes(value))
                handle.flush()
                os.fsync(handle.fileno())
            if target.exists() or target.is_symlink():
                raise InvalidExperimentStateError(
                    f"Study artifact already exists: {relative}"
                )
            staging.rename(target)
        finally:
            if staging.exists() or staging.is_symlink():
                staging.unlink(missing_ok=True)
        return target

    def read_json(self, relative: str | Path) -> dict[str, object]:
        """Read one immutable JSON object through the safe Study path boundary."""

        target = self._checked_target(relative)
        if target.is_symlink() or not target.is_file():
            raise ArtifactIntegrityError("Study artifact must be a regular file")
        try:
            raw = cast(object, json.loads(target.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("Study artifact JSON is malformed") from error
        if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
            raise ArtifactIntegrityError("Study artifact must contain a JSON object")
        return cast(dict[str, object], raw)

    def publish_directory_once(
        self,
        relative: str | Path,
        builder: Callable[[Path], None],
    ) -> Path:
        """Build a private staging directory then atomically publish it once."""

        target = self._checked_target(relative)
        self._mkdir_checked_parents(target)
        if target.exists() or target.is_symlink():
            raise InvalidExperimentStateError(
                f"Study artifact already exists: {relative}"
            )

        staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
        staging.mkdir()
        try:
            builder(staging)
            if target.exists() or target.is_symlink():
                raise InvalidExperimentStateError(
                    f"Study artifact already exists: {relative}"
                )
            staging.rename(target)
        finally:
            if staging.exists() or staging.is_symlink():
                if staging.is_dir() and not staging.is_symlink():
                    shutil.rmtree(staging)
                else:
                    staging.unlink(missing_ok=True)
        return target


__all__ = ["StudyStore"]
