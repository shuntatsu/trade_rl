"""Crash-consistent generation store for symbol-triplet stage state."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

from trade_rl.artifacts.atomic_pointer import atomic_replace_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
    SymbolTripletStageCompletion,
    load_symbol_triplet_stage_completion,
)
from trade_rl.workflows.symbol_triplet_training_cursor import (
    SymbolTripletTrainingCursor,
    SymbolTripletTrainingPlan,
    load_symbol_triplet_training_cursor,
)

SYMBOL_TRIPLET_STAGE_STATE_POINTER_SCHEMA: Final = (
    "symbol_triplet_stage_state_pointer_v1"
)
_GENERATION_SCHEMA: Final = "symbol_triplet_stage_state_generation_v1"
_CURRENT_NAME: Final = "current.json"
_CURSOR_NAME: Final = "cursor.json"
_COMPLETION_NAME: Final = "completion.json"


class _WindowsFileLockApi(Protocol):
    LK_LOCK: int
    LK_UNLCK: int

    def locking(
        self, _file_descriptor: int, _mode: int, _byte_count: int, /
    ) -> None: ...


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _json_object(path: Path, *, field: str) -> dict[str, object]:
    try:
        with open_regular_binary(path, field=field) as handle:
            raw_bytes = handle.read()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError(f"{field} is missing or unsafe") from error
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must be valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be a JSON object")
    normalized = dict(raw)
    if raw_bytes != canonical_json_bytes(normalized):
        raise ValueError(f"{field} is not canonical")
    return normalized


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".current.lock"
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            lock_api = cast(_WindowsFileLockApi, msvcrt)
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            lock_api.locking(handle.fileno(), lock_api.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                lock_api.locking(handle.fileno(), lock_api.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class SymbolTripletStageStatePointer:
    """Single committed pointer to one immutable stage-state generation."""

    plan_digest: str
    generation_digest: str
    cursor_digest: str
    completion_digest: str | None
    previous_pointer_digest: str | None
    schema_version: str = SYMBOL_TRIPLET_STAGE_STATE_POINTER_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOL_TRIPLET_STAGE_STATE_POINTER_SCHEMA:
            raise ValueError("unsupported symbol-triplet state pointer schema")
        for field, value in (
            ("plan_digest", self.plan_digest),
            ("generation_digest", self.generation_digest),
            ("cursor_digest", self.cursor_digest),
        ):
            require_sha256(value, field=f"stage_state_pointer.{field}")
        for optional_field, optional_value in (
            ("completion_digest", self.completion_digest),
            ("previous_pointer_digest", self.previous_pointer_digest),
        ):
            if optional_value is not None:
                require_sha256(
                    optional_value, field=f"stage_state_pointer.{optional_field}"
                )
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("symbol-triplet state pointer digest mismatch")
        object.__setattr__(self, "digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "completion_digest": self.completion_digest,
            "cursor_digest": self.cursor_digest,
            "generation_digest": self.generation_digest,
            "plan_digest": self.plan_digest,
            "previous_pointer_digest": self.previous_pointer_digest,
            "schema_version": self.schema_version,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> SymbolTripletStageStatePointer:
        required = {
            "completion_digest",
            "cursor_digest",
            "digest",
            "generation_digest",
            "plan_digest",
            "previous_pointer_digest",
            "schema_version",
        }
        if set(value) != required:
            raise ValueError("symbol-triplet state pointer field closure mismatch")
        for field in (
            "cursor_digest",
            "digest",
            "generation_digest",
            "plan_digest",
            "schema_version",
        ):
            if not isinstance(value[field], str):
                raise ValueError(f"stage state pointer {field} must be a string")
        for field in ("completion_digest", "previous_pointer_digest"):
            if value[field] is not None and not isinstance(value[field], str):
                raise ValueError(
                    f"stage state pointer {field} must be a string or null"
                )
        return cls(
            plan_digest=cast(str, value["plan_digest"]),
            generation_digest=cast(str, value["generation_digest"]),
            cursor_digest=cast(str, value["cursor_digest"]),
            completion_digest=cast(str | None, value["completion_digest"]),
            previous_pointer_digest=cast(str | None, value["previous_pointer_digest"]),
            schema_version=cast(str, value["schema_version"]),
            digest=cast(str, value["digest"]),
        )


class SymbolTripletStageStateStore:
    """Publish cursor/completion pairs through one atomic current pointer."""

    def __init__(self, root: str | Path, *, plan: SymbolTripletTrainingPlan) -> None:
        self.root = Path(root)
        self.plan = plan
        self.generations_root = self.root / "generations"
        self.current_path = self.root / _CURRENT_NAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.generations_root.mkdir(parents=True, exist_ok=True)

    def _generation_digest(
        self,
        *,
        cursor: SymbolTripletTrainingCursor,
        completion: SymbolTripletStageCompletion | None,
        previous_pointer_digest: str | None,
    ) -> str:
        return content_digest(
            {
                "completion_digest": (
                    None if completion is None else completion.digest
                ),
                "cursor_digest": cursor.digest,
                "plan_digest": self.plan.digest,
                "previous_pointer_digest": previous_pointer_digest,
                "schema_version": _GENERATION_SCHEMA,
            }
        )

    def _publish_generation(
        self,
        *,
        cursor: SymbolTripletTrainingCursor,
        completion: SymbolTripletStageCompletion | None,
        previous_pointer_digest: str | None,
    ) -> tuple[Path, str]:
        cursor.validate_plan(self.plan)
        if completion is None:
            if cursor.next_stage_index != 0:
                raise ValueError(
                    "advanced stage cursor requires completion in its generation"
                )
        else:
            completion.validate_plan(self.plan)
            if cursor.last_completion_digest != completion.digest:
                raise ValueError(
                    "stage generation completion does not match cursor digest"
                )
            if cursor.last_completed_stage_id != completion.stage_id:
                raise ValueError(
                    "stage generation completion does not match cursor stage"
                )
            if cursor.next_stage_index != completion.stage_index + 1:
                raise ValueError(
                    "stage generation cursor index does not follow completion"
                )
        generation_digest = self._generation_digest(
            cursor=cursor,
            completion=completion,
            previous_pointer_digest=previous_pointer_digest,
        )
        generation_root = self.generations_root / generation_digest
        try:
            generation_root.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            try:
                loaded_completion, loaded_cursor = self._load_generation(
                    generation_digest,
                    completion_digest=(
                        None if completion is None else completion.digest
                    ),
                    cursor_digest=cursor.digest,
                )
            except (OSError, ValueError):
                if self.current_path.exists():
                    current_pointer = self._load_pointer()
                    if current_pointer.generation_digest == generation_digest:
                        raise ValueError("committed stage generation is incomplete")
                if generation_root.is_symlink():
                    generation_root.unlink()
                else:
                    shutil.rmtree(generation_root)
                generation_root.mkdir(parents=False, exist_ok=False)
            else:
                if loaded_cursor != cursor or loaded_completion != completion:
                    raise ValueError(
                        "stage generation path already contains different state"
                    )
                return generation_root, generation_digest
        try:
            _write_exclusive(
                generation_root / _CURSOR_NAME,
                canonical_json_bytes(cursor.to_json_dict()),
            )
            if completion is not None:
                _write_exclusive(
                    generation_root / _COMPLETION_NAME,
                    canonical_json_bytes(completion.to_json_dict()),
                )
            _fsync_directory(generation_root)
            _fsync_directory(self.generations_root)
            loaded_completion, loaded_cursor = self._load_generation(
                generation_digest,
                completion_digest=(None if completion is None else completion.digest),
                cursor_digest=cursor.digest,
            )
            if loaded_cursor != cursor or loaded_completion != completion:
                raise ValueError("published stage generation failed exact reload")
            return generation_root, generation_digest
        except BaseException:
            # Keep partially written or complete generations unreachable. Recovery only
            # trusts current.json, and orphan generations can be garbage-collected.
            raise

    def _load_generation(
        self,
        generation_digest: str,
        *,
        completion_digest: str | None,
        cursor_digest: str,
    ) -> tuple[SymbolTripletStageCompletion | None, SymbolTripletTrainingCursor]:
        require_sha256(generation_digest, field="generation_digest")
        root = self.generations_root / generation_digest
        if not root.is_dir() or root.is_symlink():
            raise ValueError("stage state generation is missing or unsafe")
        cursor = load_symbol_triplet_training_cursor(
            root / _CURSOR_NAME,
            plan=self.plan,
        )
        if cursor.digest != cursor_digest:
            raise ValueError("stage state generation cursor digest mismatch")
        completion_path = root / _COMPLETION_NAME
        if completion_digest is None:
            if completion_path.exists():
                raise ValueError(
                    "initial stage generation contains unexpected completion"
                )
            completion = None
        else:
            completion = load_symbol_triplet_stage_completion(
                completion_path,
                plan=self.plan,
            )
            if completion.digest != completion_digest:
                raise ValueError("stage state generation completion digest mismatch")
            if cursor.last_completion_digest != completion.digest:
                raise ValueError("stage state generation cursor/completion mismatch")
        return completion, cursor

    def _load_pointer(self) -> SymbolTripletStageStatePointer:
        raw = _json_object(self.current_path, field="stage state current pointer")
        pointer = SymbolTripletStageStatePointer.from_mapping(raw)
        if pointer.plan_digest != self.plan.digest:
            raise ValueError("stage state current pointer plan mismatch")
        return pointer

    def load_current(
        self,
    ) -> tuple[
        SymbolTripletStageCompletion | None,
        SymbolTripletTrainingCursor,
        SymbolTripletStageStatePointer,
    ]:
        pointer = self._load_pointer()
        completion, cursor = self._load_generation(
            pointer.generation_digest,
            completion_digest=pointer.completion_digest,
            cursor_digest=pointer.cursor_digest,
        )
        return completion, cursor, pointer

    def initialize(
        self,
        cursor: SymbolTripletTrainingCursor,
        *,
        completion: SymbolTripletStageCompletion | None = None,
    ) -> SymbolTripletStageStatePointer:
        """Create generation zero or validate an existing current generation."""

        with _exclusive_lock(self.root):
            if self.current_path.exists():
                current_completion, current_cursor, pointer = self.load_current()
                if current_cursor != cursor or current_completion != completion:
                    raise ValueError(
                        "stage state is already initialized with different state"
                    )
                return pointer
            _, generation_digest = self._publish_generation(
                cursor=cursor,
                completion=completion,
                previous_pointer_digest=None,
            )
            pointer = SymbolTripletStageStatePointer(
                plan_digest=self.plan.digest,
                generation_digest=generation_digest,
                cursor_digest=cursor.digest,
                completion_digest=(None if completion is None else completion.digest),
                previous_pointer_digest=None,
            )
            atomic_replace_bytes(
                self.current_path,
                canonical_json_bytes(pointer.to_json_dict()),
            )
            return pointer

    def commit(
        self,
        *,
        expected_cursor_digest: str,
        completion: SymbolTripletStageCompletion,
        cursor: SymbolTripletTrainingCursor,
    ) -> SymbolTripletStageStatePointer:
        """Publish a new generation and atomically advance current.json."""

        require_sha256(expected_cursor_digest, field="expected_cursor_digest")
        with _exclusive_lock(self.root):
            _, current_cursor, current_pointer = self.load_current()
            if current_cursor.digest != expected_cursor_digest:
                raise ValueError("stale symbol-triplet stage cursor digest")
            _, generation_digest = self._publish_generation(
                cursor=cursor,
                completion=completion,
                previous_pointer_digest=current_pointer.digest,
            )
            pointer = SymbolTripletStageStatePointer(
                plan_digest=self.plan.digest,
                generation_digest=generation_digest,
                cursor_digest=cursor.digest,
                completion_digest=completion.digest,
                previous_pointer_digest=current_pointer.digest,
            )
            atomic_replace_bytes(
                self.current_path,
                canonical_json_bytes(pointer.to_json_dict()),
            )
            return pointer

    def migrate_legacy(
        self,
        *,
        cursor_path: str | Path,
        completion_path: str | Path | None,
    ) -> SymbolTripletStageStatePointer:
        """Validate a legacy pair and publish it as one authoritative generation."""

        cursor = load_symbol_triplet_training_cursor(cursor_path, plan=self.plan)
        completion = (
            None
            if completion_path is None
            else load_symbol_triplet_stage_completion(completion_path, plan=self.plan)
        )
        if cursor.next_stage_index == 0 and completion is not None:
            raise ValueError("initial legacy cursor may not have completion")
        if cursor.next_stage_index > 0:
            if completion is None:
                raise ValueError("advanced legacy cursor requires completion")
            if cursor.last_completion_digest != completion.digest:
                raise ValueError("legacy cursor/completion digest mismatch")
        return self.initialize(cursor, completion=completion)


def load_or_migrate_symbol_triplet_stage_state(
    *,
    plan: SymbolTripletTrainingPlan,
    state_root: str | Path,
    legacy_cursor_path: str | Path,
    legacy_completion_path: str | Path | None,
) -> tuple[
    SymbolTripletStageCompletion | None,
    SymbolTripletTrainingCursor,
    SymbolTripletStageStatePointer,
]:
    """Load authoritative generation state, migrating one validated legacy pair."""

    store = SymbolTripletStageStateStore(state_root, plan=plan)
    if not store.current_path.exists():
        store.migrate_legacy(
            cursor_path=legacy_cursor_path,
            completion_path=legacy_completion_path,
        )
    return store.load_current()


__all__ = [
    "SYMBOL_TRIPLET_STAGE_STATE_POINTER_SCHEMA",
    "SymbolTripletStageStatePointer",
    "SymbolTripletStageStateStore",
    "load_or_migrate_symbol_triplet_stage_state",
]
