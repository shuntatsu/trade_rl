"""Content-addressed storage for verified Stage A execution replays."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.stage_a_execution_replay import (
    StageAExecutionReplayArtifact,
    build_stage_a_execution_replay_artifact,
    validate_stage_a_execution_replay_sources,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)

STAGE_A_EXECUTION_REQUEST_INDEX_SCHEMA: Final = (
    "stage_a_execution_request_index_v1"
)
_EVENT_SUFFIX: Final = ".order-events.json"
_EVIDENCE_SUFFIX: Final = ".execution-evidence.json"
_CELL_SUFFIX: Final = ".stage-a-cell.json"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path, *, field: str) -> bytes:
    try:
        with open_regular_binary(path, field=field) as handle:
            return handle.read()
    except FileNotFoundError as error:
        raise ValueError(f"{field} is missing") from error
    except OSError as error:
        raise ValueError(f"{field} could not be read safely") from error


def _write_idempotent(path: Path, payload: bytes, *, field: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
        return path
    except FileExistsError:
        existing = _read_regular_bytes(path, field=field)
        if existing != payload:
            raise ValueError(f"{field} already exists with different bytes") from None
        return path


def _relative_path(value: object, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a canonical relative path")
    return path


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class StoredStageAExecutionReplay:
    """One replay and the canonical files that prove its execution identity."""

    root: Path
    artifact: StageAExecutionReplayArtifact
    artifact_path: Path
    event_path: Path
    evidence_path: Path
    index_path: Path

    def __post_init__(self) -> None:
        root = Path(self.root)
        expected_artifact = (
            root
            / "cells"
            / self.artifact.cell_identity.request_digest
            / f"{self.artifact.digest}{_CELL_SUFFIX}"
        )
        expected_event = (
            root
            / "events"
            / f"{self.artifact.event_artifact_digest}{_EVENT_SUFFIX}"
        )
        expected_evidence = (
            root
            / "evidence"
            / f"{self.artifact.execution_evidence_digest}{_EVIDENCE_SUFFIX}"
        )
        expected_index = (
            root
            / "by-request"
            / f"{self.artifact.cell_identity.request_digest}.json"
        )
        if Path(self.artifact_path) != expected_artifact:
            raise ValueError("Stage A execution artifact path is not canonical")
        if Path(self.event_path) != expected_event:
            raise ValueError("Stage A execution event path is not canonical")
        if Path(self.evidence_path) != expected_evidence:
            raise ValueError("Stage A execution evidence path is not canonical")
        if Path(self.index_path) != expected_index:
            raise ValueError("Stage A execution request index path is not canonical")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "artifact_path", expected_artifact)
        object.__setattr__(self, "event_path", expected_event)
        object.__setattr__(self, "evidence_path", expected_evidence)
        object.__setattr__(self, "index_path", expected_index)


class StageAExecutionPromotionStore:
    """Publish and reload immutable Stage A execution cells by request digest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _paths(
        self, artifact: StageAExecutionReplayArtifact
    ) -> tuple[Path, Path, Path, Path]:
        request_digest = artifact.cell_identity.request_digest
        artifact_path = (
            self.root
            / "cells"
            / request_digest
            / f"{artifact.digest}{_CELL_SUFFIX}"
        )
        event_path = (
            self.root
            / "events"
            / f"{artifact.event_artifact_digest}{_EVENT_SUFFIX}"
        )
        evidence_path = (
            self.root
            / "evidence"
            / f"{artifact.execution_evidence_digest}{_EVIDENCE_SUFFIX}"
        )
        index_path = self.root / "by-request" / f"{request_digest}.json"
        return artifact_path, event_path, evidence_path, index_path

    @staticmethod
    def _index_bytes(
        *, request_digest: str, artifact: StageAExecutionReplayArtifact, root: Path
    ) -> bytes:
        artifact_path = (
            root
            / "cells"
            / request_digest
            / f"{artifact.digest}{_CELL_SUFFIX}"
        )
        payload: dict[str, object] = {
            "artifact_digest": artifact.digest,
            "artifact_path": artifact_path.relative_to(root).as_posix(),
            "request_digest": request_digest,
            "schema_version": STAGE_A_EXECUTION_REQUEST_INDEX_SCHEMA,
        }
        return canonical_json_bytes({"digest": content_digest(payload), **payload}) + b"\n"

    def publish(
        self,
        *,
        request: StageAEvaluationCellRequest,
        candidate_config_digest: str,
        actions: tuple[tuple[float, ...], ...],
        observation_digests: tuple[str, ...],
        equity_curve: tuple[float, ...],
        event_artifact_path: str | Path,
        execution_evidence_path: str | Path,
    ) -> StoredStageAExecutionReplay:
        """Publish one request exactly once, accepting only identical retries."""

        event_bytes = _read_regular_bytes(
            Path(event_artifact_path), field="Stage A source execution event artifact"
        )
        evidence_bytes = _read_regular_bytes(
            Path(execution_evidence_path),
            field="Stage A source execution evidence artifact",
        )
        artifact = build_stage_a_execution_replay_artifact(
            request=request,
            candidate_config_digest=candidate_config_digest,
            actions=actions,
            observation_digests=observation_digests,
            equity_curve=equity_curve,
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes,
        )
        artifact_path, event_path, evidence_path, index_path = self._paths(artifact)
        index_bytes = self._index_bytes(
            request_digest=request.digest, artifact=artifact, root=self.root
        )

        if index_path.exists() or index_path.is_symlink():
            existing_index = _read_regular_bytes(
                index_path, field="Stage A execution request index"
            )
            if existing_index != index_bytes:
                raise ValueError("Stage A execution request is already bound")
            loaded = self.load(request.digest)
            if loaded.artifact != artifact:
                raise ValueError("Stage A execution request is already bound")
            return loaded

        _write_idempotent(
            event_path, event_bytes, field="Stage A execution event artifact"
        )
        _write_idempotent(
            evidence_path,
            evidence_bytes,
            field="Stage A execution evidence artifact",
        )
        _write_idempotent(
            artifact_path,
            artifact.raw_bytes,
            field="Stage A execution replay artifact",
        )
        try:
            _write_idempotent(
                index_path,
                index_bytes,
                field="Stage A execution request index",
            )
        except ValueError as error:
            raise ValueError("Stage A execution request is already bound") from error
        return self.load(request.digest)

    def load(self, request_digest: str) -> StoredStageAExecutionReplay:
        """Load one execution cell only through its immutable request index."""

        require_sha256(request_digest, field="stage_a_execution_request_digest")
        index_path = self.root / "by-request" / f"{request_digest}.json"
        index_bytes = _read_regular_bytes(
            index_path, field="Stage A execution request index"
        )
        try:
            raw = json.loads(index_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Stage A execution request index must be valid JSON") from error
        if not isinstance(raw, dict):
            raise ValueError("Stage A execution request index must be an object")
        required = {
            "artifact_digest",
            "artifact_path",
            "digest",
            "request_digest",
            "schema_version",
        }
        if set(raw) != required:
            raise ValueError("Stage A execution request index field closure mismatch")
        if raw["schema_version"] != STAGE_A_EXECUTION_REQUEST_INDEX_SCHEMA:
            raise ValueError("unsupported Stage A execution request index schema")
        indexed_request = _string(raw["request_digest"], field="request_digest")
        if indexed_request != request_digest:
            raise ValueError("Stage A execution request index identity mismatch")
        artifact_digest = _string(raw["artifact_digest"], field="artifact_digest")
        require_sha256(artifact_digest, field="stage_a_execution_artifact_digest")
        relative = _relative_path(raw["artifact_path"], field="artifact_path")
        expected_relative = PurePosixPath(
            "cells", request_digest, f"{artifact_digest}{_CELL_SUFFIX}"
        )
        if relative != expected_relative:
            raise ValueError("Stage A execution request artifact path mismatch")
        payload = {key: value for key, value in raw.items() if key != "digest"}
        digest = _string(raw["digest"], field="digest")
        require_sha256(digest, field="stage_a_execution_request_index.digest")
        if digest != content_digest(payload):
            raise ValueError("Stage A execution request index digest mismatch")
        expected_index_bytes = canonical_json_bytes(raw) + b"\n"
        if index_bytes != expected_index_bytes:
            raise ValueError("Stage A execution request index must use canonical encoding")

        artifact_path = self.root.joinpath(*relative.parts)
        artifact_bytes = _read_regular_bytes(
            artifact_path, field="Stage A execution replay artifact"
        )
        artifact = StageAExecutionReplayArtifact.from_json_bytes(artifact_bytes)
        if artifact.digest != artifact_digest:
            raise ValueError("Stage A execution replay artifact digest mismatch")
        if artifact.cell_identity.request_digest != request_digest:
            raise ValueError("Stage A execution replay request identity mismatch")

        _, event_path, evidence_path, canonical_index = self._paths(artifact)
        if canonical_index != index_path:
            raise ValueError("Stage A execution request index path mismatch")
        event_bytes = _read_regular_bytes(
            event_path, field="Stage A execution event artifact"
        )
        if (
            len(event_bytes) != artifact.event_artifact_size_bytes
            or hashlib.sha256(event_bytes).hexdigest()
            != artifact.event_artifact_digest
        ):
            raise ValueError("Stage A execution event artifact digest mismatch")
        evidence_bytes = _read_regular_bytes(
            evidence_path, field="Stage A execution evidence artifact"
        )
        if (
            len(evidence_bytes) != artifact.execution_evidence_size_bytes
            or hashlib.sha256(evidence_bytes).hexdigest()
            != artifact.execution_evidence_sha256
        ):
            raise ValueError("Stage A execution evidence artifact digest mismatch")
        validate_stage_a_execution_replay_sources(
            artifact,
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes,
        )
        return StoredStageAExecutionReplay(
            root=self.root,
            artifact=artifact,
            artifact_path=artifact_path,
            event_path=event_path,
            evidence_path=evidence_path,
            index_path=index_path,
        )


__all__ = [
    "STAGE_A_EXECUTION_REQUEST_INDEX_SCHEMA",
    "StageAExecutionPromotionStore",
    "StoredStageAExecutionReplay",
]
