"""Canonical, content-addressed manifests for training and walk-forward runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Final, TypeVar

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_aware_datetime, require_sha256

RUN_MANIFEST_NAME: Final = "run.json"
TRAINING_RUN_MANIFEST_SCHEMA: Final = "training_run_v2"
WALK_FORWARD_RUN_MANIFEST_SCHEMA: Final = "walk_forward_run_v1"
RUN_MANIFEST_SCHEMA: Final = TRAINING_RUN_MANIFEST_SCHEMA
_RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise ValueError("run artifact path must be relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("run artifact path contains an unsafe segment")
    normalized = path.as_posix()
    if normalized == RUN_MANIFEST_NAME:
        raise ValueError("run manifest cannot include itself")
    return normalized


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class RunFile:
    path: str
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        require_sha256(self.digest, field="run_file.digest")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("run file size_bytes must be a non-negative integer")


def _validate_common(
    *,
    digest: str,
    run_id: str,
    dataset_id: str,
    environment_digest: str,
    provenance_digest: str,
    files: tuple[RunFile, ...],
    created_at: datetime,
    production_status: str,
) -> None:
    require_sha256(digest, field="run.digest")
    if run_id in {".", ".."} or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    for field_name, value in (
        ("dataset_id", dataset_id),
        ("environment_digest", environment_digest),
        ("provenance_digest", provenance_digest),
    ):
        require_sha256(value, field=field_name)
    if not files:
        raise ValueError("run must declare artifact files")
    paths = tuple(item.path for item in files)
    if tuple(sorted(paths)) != paths:
        raise ValueError("run files must use deterministic path ordering")
    if len(set(paths)) != len(paths):
        raise ValueError("run artifact paths must be unique")
    require_aware_datetime(created_at, field="created_at")
    if production_status != "NO-GO":
        raise ValueError("unreleased runs must remain NO-GO")


def _build_files(root: Path, artifact_paths: tuple[str, ...]) -> tuple[RunFile, ...]:
    files: list[RunFile] = []
    resolved_root = root.resolve()
    for raw_path in artifact_paths:
        relative = _relative_path(raw_path)
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"run artifact must not be a symlink: {relative}")
        resolved_path = path.resolve()
        if (
            resolved_root != resolved_path
            and resolved_root not in resolved_path.parents
        ):
            raise ValueError("run artifact path escapes its root")
        if not path.is_file():
            raise FileNotFoundError(f"run artifact does not exist: {relative}")
        files.append(RunFile(relative, _file_digest(path), path.stat().st_size))
    return tuple(sorted(files, key=lambda item: item.path))


@dataclass(frozen=True, slots=True)
class TrainingRunManifest:
    digest: str
    run_id: str
    dataset_id: str
    environment_digest: str
    ensemble_digest: str
    training_config_digest: str
    provenance_digest: str
    files: tuple[RunFile, ...]
    created_at: datetime
    production_status: str = "NO-GO"
    schema_version: str = TRAINING_RUN_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        _validate_common(
            digest=self.digest,
            run_id=self.run_id,
            dataset_id=self.dataset_id,
            environment_digest=self.environment_digest,
            provenance_digest=self.provenance_digest,
            files=self.files,
            created_at=self.created_at,
            production_status=self.production_status,
        )
        require_sha256(self.ensemble_digest, field="ensemble_digest")
        require_sha256(self.training_config_digest, field="training_config_digest")
        if self.schema_version != TRAINING_RUN_MANIFEST_SCHEMA:
            raise ValueError("unsupported training run schema")
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("training run digest does not match manifest content")

    def digest_payload(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "ensemble_digest": self.ensemble_digest,
            "environment_digest": self.environment_digest,
            "files": self.files,
            "production_status": self.production_status,
            "provenance_digest": self.provenance_digest,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "training_config_digest": self.training_config_digest,
        }

    @classmethod
    def build(
        cls,
        *,
        root: Path,
        run_id: str,
        dataset_id: str,
        environment_digest: str,
        ensemble_digest: str,
        training_config_digest: str,
        provenance_digest: str,
        artifact_paths: tuple[str, ...],
        created_at: datetime,
    ) -> TrainingRunManifest:
        files = _build_files(root, artifact_paths)
        payload = {
            "created_at": created_at,
            "dataset_id": dataset_id,
            "ensemble_digest": ensemble_digest,
            "environment_digest": environment_digest,
            "files": files,
            "production_status": "NO-GO",
            "provenance_digest": provenance_digest,
            "run_id": run_id,
            "schema_version": TRAINING_RUN_MANIFEST_SCHEMA,
            "training_config_digest": training_config_digest,
        }
        return cls(
            digest=content_digest(payload),
            run_id=run_id,
            dataset_id=dataset_id,
            environment_digest=environment_digest,
            ensemble_digest=ensemble_digest,
            training_config_digest=training_config_digest,
            provenance_digest=provenance_digest,
            files=files,
            created_at=created_at,
        )


@dataclass(frozen=True, slots=True)
class WalkForwardRunManifest:
    digest: str
    run_id: str
    dataset_id: str
    environment_digest: str
    evaluation_digest: str
    workflow_config_digest: str
    policy_set_digest: str
    provenance_digest: str
    fold_count: int
    files: tuple[RunFile, ...]
    created_at: datetime
    production_status: str = "NO-GO"
    schema_version: str = WALK_FORWARD_RUN_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        _validate_common(
            digest=self.digest,
            run_id=self.run_id,
            dataset_id=self.dataset_id,
            environment_digest=self.environment_digest,
            provenance_digest=self.provenance_digest,
            files=self.files,
            created_at=self.created_at,
            production_status=self.production_status,
        )
        for field_name, value in (
            ("evaluation_digest", self.evaluation_digest),
            ("workflow_config_digest", self.workflow_config_digest),
            ("policy_set_digest", self.policy_set_digest),
        ):
            require_sha256(value, field=field_name)
        if (
            isinstance(self.fold_count, bool)
            or not isinstance(self.fold_count, int)
            or self.fold_count <= 0
        ):
            raise ValueError("fold_count must be a positive integer")
        if self.schema_version != WALK_FORWARD_RUN_MANIFEST_SCHEMA:
            raise ValueError("unsupported walk-forward run schema")
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("walk-forward run digest does not match manifest content")

    def digest_payload(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "environment_digest": self.environment_digest,
            "evaluation_digest": self.evaluation_digest,
            "files": self.files,
            "fold_count": self.fold_count,
            "policy_set_digest": self.policy_set_digest,
            "production_status": self.production_status,
            "provenance_digest": self.provenance_digest,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "workflow_config_digest": self.workflow_config_digest,
        }

    @classmethod
    def build(
        cls,
        *,
        root: Path,
        run_id: str,
        dataset_id: str,
        environment_digest: str,
        evaluation_digest: str,
        workflow_config_digest: str,
        policy_set_digest: str,
        provenance_digest: str,
        fold_count: int,
        artifact_paths: tuple[str, ...],
        created_at: datetime,
    ) -> WalkForwardRunManifest:
        files = _build_files(root, artifact_paths)
        payload = {
            "created_at": created_at,
            "dataset_id": dataset_id,
            "environment_digest": environment_digest,
            "evaluation_digest": evaluation_digest,
            "files": files,
            "fold_count": fold_count,
            "policy_set_digest": policy_set_digest,
            "production_status": "NO-GO",
            "provenance_digest": provenance_digest,
            "run_id": run_id,
            "schema_version": WALK_FORWARD_RUN_MANIFEST_SCHEMA,
            "workflow_config_digest": workflow_config_digest,
        }
        return cls(
            digest=content_digest(payload),
            run_id=run_id,
            dataset_id=dataset_id,
            environment_digest=environment_digest,
            evaluation_digest=evaluation_digest,
            workflow_config_digest=workflow_config_digest,
            policy_set_digest=policy_set_digest,
            provenance_digest=provenance_digest,
            fold_count=fold_count,
            files=files,
            created_at=created_at,
        )


def write_training_run_manifest(root: Path, manifest: TrainingRunManifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / RUN_MANIFEST_NAME
    _atomic_write(path, canonical_json_bytes(manifest))
    return path


def write_walk_forward_run_manifest(
    root: Path, manifest: WalkForwardRunManifest
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / RUN_MANIFEST_NAME
    _atomic_write(path, canonical_json_bytes(manifest))
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _read_payload(root: Path) -> Mapping[str, object]:
    path = root / RUN_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"run manifest is missing: {path}")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), field="run manifest")


def _parse_files(payload: Mapping[str, object]) -> tuple[RunFile, ...]:
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("run files must be a list")
    return tuple(
        RunFile(
            path=_string(
                _mapping(raw, field=f"files[{index}]").get("path"),
                field=f"files[{index}].path",
            ),
            digest=_string(
                _mapping(raw, field=f"files[{index}]").get("digest"),
                field=f"files[{index}].digest",
            ),
            size_bytes=_integer(
                _mapping(raw, field=f"files[{index}]").get("size_bytes"),
                field=f"files[{index}].size_bytes",
            ),
        )
        for index, raw in enumerate(raw_files)
    )


def load_training_run_manifest(root: Path) -> TrainingRunManifest:
    payload = _read_payload(root)
    schema = _string(payload.get("schema_version"), field="schema_version")
    if schema != TRAINING_RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported training run schema")
    created_at = datetime.fromisoformat(
        _string(payload.get("created_at"), field="created_at").replace("Z", "+00:00")
    )
    return TrainingRunManifest(
        digest=_string(payload.get("digest"), field="digest"),
        run_id=_string(payload.get("run_id"), field="run_id"),
        dataset_id=_string(payload.get("dataset_id"), field="dataset_id"),
        environment_digest=_string(
            payload.get("environment_digest"), field="environment_digest"
        ),
        ensemble_digest=_string(
            payload.get("ensemble_digest"), field="ensemble_digest"
        ),
        training_config_digest=_string(
            payload.get("training_config_digest"), field="training_config_digest"
        ),
        provenance_digest=_string(
            payload.get("provenance_digest"), field="provenance_digest"
        ),
        files=_parse_files(payload),
        created_at=created_at,
        production_status=_string(
            payload.get("production_status"), field="production_status"
        ),
        schema_version=schema,
    )


def load_walk_forward_run_manifest(root: Path) -> WalkForwardRunManifest:
    payload = _read_payload(root)
    schema = _string(payload.get("schema_version"), field="schema_version")
    if schema != WALK_FORWARD_RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported walk-forward run schema")
    created_at = datetime.fromisoformat(
        _string(payload.get("created_at"), field="created_at").replace("Z", "+00:00")
    )
    return WalkForwardRunManifest(
        digest=_string(payload.get("digest"), field="digest"),
        run_id=_string(payload.get("run_id"), field="run_id"),
        dataset_id=_string(payload.get("dataset_id"), field="dataset_id"),
        environment_digest=_string(
            payload.get("environment_digest"), field="environment_digest"
        ),
        evaluation_digest=_string(
            payload.get("evaluation_digest"), field="evaluation_digest"
        ),
        workflow_config_digest=_string(
            payload.get("workflow_config_digest"), field="workflow_config_digest"
        ),
        policy_set_digest=_string(
            payload.get("policy_set_digest"), field="policy_set_digest"
        ),
        provenance_digest=_string(
            payload.get("provenance_digest"), field="provenance_digest"
        ),
        fold_count=_integer(payload.get("fold_count"), field="fold_count"),
        files=_parse_files(payload),
        created_at=created_at,
        production_status=_string(
            payload.get("production_status"), field="production_status"
        ),
        schema_version=schema,
    )


ManifestT = TypeVar("ManifestT", TrainingRunManifest, WalkForwardRunManifest)


def _validate_directory(root: Path, manifest: ManifestT) -> ManifestT:
    declared = {item.path for item in manifest.files}
    actual: set[str] = set()
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"run artifact must not be a symlink: {path.relative_to(root)}"
            )
        if path.is_file() and path.name != RUN_MANIFEST_NAME:
            actual.add(path.relative_to(root).as_posix())
    undeclared = actual - declared
    missing = declared - actual
    if undeclared:
        raise ValueError(
            f"run directory contains undeclared files: {sorted(undeclared)}"
        )
    if missing:
        raise ValueError(f"run artifact is missing: {sorted(missing)}")
    for item in manifest.files:
        path = root / item.path
        resolved_path = path.resolve()
        if (
            resolved_root != resolved_path
            and resolved_root not in resolved_path.parents
        ):
            raise ValueError(f"run artifact path escapes root: {item.path}")
        if path.stat().st_size != item.size_bytes:
            raise ValueError(f"run artifact size mismatch: {item.path}")
        if _file_digest(path) != item.digest:
            raise ValueError(f"run artifact digest mismatch: {item.path}")
    return manifest


def validate_training_run_directory(root: Path) -> TrainingRunManifest:
    return _validate_directory(root, load_training_run_manifest(root))


def validate_walk_forward_run_directory(root: Path) -> WalkForwardRunManifest:
    return _validate_directory(root, load_walk_forward_run_manifest(root))


__all__ = [
    "RUN_MANIFEST_NAME",
    "RUN_MANIFEST_SCHEMA",
    "RunFile",
    "TrainingRunManifest",
    "WalkForwardRunManifest",
    "load_training_run_manifest",
    "load_walk_forward_run_manifest",
    "validate_training_run_directory",
    "validate_walk_forward_run_directory",
    "write_training_run_manifest",
    "write_walk_forward_run_manifest",
]
