"""Canonical, content-addressed training-run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_aware_datetime, require_sha256

RUN_MANIFEST_NAME: Final = "run.json"
RUN_MANIFEST_SCHEMA: Final = "training_run_v1"
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


@dataclass(frozen=True, slots=True)
class TrainingRunManifest:
    digest: str
    run_id: str
    dataset_id: str
    environment_digest: str
    ensemble_digest: str
    training_config_digest: str
    files: tuple[RunFile, ...]
    created_at: datetime
    production_status: str = "NO-GO"
    schema_version: str = RUN_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.digest, field="run.digest")
        if self.run_id in {".", ".."} or not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("run_id contains unsupported characters")
        for field_name, value in (
            ("dataset_id", self.dataset_id),
            ("environment_digest", self.environment_digest),
            ("ensemble_digest", self.ensemble_digest),
            ("training_config_digest", self.training_config_digest),
        ):
            require_sha256(value, field=field_name)
        if not self.files:
            raise ValueError("training run must declare artifact files")
        paths = tuple(item.path for item in self.files)
        if tuple(sorted(paths)) != paths:
            raise ValueError("training run files must use deterministic path ordering")
        if len(set(paths)) != len(paths):
            raise ValueError("training run artifact paths must be unique")
        require_aware_datetime(self.created_at, field="created_at")
        if self.production_status != "NO-GO":
            raise ValueError("unreleased training runs must remain NO-GO")
        if self.schema_version != RUN_MANIFEST_SCHEMA:
            raise ValueError("unsupported training run schema")
        expected = content_digest(self.digest_payload())
        if self.digest != expected:
            raise ValueError("training run digest does not match manifest content")

    def digest_payload(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "ensemble_digest": self.ensemble_digest,
            "environment_digest": self.environment_digest,
            "files": self.files,
            "production_status": self.production_status,
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
        artifact_paths: tuple[str, ...],
        created_at: datetime,
    ) -> TrainingRunManifest:
        files: list[RunFile] = []
        for raw_path in artifact_paths:
            relative = _relative_path(raw_path)
            path = root / relative
            resolved_root = root.resolve()
            resolved_path = path.resolve()
            if resolved_root != resolved_path and resolved_root not in resolved_path.parents:
                raise ValueError("run artifact path escapes its root")
            if not path.is_file():
                raise FileNotFoundError(f"run artifact does not exist: {relative}")
            files.append(
                RunFile(
                    path=relative,
                    digest=_file_digest(path),
                    size_bytes=path.stat().st_size,
                )
            )
        ordered = tuple(sorted(files, key=lambda item: item.path))
        payload = {
            "created_at": created_at,
            "dataset_id": dataset_id,
            "ensemble_digest": ensemble_digest,
            "environment_digest": environment_digest,
            "files": ordered,
            "production_status": "NO-GO",
            "run_id": run_id,
            "schema_version": RUN_MANIFEST_SCHEMA,
            "training_config_digest": training_config_digest,
        }
        return cls(
            digest=content_digest(payload),
            run_id=run_id,
            dataset_id=dataset_id,
            environment_digest=environment_digest,
            ensemble_digest=ensemble_digest,
            training_config_digest=training_config_digest,
            files=ordered,
            created_at=created_at,
        )


def write_training_run_manifest(
    root: Path,
    manifest: TrainingRunManifest,
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


def load_training_run_manifest(root: Path) -> TrainingRunManifest:
    path = root / RUN_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"training run manifest is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = _mapping(raw, field="training run manifest")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("training run files must be a list")
    files: list[RunFile] = []
    for index, raw_file in enumerate(raw_files):
        item = _mapping(raw_file, field=f"files[{index}]")
        files.append(
            RunFile(
                path=_string(item.get("path"), field=f"files[{index}].path"),
                digest=_string(item.get("digest"), field=f"files[{index}].digest"),
                size_bytes=_integer(
                    item.get("size_bytes"), field=f"files[{index}].size_bytes"
                ),
            )
        )
    created_raw = _string(payload.get("created_at"), field="created_at")
    created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
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
        files=tuple(files),
        created_at=created_at,
        production_status=_string(
            payload.get("production_status"), field="production_status"
        ),
        schema_version=_string(payload.get("schema_version"), field="schema_version"),
    )


def validate_training_run_directory(root: Path) -> TrainingRunManifest:
    manifest = load_training_run_manifest(root)
    resolved_root = root.resolve()
    for item in manifest.files:
        path = root / item.path
        resolved_path = path.resolve()
        if resolved_root != resolved_path and resolved_root not in resolved_path.parents:
            raise ValueError(f"run artifact path escapes root: {item.path}")
        if not path.is_file():
            raise ValueError(f"run artifact is missing: {item.path}")
        if path.stat().st_size != item.size_bytes:
            raise ValueError(f"run artifact size mismatch: {item.path}")
        if _file_digest(path) != item.digest:
            raise ValueError(f"run artifact digest mismatch: {item.path}")
    return manifest
