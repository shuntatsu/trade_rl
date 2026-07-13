"""Immutable serving bundles with complete content-digest validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)
from trade_rl.domain.selection import PolicyMode

BUNDLE_MANIFEST_NAME = "bundle.json"


def _relative_path(value: str) -> str:
    normalized = PurePosixPath(value)
    if normalized.is_absolute() or not normalized.parts:
        raise ValueError("bundle artifact path must be relative")
    if any(part in {"", ".", ".."} for part in normalized.parts):
        raise ValueError("bundle artifact path contains an unsafe segment")
    path = normalized.as_posix()
    if path == BUNDLE_MANIFEST_NAME:
        raise ValueError("bundle manifest cannot include itself as an artifact")
    return path


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BundleFile:
    path: str
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        require_sha256(self.digest, field="file.digest")
        if self.size_bytes < 0:
            raise ValueError("file size_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class ServingBundleManifest:
    bundle_digest: str
    dataset_id: str
    action_schema: str
    observation_schema_digest: str
    observation_size: int
    market_inputs_digest: str
    policy_mode: PolicyMode
    policy_digest: str | None
    signal_digest: str
    selection_digest: str
    release_digest: str | None
    files: tuple[BundleFile, ...]
    created_at: datetime
    schema_version: str = "serving_bundle_v3"

    def __post_init__(self) -> None:
        require_sha256(self.bundle_digest, field="bundle_digest")
        require_sha256(self.dataset_id, field="dataset_id")
        require_non_empty(self.action_schema, field="action_schema")
        require_sha256(
            self.observation_schema_digest,
            field="observation_schema_digest",
        )
        if self.observation_size <= 0:
            raise ValueError("observation_size must be positive")
        require_sha256(self.market_inputs_digest, field="market_inputs_digest")
        require_sha256(self.signal_digest, field="signal_digest")
        require_sha256(self.selection_digest, field="selection_digest")
        if self.policy_mode is PolicyMode.BASELINE_ONLY:
            if self.policy_digest is not None:
                raise ValueError("baseline_only bundle cannot contain a policy digest")
        elif self.policy_digest is None:
            raise ValueError("residual policy bundle requires a policy digest")
        else:
            require_sha256(self.policy_digest, field="policy_digest")
        if self.release_digest is not None:
            require_sha256(self.release_digest, field="release_digest")
        if not self.files:
            raise ValueError("serving bundle must contain artifact files")
        paths = tuple(file.path for file in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("serving bundle artifact paths must be unique")
        require_aware_datetime(self.created_at, field="created_at")
        require_non_empty(self.schema_version, field="schema_version")
        expected = content_digest(self.digest_payload())
        if self.bundle_digest != expected:
            raise ValueError("bundle digest does not match manifest content")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_schema": self.action_schema,
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "files": self.files,
            "observation_schema_digest": self.observation_schema_digest,
            "observation_size": self.observation_size,
            "market_inputs_digest": self.market_inputs_digest,
            "policy_digest": self.policy_digest,
            "policy_mode": self.policy_mode,
            "release_digest": self.release_digest,
            "schema_version": self.schema_version,
            "selection_digest": self.selection_digest,
            "signal_digest": self.signal_digest,
        }

    @classmethod
    def build(
        cls,
        *,
        root: Path,
        dataset_id: str,
        action_schema: str,
        observation_schema_digest: str,
        observation_size: int,
        market_inputs_digest: str,
        policy_mode: PolicyMode,
        policy_digest: str | None,
        signal_digest: str,
        selection_digest: str,
        release_digest: str | None,
        artifact_paths: tuple[str, ...],
        created_at: datetime,
    ) -> ServingBundleManifest:
        files: list[BundleFile] = []
        for raw_path in artifact_paths:
            relative = _relative_path(raw_path)
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"bundle artifact does not exist: {relative}")
            files.append(
                BundleFile(
                    path=relative,
                    digest=_file_digest(path),
                    size_bytes=path.stat().st_size,
                )
            )
        ordered = tuple(sorted(files, key=lambda item: item.path))
        payload = {
            "action_schema": action_schema,
            "created_at": created_at,
            "dataset_id": dataset_id,
            "files": ordered,
            "observation_schema_digest": observation_schema_digest,
            "observation_size": observation_size,
            "market_inputs_digest": market_inputs_digest,
            "policy_digest": policy_digest,
            "policy_mode": policy_mode,
            "release_digest": release_digest,
            "schema_version": "serving_bundle_v3",
            "selection_digest": selection_digest,
            "signal_digest": signal_digest,
        }
        return cls(
            bundle_digest=content_digest(payload),
            dataset_id=dataset_id,
            action_schema=action_schema,
            observation_schema_digest=observation_schema_digest,
            observation_size=observation_size,
            market_inputs_digest=market_inputs_digest,
            policy_mode=policy_mode,
            policy_digest=policy_digest,
            signal_digest=signal_digest,
            selection_digest=selection_digest,
            release_digest=release_digest,
            files=ordered,
            created_at=created_at,
        )


@dataclass(frozen=True, slots=True)
class ServingBundle:
    root: Path
    manifest: ServingBundleManifest


def write_serving_bundle_manifest(
    root: Path,
    manifest: ServingBundleManifest,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / BUNDLE_MANIFEST_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(manifest))
    temporary.replace(path)
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _parse_manifest(payload: Mapping[str, object]) -> ServingBundleManifest:
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("files must be a list")
    files: list[BundleFile] = []
    for index, raw_file in enumerate(raw_files):
        item = _mapping(raw_file, field=f"files[{index}]")
        files.append(
            BundleFile(
                path=_string(item.get("path"), field=f"files[{index}].path"),
                digest=_string(item.get("digest"), field=f"files[{index}].digest"),
                size_bytes=_integer(
                    item.get("size_bytes"),
                    field=f"files[{index}].size_bytes",
                ),
            )
        )
    created_at_raw = _string(payload.get("created_at"), field="created_at")
    created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
    return ServingBundleManifest(
        bundle_digest=_string(payload.get("bundle_digest"), field="bundle_digest"),
        dataset_id=_string(payload.get("dataset_id"), field="dataset_id"),
        action_schema=_string(payload.get("action_schema"), field="action_schema"),
        observation_schema_digest=_string(
            payload.get("observation_schema_digest"),
            field="observation_schema_digest",
        ),
        observation_size=_integer(
            payload.get("observation_size"),
            field="observation_size",
        ),
        market_inputs_digest=_string(
            payload.get("market_inputs_digest"),
            field="market_inputs_digest",
        ),
        policy_mode=PolicyMode(
            _string(payload.get("policy_mode"), field="policy_mode")
        ),
        policy_digest=_optional_string(
            payload.get("policy_digest"),
            field="policy_digest",
        ),
        signal_digest=_string(payload.get("signal_digest"), field="signal_digest"),
        selection_digest=_string(
            payload.get("selection_digest"),
            field="selection_digest",
        ),
        release_digest=_optional_string(
            payload.get("release_digest"),
            field="release_digest",
        ),
        files=tuple(files),
        created_at=created_at,
        schema_version=_string(payload.get("schema_version"), field="schema_version"),
    )


def load_serving_bundle(root: Path) -> ServingBundle:
    manifest_path = root / BUNDLE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"serving bundle manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _parse_manifest(_mapping(payload, field="bundle manifest"))
    for file in manifest.files:
        path = root / file.path
        if not path.is_file():
            raise ValueError(f"bundle artifact is missing: {file.path}")
        if path.stat().st_size != file.size_bytes:
            raise ValueError(f"bundle artifact size mismatch: {file.path}")
        if _file_digest(path) != file.digest:
            raise ValueError(f"bundle artifact digest mismatch: {file.path}")
    return ServingBundle(root=root, manifest=manifest)
