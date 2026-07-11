"""Immutable serving bundle manifest and validation helpers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_MANIFEST_NAME = "manifest.json"
_REQUIRED_JSON_FILES = ("metadata.json", "preprocessing.json", "risk.json")
_SUPPORTED_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_file(root: Path, path: Path) -> str:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes bundle root: {path}") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"invalid bundle path: {relative}")
    return relative.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _validate_finite(value: Any, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite value at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{location}.{key}")
        return
    raise ValueError(f"unsupported JSON value at {location}: {type(value).__name__}")


def compute_bundle_digest(files: Mapping[str, str]) -> str:
    """Return the deterministic digest of a path-to-SHA256 mapping."""
    normalized = dict(sorted(files.items()))
    return _sha256_bytes(_canonical_json(normalized))


@dataclass(frozen=True)
class ServingBundleManifest:
    schema_version: int
    model_version: str
    git_sha: str
    files: dict[str, str]
    bundle_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "git_sha": self.git_sha,
            "files": dict(sorted(self.files.items())),
            "bundle_digest": self.bundle_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ServingBundleManifest":
        files = value.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("manifest files must be a non-empty object")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
            raise ValueError("manifest file paths and digests must be strings")
        return cls(
            schema_version=int(value.get("schema_version", 0)),
            model_version=str(value.get("model_version", "")),
            git_sha=str(value.get("git_sha", "")),
            files=dict(files),
            bundle_digest=str(value.get("bundle_digest", "")),
        )


@dataclass(frozen=True)
class ServingBundle:
    root: Path
    manifest: ServingBundleManifest
    metadata: dict[str, Any]
    preprocessing: dict[str, Any]
    risk: dict[str, Any]

    @property
    def version(self) -> str:
        return self.manifest.model_version

    @property
    def bundle_digest(self) -> str:
        return self.manifest.bundle_digest

    @property
    def model_path(self) -> Path:
        model = self.root / "model.zip"
        if model.is_file():
            return model
        ensemble = self.root / "ensemble"
        if ensemble.is_dir():
            return ensemble
        raise ValueError("bundle contains neither model.zip nor ensemble/")


def _iter_bundle_files(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return sorted(
        (path for path in files if path.name != _MANIFEST_NAME),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def build_manifest(bundle_dir: str | Path) -> ServingBundleManifest:
    """Build and persist a canonical manifest for a candidate bundle."""
    root = Path(bundle_dir)
    if not root.is_dir():
        raise ValueError(f"bundle directory does not exist: {root}")
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("missing required bundle file: metadata.json")
    metadata = _load_json(metadata_path)
    model_version = metadata.get("model_version")
    git_sha = metadata.get("git_sha")
    if not isinstance(model_version, str) or not model_version:
        raise ValueError("metadata.model_version must be a non-empty string")
    if not isinstance(git_sha, str) or not git_sha:
        raise ValueError("metadata.git_sha must be a non-empty string")

    files: dict[str, str] = {}
    for path in _iter_bundle_files(root):
        relative = _safe_relative_file(root, path)
        files[relative] = _sha256_file(path)
    if not files:
        raise ValueError("bundle contains no files")

    manifest = ServingBundleManifest(
        schema_version=_SUPPORTED_SCHEMA_VERSION,
        model_version=model_version,
        git_sha=git_sha,
        files=files,
        bundle_digest=compute_bundle_digest(files),
    )
    (root / _MANIFEST_NAME).write_bytes(_canonical_json(manifest.to_dict()) + b"\n")
    return manifest


def _validate_manifest_paths(root: Path, manifest: ServingBundleManifest) -> None:
    for relative in manifest.files:
        path = Path(relative)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"invalid manifest path: {relative}")
        _safe_relative_file(root, root / path)


def _validate_bundle_schema(
    metadata: dict[str, Any], preprocessing: dict[str, Any], risk: dict[str, Any]
) -> None:
    if metadata.get("schema_version") != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError("unsupported metadata schema_version")
    if metadata.get("observation_schema_version") != 1:
        raise ValueError("unsupported observation_schema_version")
    observation_dim = metadata.get("observation_dim")
    if (
        isinstance(observation_dim, bool)
        or not isinstance(observation_dim, int)
        or observation_dim <= 0
    ):
        raise ValueError("metadata.observation_dim must be a positive integer")
    if metadata.get("observation_progress_mode") != "zero":
        raise ValueError("metadata.observation_progress_mode must be 'zero'")
    symbols = metadata.get("symbols")
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(symbol, str) and symbol for symbol in symbols)
        or len(set(symbols)) != len(symbols)
    ):
        raise ValueError("metadata.symbols must be a unique non-empty string list")

    feature_names = preprocessing.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) and name for name in feature_names)
    ):
        raise ValueError("preprocessing.feature_names must be a non-empty string list")
    global_feature_names = preprocessing.get("global_feature_names")
    if (
        not isinstance(global_feature_names, list)
        or not all(isinstance(name, str) and name for name in global_feature_names)
        or len(set(global_feature_names)) != len(global_feature_names)
    ):
        raise ValueError(
            "preprocessing.global_feature_names must be a unique string list"
        )
    feature_norm = preprocessing.get("feature_norm")
    if feature_norm not in {"none", "rank_gauss"}:
        raise ValueError("preprocessing.feature_norm must be none or rank_gauss")
    feature_mask = preprocessing.get("feature_mask")
    if feature_mask is not None:
        if (
            not isinstance(feature_mask, list)
            or len(feature_mask) != len(feature_names)
            or not all(isinstance(item, bool) for item in feature_mask)
        ):
            raise ValueError("preprocessing.feature_mask must match feature_names")
    expected_post_mask_dim = len(feature_names)
    if preprocessing.get("post_mask_dim") != expected_post_mask_dim:
        raise ValueError(
            "preprocessing.post_mask_dim must preserve the feature dimension"
        )
    run_config = metadata.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("metadata.run_config must be an object")
    include_risk_state = bool(run_config.get("obs_risk_state", False))
    expected_observation_dim = (
        len(symbols) * (len(feature_names) + 1)
        + len(global_feature_names)
        + 3
        + (4 if include_risk_state else 0)
    )
    if observation_dim != expected_observation_dim:
        raise ValueError(
            "metadata.observation_dim does not match the declared observation schema"
        )
    if not isinstance(risk.get("guardrails"), dict) or not isinstance(
        risk.get("pre_trade"), dict
    ):
        raise ValueError("risk.json requires guardrails and pre_trade objects")

    _validate_finite(metadata, "metadata")
    _validate_finite(preprocessing, "preprocessing")
    _validate_finite(risk, "risk")


def load_bundle(bundle_dir: str | Path) -> ServingBundle:
    """Load a bundle only after verifying all files, digests, and schemas."""
    root = Path(bundle_dir)
    manifest_path = root / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError("missing required bundle file: manifest.json")
    manifest = ServingBundleManifest.from_dict(_load_json(manifest_path))
    if manifest.schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema_version")
    if not manifest.model_version or not manifest.git_sha:
        raise ValueError("manifest model_version and git_sha are required")
    _validate_manifest_paths(root, manifest)

    for required in _REQUIRED_JSON_FILES:
        if required not in manifest.files:
            raise ValueError(f"manifest missing required file: {required}")
    if "model.zip" not in manifest.files and not any(
        name.startswith("ensemble/") for name in manifest.files
    ):
        raise ValueError("manifest requires model.zip or ensemble files")

    actual_files = {
        _safe_relative_file(root, path): _sha256_file(path)
        for path in _iter_bundle_files(root)
    }
    if set(actual_files) != set(manifest.files):
        raise ValueError("bundle file set does not match manifest")
    for relative, expected_digest in manifest.files.items():
        if actual_files[relative] != expected_digest:
            raise ValueError(f"digest mismatch for {relative}")
    if compute_bundle_digest(actual_files) != manifest.bundle_digest:
        raise ValueError("bundle digest mismatch")

    metadata = _load_json(root / "metadata.json")
    preprocessing = _load_json(root / "preprocessing.json")
    risk = _load_json(root / "risk.json")
    if metadata.get("model_version") != manifest.model_version:
        raise ValueError("metadata model_version does not match manifest")
    if metadata.get("git_sha") != manifest.git_sha:
        raise ValueError("metadata git_sha does not match manifest")
    _validate_bundle_schema(metadata, preprocessing, risk)

    return ServingBundle(
        root=root,
        manifest=manifest,
        metadata=metadata,
        preprocessing=preprocessing,
        risk=risk,
    )
