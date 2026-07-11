"""Immutable serving bundle manifest and validation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_MANIFEST_NAME = "manifest.json"
_REQUIRED_JSON_FILES = ("metadata.json", "preprocessing.json", "risk.json")
_SUPPORTED_SCHEMA_VERSION = 1
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,49}$")
_GIT_SHA_RE = re.compile(r"^[a-fA-F0-9]{40}$")
_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
_ENSEMBLE_MEMBER_RE = re.compile(r"^ensemble/seed_[0-9]+\.zip$")
_SUPPORTED_BASE_TIMEFRAMES = {"15m", "1h", "4h", "1d"}


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


def _require_unique_strings(value: Any, location: str, *, non_empty: bool) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    if non_empty and not value:
        raise ValueError(f"{location} must be non-empty")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{location} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{location} must contain unique values")
    return value


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
        schema_version = value.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("manifest schema_version must be an integer")
        files = value.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("manifest files must be a non-empty object")
        if not all(isinstance(k, str) and k for k in files):
            raise ValueError("manifest file paths must be non-empty strings")
        if not all(
            isinstance(v, str) and _SHA256_RE.fullmatch(v) for v in files.values()
        ):
            raise ValueError("manifest file digests must be SHA-256 values")
        bundle_digest = value.get("bundle_digest")
        if not isinstance(bundle_digest, str) or not _SHA256_RE.fullmatch(
            bundle_digest
        ):
            raise ValueError("manifest bundle_digest must be a SHA-256 value")
        return cls(
            schema_version=schema_version,
            model_version=str(value.get("model_version", "")),
            git_sha=str(value.get("git_sha", "")),
            files=dict(files),
            bundle_digest=bundle_digest.lower(),
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
        model_kind = self.metadata.get("model_kind")
        if model_kind == "single":
            model = self.root / "model.zip"
            if model.is_file():
                return model
        elif model_kind == "ensemble":
            ensemble = self.root / "ensemble"
            if ensemble.is_dir():
                return ensemble
        raise ValueError("bundle model_kind does not match its artifact layout")


def _iter_bundle_files(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return sorted(
        (path for path in files if path.name != _MANIFEST_NAME),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _validate_identity(model_version: Any, git_sha: Any, *, location: str) -> None:
    if not isinstance(model_version, str) or not _VERSION_RE.fullmatch(model_version):
        raise ValueError(f"{location}.model_version is invalid")
    if not isinstance(git_sha, str) or not _GIT_SHA_RE.fullmatch(git_sha):
        raise ValueError(f"{location}.git_sha must be a 40-character hexadecimal hash")


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
    _validate_identity(model_version, git_sha, location="metadata")

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


def _validate_model_layout(metadata: dict[str, Any], files: Mapping[str, str]) -> None:
    model_kind = metadata.get("model_kind")
    has_single = "model.zip" in files
    ensemble_members = sorted(
        name for name in files if _ENSEMBLE_MEMBER_RE.fullmatch(name)
    )
    has_other_ensemble_files = any(
        name.startswith("ensemble/") and name not in ensemble_members for name in files
    )
    if model_kind == "single":
        if not has_single or ensemble_members or has_other_ensemble_files:
            raise ValueError("single model_kind requires only model.zip")
    elif model_kind == "ensemble":
        if has_single or not ensemble_members or has_other_ensemble_files:
            raise ValueError(
                "ensemble model_kind requires one or more ensemble/seed_<n>.zip files"
            )
    else:
        raise ValueError("metadata.model_kind must be 'single' or 'ensemble'")


def _validate_bundle_schema(
    metadata: dict[str, Any], preprocessing: dict[str, Any], risk: dict[str, Any]
) -> None:
    if metadata.get("schema_version") != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError("unsupported metadata schema_version")
    _validate_identity(
        metadata.get("model_version"), metadata.get("git_sha"), location="metadata"
    )
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
    symbols = _require_unique_strings(
        metadata.get("symbols"), "metadata.symbols", non_empty=True
    )

    feature_names = _require_unique_strings(
        preprocessing.get("feature_names"),
        "preprocessing.feature_names",
        non_empty=True,
    )
    global_feature_names = _require_unique_strings(
        preprocessing.get("global_feature_names"),
        "preprocessing.global_feature_names",
        non_empty=False,
    )
    feature_norm = preprocessing.get("feature_norm")
    if feature_norm not in {"none", "rank_gauss"}:
        raise ValueError("preprocessing.feature_norm must be none or rank_gauss")
    rank_window = preprocessing.get("rank_window", 250)
    rank_min_periods = preprocessing.get("rank_min_periods", 40)
    if (
        isinstance(rank_window, bool)
        or not isinstance(rank_window, int)
        or isinstance(rank_min_periods, bool)
        or not isinstance(rank_min_periods, int)
        or rank_window <= 0
        or rank_min_periods <= 0
        or rank_min_periods > rank_window
    ):
        raise ValueError("preprocessing rank normalization window settings are invalid")
    feature_mask = preprocessing.get("feature_mask")
    if feature_mask is not None:
        if (
            not isinstance(feature_mask, list)
            or len(feature_mask) != len(feature_names)
            or not all(isinstance(item, bool) for item in feature_mask)
        ):
            raise ValueError("preprocessing.feature_mask must match feature_names")
    if preprocessing.get("post_mask_dim") != len(feature_names):
        raise ValueError(
            "preprocessing.post_mask_dim must preserve the feature dimension"
        )

    run_config = metadata.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("metadata.run_config must be an object")
    if run_config.get("observation_progress_mode", "zero") != "zero":
        raise ValueError("run_config observation_progress_mode must be 'zero'")
    obs_risk_state = run_config.get("obs_risk_state", False)
    if not isinstance(obs_risk_state, bool):
        raise ValueError("run_config.obs_risk_state must be a boolean")
    base_timeframe = run_config.get("base_timeframe", "1h")
    if base_timeframe not in _SUPPORTED_BASE_TIMEFRAMES:
        raise ValueError("run_config.base_timeframe is unsupported")
    disagreement_dr = run_config.get("disagreement_dr_max", 0.0)
    if (
        isinstance(disagreement_dr, bool)
        or not isinstance(disagreement_dr, (int, float))
        or not math.isfinite(float(disagreement_dr))
        or not 0.0 <= float(disagreement_dr) <= 1.0
    ):
        raise ValueError("run_config.disagreement_dr_max must be between 0 and 1")

    expected_observation_dim = (
        len(symbols) * (len(feature_names) + 1)
        + len(global_feature_names)
        + 3
        + (4 if obs_risk_state else 0)
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
    _validate_identity(manifest.model_version, manifest.git_sha, location="manifest")
    _validate_manifest_paths(root, manifest)

    for required in _REQUIRED_JSON_FILES:
        if required not in manifest.files:
            raise ValueError(f"manifest missing required file: {required}")

    actual_files = {
        _safe_relative_file(root, path): _sha256_file(path)
        for path in _iter_bundle_files(root)
    }
    if set(actual_files) != set(manifest.files):
        raise ValueError("bundle file set does not match manifest")
    for relative, expected_digest in manifest.files.items():
        if actual_files[relative] != expected_digest.lower():
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
    _validate_model_layout(metadata, actual_files)

    return ServingBundle(
        root=root,
        manifest=manifest,
        metadata=metadata,
        preprocessing=preprocessing,
        risk=risk,
    )
