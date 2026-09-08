"""Immutable candidate-run artifact loading and semantic identity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs.provenance import PROVENANCE_SCHEMA

_RESULT_SCHEMA = "lean_candidate_result_v1"
_ARTIFACT_IDENTITY_SCHEMA = "candidate_run_artifact_identity_v1"
_REQUIRED_FILES = frozenset({"summary.json", "returns.npz", "provenance.json"})


@dataclass(frozen=True, slots=True)
class LoadedCandidateRun:
    """Validated semantic content of one published candidate-run artifact."""

    root: Path
    summary: dict[str, object]
    returns: dict[str, np.ndarray]
    provenance: dict[str, object]


@dataclass(frozen=True, slots=True)
class CandidateRunArtifactIdentity:
    """Stable semantic identity plus exact file-level evidence."""

    schema_version: str
    result_schema_version: str
    artifact_digest: str
    summary_file_sha256: str
    summary_file_size: int
    returns_file_sha256: str
    returns_file_size: int
    provenance_file_sha256: str
    provenance_file_size: int


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed candidate {label} JSON") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"candidate {label} must be a JSON object")
    return cast(dict[str, object], raw)


def _validate_root(root: Path) -> tuple[Path, Path, Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate artifact root must be a regular directory")
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise ValueError("candidate artifact root cannot be read") from error
    names = {entry.name for entry in entries}
    if names != _REQUIRED_FILES:
        raise ValueError(
            "candidate artifact root must contain exactly "
            "summary.json, returns.npz, provenance.json"
        )
    resolved: list[Path] = []
    for name in ("summary.json", "returns.npz", "provenance.json"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"candidate artifact {name} must be a regular file, not a symlink")
        resolved.append(path)
    return resolved[0], resolved[1], resolved[2]


def _expected_return_keys(summary: dict[str, object]) -> frozenset[str]:
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list):
        raise ValueError("candidate summary by_symbol must be an array")
    keys: list[str] = []
    for symbol_entry in by_symbol:
        if not isinstance(symbol_entry, dict):
            raise ValueError("candidate summary symbol entry must be an object")
        strategies = symbol_entry.get("strategies")
        if not isinstance(strategies, list):
            raise ValueError("candidate summary strategies must be an array")
        for strategy in strategies:
            if not isinstance(strategy, dict):
                raise ValueError("candidate summary strategy entry must be an object")
            key = strategy.get("return_key")
            if not isinstance(key, str) or not key:
                raise ValueError("candidate summary return_key must be non-empty")
            keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("candidate summary return keys must be unique")
    return frozenset(keys)


def _load_returns(path: Path, *, expected_keys: frozenset[str]) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys != expected_keys:
                raise ValueError("candidate return keys do not match summary")
            for key in sorted(keys):
                try:
                    value = np.asarray(archive[key])
                except ValueError as error:
                    raise ValueError(
                        "candidate return arrays must be numeric and pickle-free"
                    ) from error
                if not np.issubdtype(value.dtype, np.number):
                    raise ValueError("candidate return arrays must be numeric")
                if value.ndim != 1:
                    raise ValueError("candidate return arrays must be one-dimensional")
                if not np.isfinite(value).all():
                    raise ValueError("candidate return arrays must be finite")
                loaded[key] = np.ascontiguousarray(value).copy()
    except (OSError, EOFError, zipfile.BadZipFile) as error:  # type: ignore[name-defined]
        raise ValueError("malformed candidate returns archive") from error
    return loaded


def _file_evidence(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return sha256(data).hexdigest(), len(data)


def _semantic_returns_payload(returns: dict[str, np.ndarray]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for key in sorted(returns):
        value = np.ascontiguousarray(returns[key])
        payload.append(
            {
                "key": key,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": sha256(value.tobytes(order="C")).hexdigest(),
            }
        )
    return payload


def load_candidate_run_artifact(root: str | Path) -> LoadedCandidateRun:
    """Load one exact three-file candidate artifact and validate semantic content."""

    artifact_root = Path(root)
    summary_path, returns_path, provenance_path = _validate_root(artifact_root)
    summary = _read_json_object(summary_path, label="summary")
    if summary.get("schema_version") != _RESULT_SCHEMA:
        raise ValueError("unsupported candidate result schema")
    provenance = _read_json_object(provenance_path, label="provenance")
    if provenance.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError("unsupported candidate provenance schema")
    returns = _load_returns(
        returns_path,
        expected_keys=_expected_return_keys(summary),
    )
    return LoadedCandidateRun(
        root=artifact_root,
        summary=summary,
        returns=returns,
        provenance=provenance,
    )


def inspect_candidate_run_artifact(root: str | Path) -> CandidateRunArtifactIdentity:
    """Return stable semantic identity plus raw file digests/sizes."""

    loaded = load_candidate_run_artifact(root)
    summary_path = loaded.root / "summary.json"
    returns_path = loaded.root / "returns.npz"
    provenance_path = loaded.root / "provenance.json"
    summary_hash, summary_size = _file_evidence(summary_path)
    returns_hash, returns_size = _file_evidence(returns_path)
    provenance_hash, provenance_size = _file_evidence(provenance_path)
    semantic_payload = {
        "schema_version": _ARTIFACT_IDENTITY_SCHEMA,
        "summary": loaded.summary,
        "returns": _semantic_returns_payload(loaded.returns),
        "provenance": loaded.provenance,
    }
    return CandidateRunArtifactIdentity(
        schema_version=_ARTIFACT_IDENTITY_SCHEMA,
        result_schema_version=_RESULT_SCHEMA,
        artifact_digest=content_digest(semantic_payload),
        summary_file_sha256=summary_hash,
        summary_file_size=summary_size,
        returns_file_sha256=returns_hash,
        returns_file_size=returns_size,
        provenance_file_sha256=provenance_hash,
        provenance_file_size=provenance_size,
    )


__all__ = [
    "CandidateRunArtifactIdentity",
    "LoadedCandidateRun",
    "inspect_candidate_run_artifact",
    "load_candidate_run_artifact",
]
