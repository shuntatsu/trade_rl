"""Canonical release-attestation sidecars for serving bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.domain.releases import ReleaseManifest

RELEASE_ATTESTATION_NAME = "release.json"


def write_release_attestation(root: Path, release: ReleaseManifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / RELEASE_ATTESTATION_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(
        canonical_json_bytes(
            {"release_digest": release.digest, **release.digest_payload()}
        )
    )
    temporary.replace(path)
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_release_attestation(root: Path) -> ReleaseManifest:
    path = Path(root) / RELEASE_ATTESTATION_NAME
    if not path.is_file():
        raise ValueError("serving bundle release attestation is missing")
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), field="release")
    selected = raw.get("selected_policy_digest")
    if selected is not None and not isinstance(selected, str):
        raise ValueError("selected_policy_digest must be a string or null")
    release = ReleaseManifest(
        version=_string(raw.get("version"), field="version"),
        git_commit=_string(raw.get("git_commit"), field="git_commit"),
        dataset_id=_string(raw.get("dataset_id"), field="dataset_id"),
        signal_digest=_string(raw.get("signal_digest"), field="signal_digest"),
        selection_digest=_string(raw.get("selection_digest"), field="selection_digest"),
        selection_evaluation_digest=_string(
            raw.get("selection_evaluation_digest"),
            field="selection_evaluation_digest",
        ),
        gate_evaluation_digest=_string(
            raw.get("gate_evaluation_digest"), field="gate_evaluation_digest"
        ),
        selected_policy_digest=selected,
        bundle_digest=_string(raw.get("bundle_digest"), field="bundle_digest"),
        created_at=datetime.fromisoformat(
            _string(raw.get("created_at"), field="created_at").replace("Z", "+00:00")
        ),
        schema_version=_string(raw.get("schema_version"), field="schema_version"),
    )
    declared = _string(raw.get("release_digest"), field="release_digest")
    if release.digest != declared:
        raise ValueError("release attestation digest mismatch")
    return release
