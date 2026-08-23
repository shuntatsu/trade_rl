"""Immutable/restart-safe evidence storage for the research-only V4 lane."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

CAUSAL_ALPHA_V4_STORE_SCHEMA: Final = "causal_alpha_v4_artifact_store_v1"


def _relative_artifact_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("V4 relative artifact path is invalid")
    return path


def _strict_payload(raw: object, *, expected_schema: str) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("V4 artifact payload must be an object")
    payload = {str(key): value for key, value in raw.items()}
    if payload.get("schema_version") != expected_schema:
        raise ValueError("V4 artifact schema mismatch")
    artifact_digest = payload.get("artifact_digest")
    if not isinstance(artifact_digest, str):
        raise ValueError("V4 artifact digest is missing")
    require_sha256(artifact_digest, field="V4 artifact digest")
    digest_payload = {key: value for key, value in payload.items() if key != "artifact_digest"}
    if content_digest(digest_payload) != artifact_digest:
        raise ValueError("V4 artifact digest mismatch")
    return payload


class CausalAlphaV4ArtifactStore:
    """Write immutable JSON leaves and validate every identity on resume."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_manifest_digest: str,
        v4_context_manifest_digest: str,
        config_digest: str,
        generator_code_digest: str,
    ) -> None:
        for field_name, value in (
            ("run_manifest_digest", run_manifest_digest),
            ("v4_context_manifest_digest", v4_context_manifest_digest),
            ("config_digest", config_digest),
            ("generator_code_digest", generator_code_digest),
        ):
            require_sha256(value, field=f"V4 store {field_name}")
        self.root = Path(root)
        self.run_manifest_digest = run_manifest_digest
        self.v4_context_manifest_digest = v4_context_manifest_digest
        self.config_digest = config_digest
        self.generator_code_digest = generator_code_digest

    def _validate_identity(self, payload: Mapping[str, object]) -> None:
        expected = (
            ("run_manifest_digest", self.run_manifest_digest, "run manifest"),
            (
                "v4_context_manifest_digest",
                self.v4_context_manifest_digest,
                "V4 context manifest",
            ),
            ("config_digest", self.config_digest, "config"),
            ("generator_code_digest", self.generator_code_digest, "generator code"),
        )
        for field_name, expected_value, label in expected:
            observed = payload.get(field_name)
            if observed is None:
                continue
            if observed != expected_value:
                raise ValueError(f"V4 artifact {label} identity mismatch")
        for field_name in (
            "contract_digest",
            "fit_digest",
            "forecast_digest",
            "target_path_digest",
        ):
            value = payload.get(field_name)
            if value is not None:
                require_sha256(value, field=f"V4 artifact {field_name}")

    def write_leaf(self, relative_path: str | Path, payload: Mapping[str, object]) -> Path:
        relative = _relative_artifact_path(relative_path)
        values = {str(key): value for key, value in payload.items()}
        schema = values.get("schema_version")
        if not isinstance(schema, str) or not schema:
            raise ValueError("V4 artifact schema is missing")
        validated = _strict_payload(values, expected_schema=schema)
        self._validate_identity(validated)
        output = self.root / relative
        encoded = canonical_json_bytes(validated) + b"\n"
        if output.exists():
            if output.is_file() and output.read_bytes() == encoded:
                return output
            raise FileExistsError(f"V4 artifact already exists with different content: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(output)
        return output

    def load_leaf(
        self,
        relative_path: str | Path,
        *,
        expected_schema: str,
    ) -> dict[str, object] | None:
        relative = _relative_artifact_path(relative_path)
        path = self.root / relative
        if not path.exists():
            return None
        if not path.is_file():
            raise ValueError("V4 artifact leaf is not a file")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("V4 artifact leaf is invalid JSON") from error
        payload = _strict_payload(raw, expected_schema=expected_schema)
        self._validate_identity(payload)
        return payload

    def envelope(
        self,
        *,
        schema_version: str,
        evidence_digest: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        require_sha256(evidence_digest, field="V4 store evidence_digest")
        body: dict[str, object] = {
            "schema_version": schema_version,
            "run_manifest_digest": self.run_manifest_digest,
            "v4_context_manifest_digest": self.v4_context_manifest_digest,
            "config_digest": self.config_digest,
            "generator_code_digest": self.generator_code_digest,
            "evidence_digest": evidence_digest,
            "payload": dict(payload),
        }
        return {**body, "artifact_digest": content_digest(body)}


__all__ = ["CAUSAL_ALPHA_V4_STORE_SCHEMA", "CausalAlphaV4ArtifactStore"]
