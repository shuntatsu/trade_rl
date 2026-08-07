"""Neutral artifact contract for structured policy exports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.policy_identity_contract import (
    HIERARCHICAL_SEQUENCE_ENCODER,
    SB3_POLICY_IDENTITY_SCHEMA,
    STRUCTURED_TIMEFRAMES,
)
from trade_rl.artifacts.verified_file import (
    file_digest_and_size,
    open_regular_binary,
)
from trade_rl.domain.canonical_json import to_json_value
from trade_rl.domain.common import require_sha256

STRUCTURED_EXPORT_SCHEMA: Final = "structured_policy_export_v2"
STRUCTURED_EXPORT_MANIFEST_NAME: Final = "structured-export.json"
STRUCTURED_EXPORT_MODEL_NAME: Final = "policy.structured.torchscript.pt"
_BASE_KEYS: Final = (
    "current_snapshot",
    "asset_state",
    "global_state",
    "active",
    "current_weights",
)
_SEQUENCE_PLANES: Final = ("values", "available", "staleness")
_SUPPORTED_DTYPES: Final = frozenset(
    {"float16", "float32", "float64", "int32", "int64", "uint8", "bool"}
)


def canonical_structured_observation_keys() -> tuple[str, ...]:
    keys = list(_BASE_KEYS)
    for timeframe in STRUCTURED_TIMEFRAMES:
        for plane in _SEQUENCE_PLANES:
            keys.append(f"sequence_{timeframe}_{plane}")
    return tuple(keys)


def _validated_policy_identity(value: object) -> dict[str, object]:
    normalized = to_json_value(value)
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError("structured export requires policy identity")
    payload = cast(dict[str, object], normalized)
    if payload.get("schema_version") != SB3_POLICY_IDENTITY_SCHEMA:
        raise ValueError("structured export policy identity schema mismatch")
    if payload.get("observation_encoder") != HIERARCHICAL_SEQUENCE_ENCODER:
        raise ValueError("structured export requires hierarchical sequence policy")
    architecture_digest = payload.get("policy_architecture_digest")
    if not isinstance(architecture_digest, str):
        raise ValueError("structured export policy lacks architecture digest")
    require_sha256(
        architecture_digest,
        field="policy_architecture_digest",
    )
    return payload


@dataclass(frozen=True, slots=True)
class StructuredInputSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("structured input name must be non-empty")
        if not self.shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.shape
        ):
            raise ValueError("structured input shape must contain positive integers")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError("structured input dtype is unsupported")


@dataclass(frozen=True, slots=True)
class StructuredExportManifest:
    digest: str
    model_path: str
    model_digest: str
    model_size_bytes: int
    policy_identity: Mapping[str, object]
    policy_identity_digest: str
    architecture_digest: str
    inputs: tuple[StructuredInputSpec, ...]
    action_size: int
    tolerance: float
    max_abs_error: float
    schema_version: str = STRUCTURED_EXPORT_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.digest, field="structured_export.digest")
        require_sha256(
            self.model_digest,
            field="structured_export.model_digest",
        )
        require_sha256(
            self.policy_identity_digest,
            field="structured_export.policy_identity_digest",
        )
        require_sha256(
            self.architecture_digest,
            field="structured_export.architecture_digest",
        )
        if self.schema_version != STRUCTURED_EXPORT_SCHEMA:
            raise ValueError("unsupported structured export schema")
        if self.model_path != STRUCTURED_EXPORT_MODEL_NAME:
            raise ValueError("structured export model path is invalid")
        if (
            isinstance(self.model_size_bytes, bool)
            or not isinstance(self.model_size_bytes, int)
            or self.model_size_bytes <= 0
        ):
            raise ValueError("structured export model must be non-empty")
        policy_payload = _validated_policy_identity(self.policy_identity)
        object.__setattr__(self, "policy_identity", policy_payload)
        if content_digest(policy_payload) != self.policy_identity_digest:
            raise ValueError("structured export policy identity digest mismatch")
        if policy_payload.get("policy_architecture_digest") != self.architecture_digest:
            raise ValueError("structured export architecture digest mismatch")
        if (
            tuple(item.name for item in self.inputs)
            != canonical_structured_observation_keys()
        ):
            raise ValueError("structured export input order is not canonical")
        if (
            isinstance(self.action_size, bool)
            or not isinstance(self.action_size, int)
            or self.action_size <= 0
        ):
            raise ValueError("structured export action_size must be positive")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("structured export tolerance must be finite and positive")
        if (
            not math.isfinite(self.max_abs_error)
            or self.max_abs_error < 0.0
            or self.max_abs_error > self.tolerance
        ):
            raise ValueError("structured export parity error exceeds tolerance")
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("structured export manifest digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_size": self.action_size,
            "architecture_digest": self.architecture_digest,
            "inputs": self.inputs,
            "max_abs_error": self.max_abs_error,
            "model_digest": self.model_digest,
            "model_path": self.model_path,
            "model_size_bytes": self.model_size_bytes,
            "policy_identity": dict(self.policy_identity),
            "policy_identity_digest": self.policy_identity_digest,
            "schema_version": self.schema_version,
            "tolerance": self.tolerance,
        }

    @classmethod
    def build(
        cls,
        *,
        model_path: Path,
        policy_identity: Mapping[str, object],
        inputs: tuple[StructuredInputSpec, ...],
        action_size: int,
        tolerance: float,
        max_abs_error: float,
    ) -> StructuredExportManifest:
        policy_payload = _validated_policy_identity(policy_identity)
        architecture_digest = policy_payload["policy_architecture_digest"]
        assert isinstance(architecture_digest, str)
        model_digest, model_size_bytes = file_digest_and_size(
            model_path,
            field="structured export model",
        )
        policy_identity_digest = content_digest(policy_payload)
        payload = {
            "action_size": action_size,
            "architecture_digest": architecture_digest,
            "inputs": inputs,
            "max_abs_error": max_abs_error,
            "model_digest": model_digest,
            "model_path": STRUCTURED_EXPORT_MODEL_NAME,
            "model_size_bytes": model_size_bytes,
            "policy_identity": policy_payload,
            "policy_identity_digest": policy_identity_digest,
            "schema_version": STRUCTURED_EXPORT_SCHEMA,
            "tolerance": tolerance,
        }
        return cls(
            digest=content_digest(payload),
            model_path=STRUCTURED_EXPORT_MODEL_NAME,
            model_digest=model_digest,
            model_size_bytes=model_size_bytes,
            policy_identity=policy_payload,
            policy_identity_digest=policy_identity_digest,
            architecture_digest=architecture_digest,
            inputs=inputs,
            action_size=action_size,
            tolerance=tolerance,
            max_abs_error=max_abs_error,
            schema_version=STRUCTURED_EXPORT_SCHEMA,
        )


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


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def load_structured_export_manifest_bytes(
    raw: bytes,
) -> StructuredExportManifest:
    """Parse one exact canonical structured-export manifest byte sequence."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("structured export manifest must be valid JSON") from error
    payload = _mapping(value, field="structured export manifest")
    expected = {
        "action_size",
        "architecture_digest",
        "digest",
        "inputs",
        "max_abs_error",
        "model_digest",
        "model_path",
        "model_size_bytes",
        "policy_identity",
        "policy_identity_digest",
        "schema_version",
        "tolerance",
    }
    if set(payload) != expected:
        raise ValueError("structured export manifest fields are invalid")
    raw_inputs = payload["inputs"]
    if not isinstance(raw_inputs, list):
        raise ValueError("structured export inputs must be a list")
    inputs: list[StructuredInputSpec] = []
    for index, raw_input in enumerate(raw_inputs):
        item = _mapping(raw_input, field=f"inputs[{index}]")
        if set(item) != {"dtype", "name", "shape"}:
            raise ValueError("structured input manifest fields are invalid")
        raw_shape = item["shape"]
        if not isinstance(raw_shape, list):
            raise ValueError("structured input shape must be a list")
        inputs.append(
            StructuredInputSpec(
                name=_string(
                    item["name"],
                    field=f"inputs[{index}].name",
                ),
                shape=tuple(
                    _integer(
                        shape_value,
                        field=f"inputs[{index}].shape",
                    )
                    for shape_value in raw_shape
                ),
                dtype=_string(
                    item["dtype"],
                    field=f"inputs[{index}].dtype",
                ),
            )
        )
    policy_identity = _mapping(
        payload["policy_identity"],
        field="policy_identity",
    )
    manifest = StructuredExportManifest(
        digest=_string(payload["digest"], field="digest"),
        model_path=_string(
            payload["model_path"],
            field="model_path",
        ),
        model_digest=_string(
            payload["model_digest"],
            field="model_digest",
        ),
        model_size_bytes=_integer(
            payload["model_size_bytes"],
            field="model_size_bytes",
        ),
        policy_identity=dict(policy_identity),
        policy_identity_digest=_string(
            payload["policy_identity_digest"],
            field="policy_identity_digest",
        ),
        architecture_digest=_string(
            payload["architecture_digest"],
            field="architecture_digest",
        ),
        inputs=tuple(inputs),
        action_size=_integer(
            payload["action_size"],
            field="action_size",
        ),
        tolerance=_number(
            payload["tolerance"],
            field="tolerance",
        ),
        max_abs_error=_number(
            payload["max_abs_error"],
            field="max_abs_error",
        ),
        schema_version=_string(
            payload["schema_version"],
            field="schema_version",
        ),
    )
    if raw != canonical_json_bytes(manifest):
        raise ValueError("structured export manifest must use canonical encoding")
    return manifest


def load_structured_export_manifest(path: Path) -> StructuredExportManifest:
    with open_regular_binary(
        path,
        field="structured export manifest",
    ) as handle:
        return load_structured_export_manifest_bytes(handle.read())


__all__ = [
    "STRUCTURED_EXPORT_MANIFEST_NAME",
    "STRUCTURED_EXPORT_MODEL_NAME",
    "STRUCTURED_EXPORT_SCHEMA",
    "STRUCTURED_TIMEFRAMES",
    "StructuredExportManifest",
    "StructuredInputSpec",
    "canonical_structured_observation_keys",
    "load_structured_export_manifest",
    "load_structured_export_manifest_bytes",
]
