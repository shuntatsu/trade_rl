"""Canonical policy identity shared by SB3 training, checkpoints, and serving."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.sequence_architecture import sequence_architecture_identity
from trade_rl.rl.sequence_policy import SequencePolicyArchitecture

SB3_POLICY_IDENTITY_ATTRIBUTE: Final = "_trade_rl_policy_identity"
SB3_POLICY_IDENTITY_SCHEMA: Final = "sb3_policy_identity_v1"


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty string tuple")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must be unique")
    return value


def _validated_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("SB3 policy identity must be a non-empty mapping")
    payload = dict(value)
    if payload.get("schema_version") != SB3_POLICY_IDENTITY_SCHEMA:
        raise ValueError("unsupported SB3 policy identity schema")
    encoder = payload.get("observation_encoder")
    if encoder not in {"flat_mlp", "asset_set", "hierarchical_sequence_v2"}:
        raise ValueError("SB3 policy identity observation encoder is invalid")
    if encoder == "hierarchical_sequence_v2":
        architecture = payload.get("sequence_architecture")
        digest = payload.get("sequence_architecture_digest")
        if not isinstance(architecture, Mapping) or not architecture:
            raise ValueError("sequence architecture identity is missing")
        if not isinstance(digest, str) or digest != content_digest(dict(architecture)):
            raise ValueError("sequence architecture identity digest mismatch")
    elif any(
        key in payload
        for key in ("sequence_architecture", "sequence_architecture_digest")
    ):
        raise ValueError("non-sequence policy cannot declare sequence architecture")
    canonical_json_bytes(payload)
    return payload


def _actual_sequence_architecture(model: object) -> SequencePolicyArchitecture:
    policy = getattr(model, "policy", None)
    extractor = getattr(policy, "features_extractor", None)
    asset_encoder = getattr(extractor, "asset_encoder", None)
    architecture = getattr(asset_encoder, "architecture", None)
    if not isinstance(architecture, SequencePolicyArchitecture):
        raise ValueError(
            "hierarchical sequence model does not expose its validated architecture"
        )
    return architecture


def bind_sb3_policy_identity(model: Any, assembly: object) -> dict[str, object]:
    """Bind the actual assembled policy identity to an SB3 model.

    The hierarchical identity is derived from the constructed feature extractor rather
    than copied from configuration.  This makes configuration/implementation drift
    visible at checkpoint and serving boundaries.
    """

    encoder = getattr(assembly, "observation_encoder", None)
    if encoder not in {"flat_mlp", "asset_set", "hierarchical_sequence_v2"}:
        raise ValueError("SB3 assembly observation encoder is invalid")
    payload: dict[str, object] = {
        "observation_encoder": encoder,
        "schema_version": SB3_POLICY_IDENTITY_SCHEMA,
    }
    if encoder == "hierarchical_sequence_v2":
        symbols = _string_tuple(
            getattr(assembly, "sequence_symbols", None), field="sequence_symbols"
        )
        action_names = _string_tuple(
            getattr(assembly, "sequence_action_names", None),
            field="sequence_action_names",
        )
        identity = sequence_architecture_identity(
            _actual_sequence_architecture(model),
            symbols=symbols,
            action_names=action_names,
        )
        payload["sequence_architecture"] = identity.digest_payload()
        payload["sequence_architecture_digest"] = identity.digest
    resolved = _validated_payload(payload)
    setattr(model, SB3_POLICY_IDENTITY_ATTRIBUTE, resolved)
    return dict(resolved)


def model_sb3_policy_identity(model: object) -> dict[str, object] | None:
    """Read and validate a model-bound policy identity without mutating the model."""

    raw = getattr(model, SB3_POLICY_IDENTITY_ATTRIBUTE, None)
    if raw is None:
        return None
    return _validated_payload(raw)


def validate_model_sb3_policy_identity(
    model: object,
    expected: Mapping[str, object],
) -> None:
    """Fail closed when a loaded model does not match the expected policy identity."""

    expected_payload = _validated_payload(expected)
    observed = model_sb3_policy_identity(model)
    if observed != expected_payload:
        raise ValueError("SB3 policy architecture identity mismatch")


__all__ = [
    "SB3_POLICY_IDENTITY_ATTRIBUTE",
    "SB3_POLICY_IDENTITY_SCHEMA",
    "bind_sb3_policy_identity",
    "model_sb3_policy_identity",
    "validate_model_sb3_policy_identity",
]
