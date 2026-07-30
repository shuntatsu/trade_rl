"""Canonical policy identity shared by SB3 training, checkpoints, and serving."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.observations import CURRENT_WEIGHT_SOURCE
from trade_rl.rl.sequence_architecture import sequence_architecture_identity
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.rl.sequence_policy import SequencePolicyArchitecture

SB3_POLICY_IDENTITY_ATTRIBUTE: Final = "_trade_rl_policy_identity"
SB3_POLICY_IDENTITY_SCHEMA: Final = "sb3_policy_identity_v3"
LEGACY_SB3_POLICY_IDENTITY_SCHEMA: Final = "sb3_policy_identity_v2"
POLICY_ARCHITECTURE_SCHEMA: Final = "hierarchical_gate_target_policy_v2"
HIERARCHICAL_EXPLORATION_SCHEMA: Final = "hierarchical_exploration_v1"
HIERARCHICAL_ACTION_DISTRIBUTION: Final = (
    "masked_shared_squashed_diag_gaussian_v1"
)
HIERARCHICAL_EXPLORATION_COUPLING: Final = (
    "post_composition_gate_independent_v1"
)
HIERARCHICAL_LOG_STD_PARAMETERIZATION: Final = "shared_scalar_v1"
HIERARCHICAL_ACTOR_HEAD: Final = "hierarchical_gate_target_v1"
CURRENT_WEIGHT_KEY: Final = "current_weights"
_INTERNAL_ACTION_DISTRIBUTION_NAMES: Final = frozenset(
    {
        "masked_shared_squashed_diag_gaussian",
        HIERARCHICAL_ACTION_DISTRIBUTION,
    }
)


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


def _positive_temperature(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def current_weight_observation_identity(n_symbols: int) -> dict[str, object]:
    if isinstance(n_symbols, bool) or not isinstance(n_symbols, int) or n_symbols <= 0:
        raise ValueError("current-weight symbol count must be a positive integer")
    return {
        "bounds": (-1.0, 1.0),
        "dtype": "float32",
        "key": CURRENT_WEIGHT_KEY,
        "observation_schema": SEQUENCE_OBSERVATION_SCHEMA,
        "shape": (n_symbols,),
        "source": CURRENT_WEIGHT_SOURCE,
    }


def _validated_current_weight_identity(
    value: object, *, n_symbols: int
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("current-weight observation identity is missing")
    observed = dict(value)
    expected = current_weight_observation_identity(n_symbols)
    normalized = {
        **observed,
        "bounds": tuple(observed.get("bounds", ())),
        "shape": tuple(observed.get("shape", ())),
    }
    if normalized != expected:
        raise ValueError("current-weight observation identity mismatch")
    return expected


def _hierarchical_exploration_payload() -> dict[str, object]:
    return {
        "action_distribution": HIERARCHICAL_ACTION_DISTRIBUTION,
        "change_intensity_coupling": HIERARCHICAL_EXPLORATION_COUPLING,
        "log_std_parameterization": HIERARCHICAL_LOG_STD_PARAMETERIZATION,
        "state_dependent_noise": False,
        "schema_version": HIERARCHICAL_EXPLORATION_SCHEMA,
        "squashing": "tanh",
    }


def _validated_hierarchical_exploration(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("hierarchical exploration identity is missing")
    observed = dict(value)
    expected = _hierarchical_exploration_payload()
    if observed != expected:
        raise ValueError("hierarchical exploration identity mismatch")
    return expected


def _actual_hierarchical_exploration(policy: object) -> dict[str, object]:
    distribution = getattr(policy, "action_distribution_name", None)
    if distribution not in _INTERNAL_ACTION_DISTRIBUTION_NAMES:
        raise ValueError("hierarchical action distribution identity mismatch")
    log_std = getattr(policy, "log_std", None)
    shape = tuple(getattr(log_std, "shape", ()))
    if shape != (1,):
        raise ValueError("hierarchical actor requires shared scalar log_std")
    if getattr(policy, "use_sde", None) is not False:
        raise ValueError("hierarchical actor does not support gSDE")
    return _hierarchical_exploration_payload()


def _policy_architecture_payload(
    *,
    actor_head: str,
    gate_temperature: float,
    sequence_architecture_digest: str,
    current_weight_observation: Mapping[str, object],
    exploration_contract: Mapping[str, object],
) -> dict[str, object]:
    return {
        "actor_head": actor_head,
        "current_weight_observation": dict(current_weight_observation),
        "exploration_contract": dict(exploration_contract),
        "gate_temperature": gate_temperature,
        "observation_encoder": "hierarchical_sequence_v2",
        "schema_version": POLICY_ARCHITECTURE_SCHEMA,
        "sequence_architecture_digest": sequence_architecture_digest,
    }


def _validated_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("SB3 policy identity must be a non-empty mapping")
    payload = dict(value)
    schema = payload.get("schema_version")
    if schema == LEGACY_SB3_POLICY_IDENTITY_SCHEMA:
        raise ValueError(
            f"migrate {LEGACY_SB3_POLICY_IDENTITY_SCHEMA} to "
            f"{SB3_POLICY_IDENTITY_SCHEMA}"
        )
    if schema != SB3_POLICY_IDENTITY_SCHEMA:
        raise ValueError("unsupported SB3 policy identity schema")
    encoder = payload.get("observation_encoder")
    if encoder not in {"flat_mlp", "asset_set", "hierarchical_sequence_v2"}:
        raise ValueError("SB3 policy identity observation encoder is invalid")
    if encoder == "hierarchical_sequence_v2":
        architecture = payload.get("sequence_architecture")
        sequence_digest = payload.get("sequence_architecture_digest")
        if not isinstance(architecture, Mapping) or not architecture:
            raise ValueError("sequence architecture identity is missing")
        architecture_payload = dict(architecture)
        if not isinstance(sequence_digest, str) or sequence_digest != content_digest(
            architecture_payload
        ):
            raise ValueError("sequence architecture identity digest mismatch")
        for tuple_field in ("action_names", "symbols"):
            raw_tuple = architecture_payload.get(tuple_field)
            if isinstance(raw_tuple, list):
                architecture_payload[tuple_field] = tuple(raw_tuple)
        payload["sequence_architecture"] = architecture_payload
        raw_symbols = architecture_payload.get("symbols")
        symbols = _string_tuple(
            tuple(raw_symbols)
            if isinstance(raw_symbols, list | tuple)
            else raw_symbols,
            field="sequence architecture symbols",
        )
        actor_head = payload.get("actor_head")
        if actor_head != HIERARCHICAL_ACTOR_HEAD:
            raise ValueError("hierarchical actor-head identity mismatch")
        temperature = _positive_temperature(
            payload.get("gate_temperature"), field="gate_temperature"
        )
        current_weight = _validated_current_weight_identity(
            payload.get("current_weight_observation"), n_symbols=len(symbols)
        )
        exploration = _validated_hierarchical_exploration(
            payload.get("exploration_contract")
        )
        payload["current_weight_observation"] = current_weight
        payload["exploration_contract"] = exploration
        payload["gate_temperature"] = temperature
        architecture_contract = _policy_architecture_payload(
            actor_head=actor_head,
            gate_temperature=temperature,
            sequence_architecture_digest=sequence_digest,
            current_weight_observation=current_weight,
            exploration_contract=exploration,
        )
        architecture_digest = payload.get("policy_architecture_digest")
        if not isinstance(
            architecture_digest, str
        ) or architecture_digest != content_digest(architecture_contract):
            raise ValueError("policy architecture identity digest mismatch")
    elif any(
        key in payload
        for key in (
            "actor_head",
            "current_weight_observation",
            "exploration_contract",
            "gate_temperature",
            "policy_architecture_digest",
            "sequence_architecture",
            "sequence_architecture_digest",
        )
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
    """Bind the actual assembled encoder and actor identity to an SB3 model."""
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
        policy = getattr(model, "policy", None)
        actual_head = getattr(policy, "shared_actor_head", None)
        expected_head = getattr(assembly, "policy_actor_head", None)
        if actual_head != HIERARCHICAL_ACTOR_HEAD or expected_head != actual_head:
            raise ValueError("hierarchical actor-head assembly identity mismatch")
        actual_temperature = _positive_temperature(
            getattr(policy, "shared_actor_gate_temperature", None),
            field="model gate temperature",
        )
        expected_temperature = _positive_temperature(
            getattr(assembly, "hierarchical_gate_temperature", None),
            field="assembly gate temperature",
        )
        if not math.isclose(
            actual_temperature, expected_temperature, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("hierarchical gate-temperature assembly identity mismatch")
        current_weight = current_weight_observation_identity(len(symbols))
        exploration = _actual_hierarchical_exploration(policy)
        policy_architecture = _policy_architecture_payload(
            actor_head=actual_head,
            gate_temperature=actual_temperature,
            sequence_architecture_digest=identity.digest,
            current_weight_observation=current_weight,
            exploration_contract=exploration,
        )
        payload.update(
            {
                "actor_head": actual_head,
                "current_weight_observation": current_weight,
                "exploration_contract": exploration,
                "gate_temperature": actual_temperature,
                "policy_architecture_digest": content_digest(policy_architecture),
                "sequence_architecture": identity.digest_payload(),
                "sequence_architecture_digest": identity.digest,
            }
        )
    resolved = _validated_payload(payload)
    setattr(model, SB3_POLICY_IDENTITY_ATTRIBUTE, resolved)
    return dict(resolved)


def validated_sb3_policy_identity(value: object) -> dict[str, object]:
    """Validate and copy one serialized policy identity payload."""
    return dict(_validated_payload(value))


def model_sb3_policy_identity(model: object) -> dict[str, object] | None:
    raw = getattr(model, SB3_POLICY_IDENTITY_ATTRIBUTE, None)
    if raw is None:
        return None
    return _validated_payload(raw)


def validate_model_sb3_policy_identity(
    model: object, expected: Mapping[str, object]
) -> None:
    expected_payload = _validated_payload(expected)
    observed = model_sb3_policy_identity(model)
    if observed != expected_payload:
        raise ValueError("SB3 policy architecture identity mismatch")


__all__ = [
    "CURRENT_WEIGHT_KEY",
    "HIERARCHICAL_ACTION_DISTRIBUTION",
    "HIERARCHICAL_ACTOR_HEAD",
    "HIERARCHICAL_EXPLORATION_COUPLING",
    "HIERARCHICAL_EXPLORATION_SCHEMA",
    "HIERARCHICAL_LOG_STD_PARAMETERIZATION",
    "POLICY_ARCHITECTURE_SCHEMA",
    "SB3_POLICY_IDENTITY_ATTRIBUTE",
    "SB3_POLICY_IDENTITY_SCHEMA",
    "bind_sb3_policy_identity",
    "current_weight_observation_identity",
    "model_sb3_policy_identity",
    "validate_model_sb3_policy_identity",
    "validated_sb3_policy_identity",
]
