"""Identity contracts for one generic policy and concrete deployments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.instrument_episode_routing import (
    GENERIC_INSTRUMENT_ACTION_NAMES,
    GENERIC_INSTRUMENT_SYMBOL,
    GENERIC_INSTRUMENT_SYMBOLS,
)

UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA = (
    "universal_single_instrument_policy_v1"
)
UNIVERSAL_SINGLE_INSTRUMENT_ACTION_SCHEMA = "single_target_weight_action_v1"
SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA = (
    "single_instrument_deployment_binding_v1"
)


def _require_non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_digest(value: object, *, field: str) -> str:
    resolved = _require_non_empty_string(value, field=field)
    require_sha256(resolved, field=field)
    return resolved


def _require_exact_fields(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    field: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{field} fields mismatch: missing={missing}, extra={extra}")


def _serialized_string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a string list or tuple")
    resolved = tuple(value)
    if not resolved or any(not isinstance(item, str) or not item for item in resolved):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must contain unique strings")
    return resolved


_POLICY_DIGEST_FIELDS = (
    "architecture_digest",
    "observation_schema_digest",
    "instrument_descriptor_schema_digest",
    "normalizer_digest",
    "reward_environment_digest",
    "training_catalog_digest",
    "training_symbol_split_digest",
    "training_symbols_digest",
    "zero_shot_evidence_digest",
)
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "action_schema",
        "generic_symbols",
        "generic_action_names",
        *_POLICY_DIGEST_FIELDS,
        "policy_digest",
    }
)


@dataclass(frozen=True, slots=True)
class UniversalSingleInstrumentPolicyIdentity:
    """Ticker-free identity shared by training, checkpoints, and serving."""

    architecture_digest: str
    observation_schema_digest: str
    instrument_descriptor_schema_digest: str
    normalizer_digest: str
    reward_environment_digest: str
    training_catalog_digest: str
    training_symbol_split_digest: str
    training_symbols_digest: str
    zero_shot_evidence_digest: str
    generic_symbols: tuple[str, ...] = GENERIC_INSTRUMENT_SYMBOLS
    generic_action_names: tuple[str, ...] = GENERIC_INSTRUMENT_ACTION_NAMES
    action_schema: str = UNIVERSAL_SINGLE_INSTRUMENT_ACTION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in _POLICY_DIGEST_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _require_digest(getattr(self, field_name), field=field_name),
            )
        generic_symbols = _serialized_string_tuple(
            self.generic_symbols,
            field="generic_symbols",
        )
        if generic_symbols != GENERIC_INSTRUMENT_SYMBOLS:
            raise ValueError(
                "generic_symbols must be the identity-free INSTRUMENT slot"
            )
        generic_action_names = _serialized_string_tuple(
            self.generic_action_names,
            field="generic_action_names",
        )
        if generic_action_names != GENERIC_INSTRUMENT_ACTION_NAMES:
            raise ValueError(
                "generic_action_names must contain one INSTRUMENT target weight"
            )
        action_schema = _require_non_empty_string(
            self.action_schema,
            field="action_schema",
        )
        if action_schema != UNIVERSAL_SINGLE_INSTRUMENT_ACTION_SCHEMA:
            raise ValueError("action_schema must remain the universal scalar contract")
        object.__setattr__(self, "generic_symbols", generic_symbols)
        object.__setattr__(self, "generic_action_names", generic_action_names)
        object.__setattr__(self, "action_schema", action_schema)

    def digest_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA,
            "action_schema": self.action_schema,
            "generic_symbols": self.generic_symbols,
            "generic_action_names": self.generic_action_names,
        }
        payload.update(
            {field_name: getattr(self, field_name) for field_name in _POLICY_DIGEST_FIELDS}
        )
        return payload

    @property
    def policy_digest(self) -> str:
        return content_digest(self.digest_payload())

    def to_json_dict(self) -> dict[str, object]:
        payload = {**self.digest_payload(), "policy_digest": self.policy_digest}
        canonical_json_bytes(payload)
        return payload

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, object],
    ) -> UniversalSingleInstrumentPolicyIdentity:
        if not isinstance(value, Mapping):
            raise ValueError("universal policy identity must be a mapping")
        payload = dict(value)
        _require_exact_fields(
            payload,
            expected=_POLICY_FIELDS,
            field="universal policy identity",
        )
        if payload["schema_version"] != UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA:
            raise ValueError("universal policy identity schema mismatch")
        observed_digest = _require_digest(
            payload["policy_digest"],
            field="policy_digest",
        )
        identity = cls(
            architecture_digest=payload["architecture_digest"],  # type: ignore[arg-type]
            observation_schema_digest=payload[
                "observation_schema_digest"
            ],  # type: ignore[arg-type]
            instrument_descriptor_schema_digest=payload[
                "instrument_descriptor_schema_digest"
            ],  # type: ignore[arg-type]
            normalizer_digest=payload["normalizer_digest"],  # type: ignore[arg-type]
            reward_environment_digest=payload[
                "reward_environment_digest"
            ],  # type: ignore[arg-type]
            training_catalog_digest=payload[
                "training_catalog_digest"
            ],  # type: ignore[arg-type]
            training_symbol_split_digest=payload[
                "training_symbol_split_digest"
            ],  # type: ignore[arg-type]
            training_symbols_digest=payload[
                "training_symbols_digest"
            ],  # type: ignore[arg-type]
            zero_shot_evidence_digest=payload[
                "zero_shot_evidence_digest"
            ],  # type: ignore[arg-type]
            generic_symbols=_serialized_string_tuple(
                payload["generic_symbols"],
                field="generic_symbols",
            ),
            generic_action_names=_serialized_string_tuple(
                payload["generic_action_names"],
                field="generic_action_names",
            ),
            action_schema=payload["action_schema"],  # type: ignore[arg-type]
        )
        if observed_digest != identity.policy_digest:
            raise ValueError("universal policy identity digest mismatch")
        canonical_json_bytes(payload)
        return identity


_DEPLOYMENT_DIGEST_FIELDS = (
    "policy_digest",
    "market_instrument_contract_digest",
    "dataset_feature_schema_digest",
    "execution_metadata_digest",
    "instrument_descriptor_evidence_digest",
)
_DEPLOYMENT_FIELDS = frozenset(
    {
        "schema_version",
        "concrete_symbol",
        *_DEPLOYMENT_DIGEST_FIELDS,
        "seen_in_training",
        "binding_digest",
    }
)


@dataclass(frozen=True, slots=True)
class SingleInstrumentDeploymentBinding:
    """The only U2 identity that binds a generic policy to a concrete ticker."""

    policy_digest: str
    concrete_symbol: str
    market_instrument_contract_digest: str
    dataset_feature_schema_digest: str
    execution_metadata_digest: str
    instrument_descriptor_evidence_digest: str
    seen_in_training: bool

    def __post_init__(self) -> None:
        for field_name in _DEPLOYMENT_DIGEST_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _require_digest(getattr(self, field_name), field=field_name),
            )
        concrete_symbol = _require_non_empty_string(
            self.concrete_symbol,
            field="concrete_symbol",
        )
        if concrete_symbol == GENERIC_INSTRUMENT_SYMBOL:
            raise ValueError("concrete_symbol must not use the generic INSTRUMENT slot")
        if not isinstance(self.seen_in_training, bool):
            raise ValueError("seen_in_training must be a boolean")
        object.__setattr__(self, "concrete_symbol", concrete_symbol)

    def digest_payload(self) -> dict[str, object]:
        return {
            "schema_version": SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA,
            "policy_digest": self.policy_digest,
            "concrete_symbol": self.concrete_symbol,
            "market_instrument_contract_digest": (
                self.market_instrument_contract_digest
            ),
            "dataset_feature_schema_digest": self.dataset_feature_schema_digest,
            "execution_metadata_digest": self.execution_metadata_digest,
            "instrument_descriptor_evidence_digest": (
                self.instrument_descriptor_evidence_digest
            ),
            "seen_in_training": self.seen_in_training,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())

    def to_json_dict(self) -> dict[str, object]:
        payload = {**self.digest_payload(), "binding_digest": self.digest}
        canonical_json_bytes(payload)
        return payload

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, object],
    ) -> SingleInstrumentDeploymentBinding:
        if not isinstance(value, Mapping):
            raise ValueError("single-instrument deployment binding must be a mapping")
        payload = dict(value)
        _require_exact_fields(
            payload,
            expected=_DEPLOYMENT_FIELDS,
            field="single-instrument deployment binding",
        )
        if payload["schema_version"] != SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA:
            raise ValueError("single-instrument deployment binding schema mismatch")
        observed_digest = _require_digest(
            payload["binding_digest"],
            field="binding_digest",
        )
        binding = cls(
            policy_digest=payload["policy_digest"],  # type: ignore[arg-type]
            concrete_symbol=payload["concrete_symbol"],  # type: ignore[arg-type]
            market_instrument_contract_digest=payload[
                "market_instrument_contract_digest"
            ],  # type: ignore[arg-type]
            dataset_feature_schema_digest=payload[
                "dataset_feature_schema_digest"
            ],  # type: ignore[arg-type]
            execution_metadata_digest=payload[
                "execution_metadata_digest"
            ],  # type: ignore[arg-type]
            instrument_descriptor_evidence_digest=payload[
                "instrument_descriptor_evidence_digest"
            ],  # type: ignore[arg-type]
            seen_in_training=payload["seen_in_training"],  # type: ignore[arg-type]
        )
        if observed_digest != binding.digest:
            raise ValueError("single-instrument deployment binding digest mismatch")
        canonical_json_bytes(payload)
        return binding


__all__ = [
    "SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA",
    "UNIVERSAL_SINGLE_INSTRUMENT_ACTION_SCHEMA",
    "UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA",
    "SingleInstrumentDeploymentBinding",
    "UniversalSingleInstrumentPolicyIdentity",
]
