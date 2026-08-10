"""Generic policy and concrete deployment identities for universal instruments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.rl.universal_instrument_binding import (
    GENERIC_INSTRUMENT_SYMBOLS,
    GENERIC_TARGET_WEIGHT_ACTION_NAMES,
)

UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA: Final = (
    "universal_single_instrument_policy_v1"
)
SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA: Final = (
    "single_instrument_deployment_binding_v1"
)

_POLICY_MANIFEST_KEYS: Final = frozenset(
    {
        "action_names",
        "action_schema_digest",
        "action_shape",
        "architecture_digest",
        "instrument_descriptor_schema_digest",
        "normalizer_digest",
        "observation_schema_digest",
        "policy_symbols",
        "reward_environment_digest",
        "schema_version",
        "training_catalog_digest",
        "training_symbol_split_digest",
        "training_symbols_digest",
        "zero_shot_evidence_digest",
    }
)
_DEPLOYMENT_BINDING_KEYS: Final = frozenset(
    {
        "concrete_symbol",
        "dataset_feature_schema_digest",
        "execution_metadata_digest",
        "instrument_descriptor_evidence_digest",
        "market_instrument_contract_digest",
        "policy_digest",
        "schema_version",
        "seen_in_training",
    }
)


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _require_sha256_string(value: object, *, field: str) -> str:
    return require_sha256(_require_string(value, field=field), field=field)


def _require_field_closure(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    field: str,
) -> dict[str, object]:
    payload = dict(value)
    if frozenset(payload) != expected:
        raise ValueError(f"{field} field closure mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class UniversalSingleInstrumentPolicyManifest:
    """Generic policy identity with no concrete ticker binding."""

    architecture_digest: str
    observation_schema_digest: str
    action_schema_digest: str
    instrument_descriptor_schema_digest: str
    normalizer_digest: str
    reward_environment_digest: str
    training_catalog_digest: str
    training_symbol_split_digest: str
    training_symbols_digest: str
    zero_shot_evidence_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "architecture_digest",
            "observation_schema_digest",
            "action_schema_digest",
            "instrument_descriptor_schema_digest",
            "normalizer_digest",
            "reward_environment_digest",
            "training_catalog_digest",
            "training_symbol_split_digest",
            "training_symbols_digest",
            "zero_shot_evidence_digest",
        ):
            raw_value = getattr(self, field_name)
            if not isinstance(raw_value, str):
                raise TypeError(f"{field_name} must be a string")
            require_sha256(raw_value, field=field_name)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "action_names": list(GENERIC_TARGET_WEIGHT_ACTION_NAMES),
            "action_schema_digest": self.action_schema_digest,
            "action_shape": [1],
            "architecture_digest": self.architecture_digest,
            "instrument_descriptor_schema_digest": (
                self.instrument_descriptor_schema_digest
            ),
            "normalizer_digest": self.normalizer_digest,
            "observation_schema_digest": self.observation_schema_digest,
            "policy_symbols": list(GENERIC_INSTRUMENT_SYMBOLS),
            "reward_environment_digest": self.reward_environment_digest,
            "schema_version": UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA,
            "training_catalog_digest": self.training_catalog_digest,
            "training_symbol_split_digest": self.training_symbol_split_digest,
            "training_symbols_digest": self.training_symbols_digest,
            "zero_shot_evidence_digest": self.zero_shot_evidence_digest,
        }

    @property
    def policy_digest(self) -> str:
        return content_digest(self.to_json_dict())

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, object],
    ) -> UniversalSingleInstrumentPolicyManifest:
        if not isinstance(value, Mapping):
            raise TypeError("universal policy manifest must be a mapping")
        payload = _require_field_closure(
            value,
            expected=_POLICY_MANIFEST_KEYS,
            field="universal policy manifest",
        )
        if payload["schema_version"] != UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA:
            raise ValueError("universal policy manifest schema mismatch")
        if payload["policy_symbols"] != list(GENERIC_INSTRUMENT_SYMBOLS):
            raise ValueError("universal policy symbols must remain generic")
        if payload["action_names"] != list(GENERIC_TARGET_WEIGHT_ACTION_NAMES):
            raise ValueError("universal policy action names must remain generic")
        if payload["action_shape"] != [1]:
            raise ValueError("universal policy action shape must be [1]")
        return cls(
            architecture_digest=_require_sha256_string(
                payload["architecture_digest"],
                field="architecture_digest",
            ),
            observation_schema_digest=_require_sha256_string(
                payload["observation_schema_digest"],
                field="observation_schema_digest",
            ),
            action_schema_digest=_require_sha256_string(
                payload["action_schema_digest"],
                field="action_schema_digest",
            ),
            instrument_descriptor_schema_digest=_require_sha256_string(
                payload["instrument_descriptor_schema_digest"],
                field="instrument_descriptor_schema_digest",
            ),
            normalizer_digest=_require_sha256_string(
                payload["normalizer_digest"],
                field="normalizer_digest",
            ),
            reward_environment_digest=_require_sha256_string(
                payload["reward_environment_digest"],
                field="reward_environment_digest",
            ),
            training_catalog_digest=_require_sha256_string(
                payload["training_catalog_digest"],
                field="training_catalog_digest",
            ),
            training_symbol_split_digest=_require_sha256_string(
                payload["training_symbol_split_digest"],
                field="training_symbol_split_digest",
            ),
            training_symbols_digest=_require_sha256_string(
                payload["training_symbols_digest"],
                field="training_symbols_digest",
            ),
            zero_shot_evidence_digest=_require_sha256_string(
                payload["zero_shot_evidence_digest"],
                field="zero_shot_evidence_digest",
            ),
        )


@dataclass(frozen=True, slots=True)
class SingleInstrumentDeploymentBinding:
    """Bind one generic policy digest to one concrete deployment instrument."""

    policy_digest: str
    concrete_symbol: str
    market_instrument_contract_digest: str
    dataset_feature_schema_digest: str
    execution_metadata_digest: str
    instrument_descriptor_evidence_digest: str
    seen_in_training: bool

    def __post_init__(self) -> None:
        for field_name in (
            "policy_digest",
            "market_instrument_contract_digest",
            "dataset_feature_schema_digest",
            "execution_metadata_digest",
            "instrument_descriptor_evidence_digest",
        ):
            raw_value = getattr(self, field_name)
            if not isinstance(raw_value, str):
                raise TypeError(f"{field_name} must be a string")
            require_sha256(raw_value, field=field_name)
        if not isinstance(self.concrete_symbol, str):
            raise TypeError("concrete_symbol must be a string")
        object.__setattr__(
            self,
            "concrete_symbol",
            require_non_empty(
                self.concrete_symbol,
                field="concrete_symbol",
            ),
        )
        if not isinstance(self.seen_in_training, bool):
            raise TypeError("seen_in_training must be a boolean")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "concrete_symbol": self.concrete_symbol,
            "dataset_feature_schema_digest": self.dataset_feature_schema_digest,
            "execution_metadata_digest": self.execution_metadata_digest,
            "instrument_descriptor_evidence_digest": (
                self.instrument_descriptor_evidence_digest
            ),
            "market_instrument_contract_digest": (
                self.market_instrument_contract_digest
            ),
            "policy_digest": self.policy_digest,
            "schema_version": SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA,
            "seen_in_training": self.seen_in_training,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_json_dict())

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, object],
    ) -> SingleInstrumentDeploymentBinding:
        if not isinstance(value, Mapping):
            raise TypeError("single-instrument deployment binding must be a mapping")
        payload = _require_field_closure(
            value,
            expected=_DEPLOYMENT_BINDING_KEYS,
            field="single-instrument deployment binding",
        )
        if (
            payload["schema_version"]
            != SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA
        ):
            raise ValueError("single-instrument deployment binding schema mismatch")
        seen_in_training = payload["seen_in_training"]
        if not isinstance(seen_in_training, bool):
            raise TypeError("seen_in_training must be a boolean")
        return cls(
            policy_digest=_require_sha256_string(
                payload["policy_digest"],
                field="policy_digest",
            ),
            concrete_symbol=_require_string(
                payload["concrete_symbol"],
                field="concrete_symbol",
            ),
            market_instrument_contract_digest=_require_sha256_string(
                payload["market_instrument_contract_digest"],
                field="market_instrument_contract_digest",
            ),
            dataset_feature_schema_digest=_require_sha256_string(
                payload["dataset_feature_schema_digest"],
                field="dataset_feature_schema_digest",
            ),
            execution_metadata_digest=_require_sha256_string(
                payload["execution_metadata_digest"],
                field="execution_metadata_digest",
            ),
            instrument_descriptor_evidence_digest=_require_sha256_string(
                payload["instrument_descriptor_evidence_digest"],
                field="instrument_descriptor_evidence_digest",
            ),
            seen_in_training=seen_in_training,
        )


__all__ = [
    "SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA",
    "UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA",
    "SingleInstrumentDeploymentBinding",
    "UniversalSingleInstrumentPolicyManifest",
]
