from __future__ import annotations

import json

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.universal_instrument_binding import (
    GENERIC_INSTRUMENT_SYMBOLS,
    GENERIC_TARGET_WEIGHT_ACTION_NAMES,
)
from trade_rl.rl.universal_single_instrument_identity import (
    SingleInstrumentDeploymentBinding,
    UniversalSingleInstrumentPolicyManifest,
)


def _digest(value: object) -> str:
    return content_digest(value)


def _manifest() -> UniversalSingleInstrumentPolicyManifest:
    return UniversalSingleInstrumentPolicyManifest(
        architecture_digest=_digest("architecture"),
        observation_schema_digest=_digest("observation"),
        action_schema_digest=_digest("action"),
        instrument_descriptor_schema_digest=_digest("descriptor-schema"),
        normalizer_digest=_digest("normalizer"),
        reward_environment_digest=_digest("reward-environment"),
        training_catalog_digest=_digest("catalog"),
        training_symbol_split_digest=_digest("split"),
        training_symbols_digest=_digest(("BTCUSDT", "ETHUSDT")),
        zero_shot_evidence_digest=_digest("zero-shot"),
    )


def test_universal_policy_manifest_is_generic_and_canonical() -> None:
    manifest = _manifest()
    payload = manifest.to_json_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["policy_symbols"] == list(GENERIC_INSTRUMENT_SYMBOLS)
    assert payload["action_names"] == list(GENERIC_TARGET_WEIGHT_ACTION_NAMES)
    assert payload["action_shape"] == [1]
    assert "BTCUSDT" not in serialized
    assert "ETHUSDT" not in serialized
    assert manifest.policy_digest == content_digest(payload)
    assert UniversalSingleInstrumentPolicyManifest.from_json_dict(payload) == manifest


def test_concrete_symbol_lives_only_in_deployment_binding() -> None:
    manifest = _manifest()
    binding = SingleInstrumentDeploymentBinding(
        policy_digest=manifest.policy_digest,
        concrete_symbol="XRPUSDT",
        market_instrument_contract_digest=_digest("market-contract"),
        dataset_feature_schema_digest=_digest("feature-schema"),
        execution_metadata_digest=_digest("execution"),
        instrument_descriptor_evidence_digest=_digest("descriptor-evidence"),
        seen_in_training=False,
    )

    assert binding.to_json_dict()["concrete_symbol"] == "XRPUSDT"
    assert binding.to_json_dict()["seen_in_training"] is False
    assert binding.digest == content_digest(binding.to_json_dict())
    assert (
        SingleInstrumentDeploymentBinding.from_json_dict(binding.to_json_dict())
        == binding
    )
    assert "XRPUSDT" not in json.dumps(manifest.to_json_dict(), sort_keys=True)


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_policy_manifest_rejects_invalid_digests(field: str) -> None:
    payload = {
        "architecture_digest": _digest("architecture"),
        "observation_schema_digest": _digest("observation"),
        "action_schema_digest": _digest("action"),
        "instrument_descriptor_schema_digest": _digest("descriptor-schema"),
        "normalizer_digest": _digest("normalizer"),
        "reward_environment_digest": _digest("reward-environment"),
        "training_catalog_digest": _digest("catalog"),
        "training_symbol_split_digest": _digest("split"),
        "training_symbols_digest": _digest("symbols"),
        "zero_shot_evidence_digest": _digest("zero-shot"),
    }
    payload[field] = "invalid"

    with pytest.raises(ValueError, match="SHA-256"):
        UniversalSingleInstrumentPolicyManifest(**payload)


def test_deployment_binding_validates_concrete_identity_and_boolean() -> None:
    manifest = _manifest()
    common = {
        "policy_digest": manifest.policy_digest,
        "concrete_symbol": "BTCUSDT",
        "market_instrument_contract_digest": _digest("market-contract"),
        "dataset_feature_schema_digest": _digest("feature-schema"),
        "execution_metadata_digest": _digest("execution"),
        "instrument_descriptor_evidence_digest": _digest("descriptor-evidence"),
        "seen_in_training": True,
    }

    with pytest.raises(ValueError, match="non-empty"):
        SingleInstrumentDeploymentBinding(**{**common, "concrete_symbol": " "})
    with pytest.raises(TypeError, match="boolean"):
        SingleInstrumentDeploymentBinding(**{**common, "seen_in_training": 1})
    with pytest.raises(ValueError, match="SHA-256"):
        SingleInstrumentDeploymentBinding(**{**common, "policy_digest": "bad"})
