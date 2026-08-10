from __future__ import annotations

import json
from dataclasses import replace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.instrument_episode_routing import (
    GENERIC_INSTRUMENT_ACTION_NAMES,
    GENERIC_INSTRUMENT_SYMBOLS,
)
from trade_rl.rl.universal_policy_identity import (
    SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA,
    UNIVERSAL_SINGLE_INSTRUMENT_ACTION_SCHEMA,
    UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA,
    SingleInstrumentDeploymentBinding,
    UniversalSingleInstrumentPolicyIdentity,
)


def _digest(label: str) -> str:
    return content_digest(label)


def _policy() -> UniversalSingleInstrumentPolicyIdentity:
    return UniversalSingleInstrumentPolicyIdentity(
        architecture_digest=_digest("architecture"),
        observation_schema_digest=_digest("observation"),
        instrument_descriptor_schema_digest=_digest("descriptor-schema"),
        normalizer_digest=_digest("normalizer"),
        reward_environment_digest=_digest("reward-environment"),
        training_catalog_digest=_digest("catalog"),
        training_symbol_split_digest=_digest("split"),
        training_symbols_digest=_digest(("BTCUSDT", "ETHUSDT", "SOLUSDT")),
        zero_shot_evidence_digest=_digest("zero-shot-evidence"),
    )


def test_universal_policy_identity_is_generic_and_digest_bound() -> None:
    identity = _policy()

    payload = identity.to_json_dict()

    assert payload["schema_version"] == UNIVERSAL_SINGLE_INSTRUMENT_POLICY_SCHEMA
    assert payload["action_schema"] == UNIVERSAL_SINGLE_INSTRUMENT_ACTION_SCHEMA
    assert payload["generic_symbols"] == GENERIC_INSTRUMENT_SYMBOLS
    assert payload["generic_action_names"] == GENERIC_INSTRUMENT_ACTION_NAMES
    assert payload["policy_digest"] == identity.policy_digest
    assert UniversalSingleInstrumentPolicyIdentity.from_json_dict(payload) == identity


def test_universal_policy_payload_cannot_contain_concrete_ticker_identity() -> None:
    payload = _policy().to_json_dict()
    serialized = json.dumps(payload, sort_keys=True)

    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        assert symbol not in serialized

    with pytest.raises(ValueError, match="fields|concrete"):
        UniversalSingleInstrumentPolicyIdentity.from_json_dict(
            {**payload, "concrete_symbol": "BTCUSDT"}
        )


def test_universal_policy_identity_rejects_invalid_fixed_contract_or_digest() -> None:
    identity = _policy()

    with pytest.raises(ValueError, match="generic_symbols"):
        replace(identity, generic_symbols=("BTCUSDT",))
    with pytest.raises(ValueError, match="generic_action_names"):
        replace(identity, generic_action_names=("target_weight:BTCUSDT",))
    with pytest.raises(ValueError, match="action_schema"):
        replace(identity, action_schema="portfolio_weights_v1")

    tampered = identity.to_json_dict()
    tampered["normalizer_digest"] = _digest("other-normalizer")
    with pytest.raises(ValueError, match="digest"):
        UniversalSingleInstrumentPolicyIdentity.from_json_dict(tampered)


def test_universal_policy_identity_rejects_bad_digest_field() -> None:
    with pytest.raises(ValueError, match="architecture_digest"):
        replace(_policy(), architecture_digest="not-a-digest")


def test_deployment_binding_is_the_only_contract_with_concrete_symbol() -> None:
    policy = _policy()
    binding = SingleInstrumentDeploymentBinding(
        policy_digest=policy.policy_digest,
        concrete_symbol="XRPUSDT",
        market_instrument_contract_digest=_digest("market-instrument"),
        dataset_feature_schema_digest=_digest("feature-schema"),
        execution_metadata_digest=_digest("execution"),
        instrument_descriptor_evidence_digest=_digest("descriptor-evidence"),
        seen_in_training=False,
    )

    payload = binding.to_json_dict()

    assert payload["schema_version"] == SINGLE_INSTRUMENT_DEPLOYMENT_BINDING_SCHEMA
    assert payload["concrete_symbol"] == "XRPUSDT"
    assert payload["seen_in_training"] is False
    assert payload["binding_digest"] == binding.digest
    assert SingleInstrumentDeploymentBinding.from_json_dict(payload) == binding


def test_deployment_binding_supports_seen_and_unseen_without_changing_policy() -> None:
    policy = _policy()
    unseen = SingleInstrumentDeploymentBinding(
        policy_digest=policy.policy_digest,
        concrete_symbol="XRPUSDT",
        market_instrument_contract_digest=_digest("market-instrument:xrp"),
        dataset_feature_schema_digest=_digest("feature-schema"),
        execution_metadata_digest=_digest("execution:xrp"),
        instrument_descriptor_evidence_digest=_digest("descriptor:xrp"),
        seen_in_training=False,
    )
    seen = SingleInstrumentDeploymentBinding(
        policy_digest=policy.policy_digest,
        concrete_symbol="BTCUSDT",
        market_instrument_contract_digest=_digest("market-instrument:btc"),
        dataset_feature_schema_digest=_digest("feature-schema"),
        execution_metadata_digest=_digest("execution:btc"),
        instrument_descriptor_evidence_digest=_digest("descriptor:btc"),
        seen_in_training=True,
    )

    assert unseen.policy_digest == seen.policy_digest == policy.policy_digest
    assert unseen.digest != seen.digest


def test_deployment_binding_rejects_generic_symbol_non_boolean_and_tampering() -> None:
    policy = _policy()
    kwargs = {
        "policy_digest": policy.policy_digest,
        "concrete_symbol": "XRPUSDT",
        "market_instrument_contract_digest": _digest("market-instrument"),
        "dataset_feature_schema_digest": _digest("feature-schema"),
        "execution_metadata_digest": _digest("execution"),
        "instrument_descriptor_evidence_digest": _digest("descriptor-evidence"),
        "seen_in_training": False,
    }

    with pytest.raises(ValueError, match="concrete_symbol"):
        SingleInstrumentDeploymentBinding(**{**kwargs, "concrete_symbol": "INSTRUMENT"})
    with pytest.raises(ValueError, match="seen_in_training"):
        SingleInstrumentDeploymentBinding(**{**kwargs, "seen_in_training": 1})

    binding = SingleInstrumentDeploymentBinding(**kwargs)
    tampered = binding.to_json_dict()
    tampered["concrete_symbol"] = "ADAUSDT"
    with pytest.raises(ValueError, match="digest"):
        SingleInstrumentDeploymentBinding.from_json_dict(tampered)
    with pytest.raises(ValueError, match="fields"):
        SingleInstrumentDeploymentBinding.from_json_dict(
            {**binding.to_json_dict(), "symbol_index": 4}
        )
