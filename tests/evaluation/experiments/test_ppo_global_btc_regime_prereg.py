from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments.ppo_global_btc_regime_prereg import (
    canonical_ppo_global_btc_regime_protocol,
    load_ppo_global_btc_regime_protocol,
)

_MAIN_SHA = "c5a1ce8395feaedd8833f7e596fddc9d8f3115fc"
_DATASET_ID = "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
_DATASET_DIGEST = "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
_STUDY_DIGEST = "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
_EVIDENCE_FINGERPRINT = (
    "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
)


def test_protocol_freezes_one_global_btc_regime_factor() -> None:
    protocol = canonical_ppo_global_btc_regime_protocol()

    assert protocol.schema_version == "ppo_global_btc_regime_prereg_v1"
    assert protocol.source_main_sha == _MAIN_SHA
    assert protocol.source_dataset_id == _DATASET_ID
    assert protocol.source_dataset_artifact_digest == _DATASET_DIGEST
    assert protocol.source_study_digest == _STUDY_DIGEST
    assert protocol.source_evidence_fingerprint == _EVIDENCE_FINGERPRINT

    assert protocol.controlled_factor == "ppo_global_btc_regime_context"
    assert protocol.reference_symbol == "BTCUSDT"
    assert protocol.reference_feature == "1h__log_return_24bar"
    assert protocol.reference_requires_available is True
    assert protocol.reference_requires_finite is True
    assert protocol.reference_includes_normalized_staleness is True
    assert protocol.ppo_seeds == (0, 1, 2, 3, 4)

    assert protocol.factor_slots_authorized == 1
    assert protocol.action_contract_changed is False
    assert protocol.execution_contract_changed is False
    assert protocol.reward_contract_changed is False
    assert protocol.risk_contract_changed is False
    assert protocol.economics_contract_changed is False
    assert protocol.final_test_access_authorized is False
    assert protocol.economic_execution_authorized is False
    assert protocol.economic_result_inspected is False


def test_protocol_rejects_preregistered_semantic_drift() -> None:
    protocol = canonical_ppo_global_btc_regime_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"source_main_sha": "0" * 40},
        {"source_dataset_id": "0" * 64},
        {"source_dataset_artifact_digest": "0" * 64},
        {"source_study_digest": "0" * 64},
        {"source_evidence_fingerprint": "0" * 64},
        {"controlled_factor": "instrument_embedding"},
        {"reference_symbol": "ETHUSDT"},
        {"reference_feature": "1h__log_return_48bar"},
        {"reference_requires_available": False},
        {"reference_requires_finite": False},
        {"reference_includes_normalized_staleness": False},
        {"ppo_seeds": (0, 1, 2, 3, 4, 5)},
        {"factor_slots_authorized": 2},
        {"action_contract_changed": True},
        {"execution_contract_changed": True},
        {"reward_contract_changed": True},
        {"risk_contract_changed": True},
        {"economics_contract_changed": True},
        {"final_test_access_authorized": True},
        {"economic_execution_authorized": True},
        {"economic_result_inspected": True},
    )

    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_protocol_loader_is_strict_and_round_trips(tmp_path: Path) -> None:
    protocol = canonical_ppo_global_btc_regime_protocol()
    payload = protocol.to_payload()
    path = tmp_path / "ppo-global-btc-regime-prereg.json"

    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    reconstructed = load_ppo_global_btc_regime_protocol(path)
    assert reconstructed.to_payload() == payload
    assert reconstructed.digest == protocol.digest

    unknown = dict(payload)
    unknown["instrument_profile"] = {"volatility_30d": True}
    path.write_text(
        json.dumps(unknown, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keys|unknown|preregistered"):
        load_ppo_global_btc_regime_protocol(path)


def test_protocol_payload_contains_no_result_or_rescue_authority() -> None:
    protocol = canonical_ppo_global_btc_regime_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "pnl",
        "returns",
        "profitability",
        "economic_result",
        "final_test_result",
        "instrument_profile",
        "symbol_embedding",
        "btc_residual",
        "beta_fit",
        "zero_snap",
        "neutral_decay",
        "gate_relaxation",
        "extra_seeds",
        "extra_symbols",
    }
    assert forbidden.isdisjoint(payload)
    assert payload["factor_slots_authorized"] == 1
    assert payload["economic_execution_authorized"] is False
    assert len(protocol.digest) == 64
