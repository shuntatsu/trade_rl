from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

_PREREG_HEAD = "d84235185c5e79327ea7ced7073f0026970ba46b"
_PREREG_DIGEST = "65a803034fca8180c165728022221d7b3c2fa5dcdd8349d3ca52c3a6adc72228"
_IMPLEMENTATION_HEAD = "1ae494ce182189c5be61e8007dc956f912d8b9d5"
_IMPLEMENTATION_TREE = "4f14c5a06505b8ff9811b473bf6dea296453a13b"
_IMPLEMENTATION_INDEX_DIGEST = (
    "d5390a0025a92e4e2a1e987b7b7ac8ee6ef3974c5c8247b0256ae70c0d2d6e10"
)
_DATASET_ID = "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
_STUDY_DIGEST = "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
_BASELINE_EVIDENCE_FINGERPRINT = (
    "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
)


def _module():
    return __import__(
        "trade_rl.evaluation.experiments.ppo_global_btc_regime_evaluation_prereg",
        fromlist=["*"],
    )


def test_protocol_binds_prereg_implementation_and_portable_baseline_authorities() -> (
    None
):
    protocol = _module().canonical_ppo_global_btc_regime_evaluation_protocol()

    assert protocol.schema_version == "ppo_global_btc_regime_evaluation_prereg_v1"
    assert protocol.issue_number == 609
    assert protocol.factor_issue_number == 605
    assert protocol.implementation_issue_number == 607
    assert protocol.factor_prereg_head == _PREREG_HEAD
    assert protocol.factor_prereg_protocol_digest == _PREREG_DIGEST
    assert protocol.implementation_head == _IMPLEMENTATION_HEAD
    assert protocol.implementation_tree == _IMPLEMENTATION_TREE
    assert protocol.implementation_index_digest == _IMPLEMENTATION_INDEX_DIGEST
    assert protocol.source_dataset_id == _DATASET_ID
    assert protocol.source_dataset_artifact_digest == _DATASET_ARTIFACT_DIGEST
    assert protocol.source_study_digest == _STUDY_DIGEST
    assert protocol.baseline_evidence_fingerprint == _BASELINE_EVIDENCE_FINGERPRINT


def test_protocol_freezes_exact_one_factor_candidate_and_existing_analysis_contract() -> (
    None
):
    protocol = _module().canonical_ppo_global_btc_regime_evaluation_protocol()

    assert protocol.controlled_factor == "FEATURE_SET"
    assert protocol.semantic_factor == "ppo_global_btc_regime_context"
    assert protocol.baseline_observation_schema == "ppo_observation_v2"
    assert (
        protocol.candidate_observation_schema == "ppo_observation_v3_global_btc_regime"
    )
    assert protocol.ppo_seeds == (0, 1, 2, 3, 4)
    assert protocol.ppo_total_timesteps == 100_000
    assert protocol.n_bootstrap == 2_000
    assert protocol.bootstrap_seed == 1_729
    assert protocol.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    assert protocol.unaffected_strategy_names == (
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
        "cash",
        "constant_long",
        "constant_short",
    )
    assert protocol.comparison_schema == "controlled_evidence_comparison_v2"
    assert protocol.unaffected_raw_return_invariance_required is True
    assert protocol.only_candidate_ppo_observation_may_change is True
    assert protocol.source_dataset_may_change is False
    assert protocol.development_window_may_change is False
    assert protocol.ppo_hyperparameters_may_change is False
    assert protocol.execution_economics_may_change is False


def test_protocol_freezes_strict_robust_profit_decision_rule() -> None:
    protocol = _module().canonical_ppo_global_btc_regime_evaluation_protocol()

    assert protocol.accept_requires_positive_factor_effect_symbols == 5
    assert protocol.accept_requires_positive_candidate_return_symbols == 5
    assert protocol.accept_requires_positive_cross_symbol_median_seeds == 5
    assert protocol.accept_requires_cross_symbol_median_factor_effect_gt_zero is True
    assert protocol.accept_requires_cross_symbol_median_candidate_return_gt_zero is True
    assert protocol.accept_requires_no_new_termination is True
    assert protocol.accept_requires_all_unaffected_raw_returns_equal is True
    assert protocol.valid_non_accept_decision == "KEEP_BASELINE"
    assert protocol.invalid_decision == "INVALID"
    assert protocol.bootstrap_statistics_are_diagnostic_only is True
    assert protocol.no_result_dependent_rescue is True


def test_protocol_keeps_forbidden_boundaries_closed() -> None:
    protocol = _module().canonical_ppo_global_btc_regime_evaluation_protocol()

    assert protocol.economic_execution_authorized is False
    assert protocol.economic_result_inspected is False
    assert protocol.final_test_access_authorized is False
    assert protocol.production_eligible is False
    assert protocol.live_trading_authorized is False
    assert protocol.merge_authorized is False


def test_protocol_rejects_any_authority_or_decision_rule_drift() -> None:
    protocol = _module().canonical_ppo_global_btc_regime_evaluation_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"factor_prereg_head": "0" * 40},
        {"factor_prereg_protocol_digest": "0" * 64},
        {"implementation_head": "0" * 40},
        {"implementation_tree": "0" * 40},
        {"implementation_index_digest": "0" * 64},
        {"source_dataset_id": "0" * 64},
        {"source_dataset_artifact_digest": "0" * 64},
        {"source_study_digest": "0" * 64},
        {"baseline_evidence_fingerprint": "0" * 64},
        {"controlled_factor": "PPO_TRAINING_BUDGET"},
        {"semantic_factor": "symbol_embedding"},
        {"candidate_observation_schema": "ppo_observation_v2"},
        {"ppo_seeds": (0, 1, 2, 3, 4, 5)},
        {"ppo_total_timesteps": 200_000},
        {"n_bootstrap": 10_000},
        {"bootstrap_seed": 0},
        {"accept_requires_positive_factor_effect_symbols": 4},
        {"accept_requires_positive_candidate_return_symbols": 4},
        {"accept_requires_positive_cross_symbol_median_seeds": 4},
        {"accept_requires_cross_symbol_median_factor_effect_gt_zero": False},
        {"accept_requires_cross_symbol_median_candidate_return_gt_zero": False},
        {"accept_requires_no_new_termination": False},
        {"accept_requires_all_unaffected_raw_returns_equal": False},
        {"valid_non_accept_decision": "INCONCLUSIVE"},
        {"bootstrap_statistics_are_diagnostic_only": False},
        {"no_result_dependent_rescue": False},
        {"economic_execution_authorized": True},
        {"economic_result_inspected": True},
        {"final_test_access_authorized": True},
        {"production_eligible": True},
        {"live_trading_authorized": True},
        {"merge_authorized": True},
    )

    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_protocol_loader_is_canonical_strict_and_result_blind(tmp_path: Path) -> None:
    module = _module()
    protocol = module.canonical_ppo_global_btc_regime_evaluation_protocol()
    payload = protocol.to_payload()
    path = tmp_path / "ppo-global-btc-regime-evaluation-prereg.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    reconstructed = module.load_ppo_global_btc_regime_evaluation_protocol(path)
    assert reconstructed.to_payload() == payload
    assert reconstructed.digest == protocol.digest

    forbidden_result_fields = {
        "candidate_result",
        "candidate_returns",
        "candidate_pnl",
        "comparison_result",
        "decision_result",
        "positive_factor_effect_symbols_observed",
        "positive_candidate_return_symbols_observed",
        "winner",
    }
    assert forbidden_result_fields.isdisjoint(payload)

    extra = dict(payload)
    extra["candidate_pnl"] = 1.0
    path.write_text(
        json.dumps(extra, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keys|unknown|preregistered"):
        module.load_ppo_global_btc_regime_evaluation_protocol(path)
