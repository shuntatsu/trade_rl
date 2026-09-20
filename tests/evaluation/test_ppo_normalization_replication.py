from __future__ import annotations

import json

import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.ppo_normalization_replication import (
    CANDIDATE_ARM,
    CONTROL_ARM,
    PPO_NORMALIZATION_PROTOCOL_SCHEMA,
    expected_ppo_normalization_protocol,
    load_ppo_normalization_protocol,
    ppo_normalization_protocol_bytes,
)


def test_protocol_freezes_single_normalization_factor_on_current_economics() -> None:
    protocol = expected_ppo_normalization_protocol()

    assert protocol["schema"] == PPO_NORMALIZATION_PROTOCOL_SCHEMA
    assert protocol["software_authority"]["minimum_main_sha"] == (
        "2068dbba481945092a082a79aa640edce8f951b8"
    )
    assert protocol["source"] == {
        "dataset_id": "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
        "dataset_artifact_digest": "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7",
        "study_digest": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    }

    common = protocol["common"]
    assert common["symbols"] == [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    ]
    assert common["seeds"] == [0, 1, 2, 3, 4]
    assert common["caller_total_timesteps"] == 262_144
    assert common["training_layout"] == "sequential"
    assert common["rollout_steps_per_env"] is None
    assert common["initial_capital"] == 10_000.0
    assert common["gross_budget"] == 0.1
    assert common["risk_config"] is None
    assert common["settle_terminal_position"] is True
    assert common["execution_policy"]["borrow_rate_multiplier"] == 1.0
    assert common["execution_policy"]["processing_bar_volume_capacity"] is False
    assert common["execution_policy"]["max_leverage"] == 1.0
    assert common["development_window"] == {
        "start_inclusive": "2023-01-01T00:00:00Z",
        "stop_exclusive": "2025-01-01T00:00:00Z",
        "intervals": 17_544,
    }
    assert common["fit_scope"] == (
        "source_study_baseline_config.feature_indices/fit_symbol_indices/fit_cutoff"
    )
    assert common["observation"]["schema_version"] == "ppo_observation_v2"

    control = protocol["arms"][CONTROL_ARM]
    candidate = protocol["arms"][CANDIDATE_ARM]
    assert control == {
        "normalize_features": False,
        "normalizer_required": False,
    }
    assert candidate == {
        "normalize_features": True,
        "normalizer_required": True,
    }
    assert set(control) == set(candidate)
    changed = {key for key in control if control[key] != candidate[key]}
    assert changed == {"normalize_features", "normalizer_required"}


def test_protocol_binds_runtime_artifact_and_known_scope_mismatch() -> None:
    protocol = expected_ppo_normalization_protocol()

    assert protocol["runtime"] == {
        "python": "3.12",
        "stable_baselines3": "2.3.2",
        "torch": "2.4.1",
        "gymnasium": "0.29.1",
        "device": "cpu",
        "torch_num_threads": 1,
    }
    assert protocol["artifact"] == {
        "schema": "ppo_inference_bundle_v1",
        "raw_normalizer": None,
        "normalized_requires_fitted_normalizer": True,
        "feature_schema_must_match": True,
        "policy_bytes_must_match": True,
    }
    assert protocol["known_common_limitations"] == {
        "training_account_scope": "one_active_symbol_per_account_episode",
        "evaluation_account_scope": "directional_shared_cash_account",
        "account_scope_factor_isolated_elsewhere": True,
    }


def test_protocol_freezes_relative_and_absolute_decision_boundaries() -> None:
    protocol = expected_ppo_normalization_protocol()
    decision = protocol["decision"]

    assert decision["relative"] == {
        "minimum_paired_seed_wins": 4,
        "median_paired_total_return_delta_strictly_positive": True,
        "candidate_intervals_exact": 17_544,
        "candidate_max_drawdown_lte": 0.2,
        "candidate_termination_count_lte_control": True,
        "failure_decision": "KEEP_BASELINE",
    }
    assert decision["absolute"] == {
        "positive_full_return": True,
        "positive_2023_return": True,
        "positive_2024_return": True,
        "max_drawdown_lte": 0.2,
        "complete_intervals": 17_544,
        "no_termination": True,
        "terminal_flat": True,
        "minimum_base_and_stress_seed_passes": 4,
        "positive_family_median_full_and_year_returns": True,
        "stresses": [
            {"cost_multiplier": 2.0, "latency_bars": 0},
            {"cost_multiplier": 1.0, "latency_bars": 1},
        ],
        "stress_year_returns_must_be_positive": False,
        "relative_pass_absolute_fail": "RELATIVE_IMPROVEMENT_ONLY",
        "full_pass": "PROSPECTIVE_PAPER_REQUIRED",
    }
    assert protocol["boundaries"] == {
        "reused_development_evidence": True,
        "historical_645_663_same_economics_control": False,
        "control_new_fit_required": True,
        "candidate_new_fit_required": True,
        "historical_models_reused": False,
        "economic_execution_authorized": False,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def test_protocol_codec_is_canonical_and_fails_closed_on_mutation() -> None:
    expected = expected_ppo_normalization_protocol()
    raw = ppo_normalization_protocol_bytes()

    assert raw == canonical_json_bytes(expected)
    assert content_digest(expected) == content_digest(
        load_ppo_normalization_protocol(raw)
    )

    mutated = json.loads(raw)
    mutated["arms"][CANDIDATE_ARM]["normalize_features"] = False
    with pytest.raises(ValueError, match="frozen"):
        load_ppo_normalization_protocol(canonical_json_bytes(mutated))

    unknown = json.loads(raw)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="frozen"):
        load_ppo_normalization_protocol(canonical_json_bytes(unknown))

    with pytest.raises(ValueError, match="canonical"):
        load_ppo_normalization_protocol(raw + b"\n")
