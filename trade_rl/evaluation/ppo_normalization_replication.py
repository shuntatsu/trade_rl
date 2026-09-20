"""Result-blind protocol for corrected-economics PPO normalization replication."""

from __future__ import annotations

import json
from typing import Final

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.strategies.rl.ppo import ppo_observation_contract_payload

PPO_NORMALIZATION_PROTOCOL_SCHEMA: Final = "ppo_normalization_corrected_replication_v1"
CONTROL_ARM: Final = "control_raw"
CANDIDATE_ARM: Final = "candidate_normalized"

_MINIMUM_MAIN_SHA: Final = "2068dbba481945092a082a79aa640edce8f951b8"
_SOURCE_DATASET_ID: Final = (
    "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
)
_SOURCE_DATASET_ARTIFACT_DIGEST: Final = (
    "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
)
_SOURCE_STUDY_DIGEST: Final = (
    "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"
)

_SOFTWARE_BLOBS: Final = {
    "trade_rl/strategies/rl/ppo.py": "5439e59b5fe08764387e54fc461fb2943c632aa6",
    "trade_rl/strategies/rl/ppo_normalization.py": (
        "3eb4f744a8c8bc40ece4b1b0c508df825a65e352"
    ),
    "trade_rl/strategies/rl/ppo_artifact.py": (
        "8684bd17f7a33e29917ab5a6906ab77bdc969db6"
    ),
    "trade_rl/evaluation/directional_contract.py": (
        "dce26c258e03e6b1658f8c6d43c15e012f5084e0"
    ),
    "trade_rl/evaluation/directional.py": "dfcdb1da984fd9813ae97ffdc0805b69104bfe5c",
    "trade_rl/evaluation/directional_candidates.py": (
        "7e498f4302c0f957238849cee7e960931d26ca64"
    ),
    "pyproject.toml": "6cee9e647951f521aa17a3dd94f0ab1d3ffcc678",
}


def _execution_policy() -> dict[str, object]:
    payload = DIRECTIONAL_BASE_EXECUTION_COST.execution_policy_payload()
    if (
        payload["borrow_rate_multiplier"] != 1.0
        or payload["processing_bar_volume_capacity"] is not False
        or payload["max_leverage"] != 1.0
    ):
        raise RuntimeError(
            "directional execution economics drifted from preregistration"
        )
    return payload


def expected_ppo_normalization_protocol() -> dict[str, object]:
    """Return the frozen result-blind comparison contract."""

    return {
        "schema": PPO_NORMALIZATION_PROTOCOL_SCHEMA,
        "software_authority": {
            "minimum_main_sha": _MINIMUM_MAIN_SHA,
            "source_blobs": dict(_SOFTWARE_BLOBS),
            "corrected_borrow_runtime_merge": (
                "786e7d3f022741c30e5712b4fd50cf48921742ed"
            ),
            "terminal_settlement_merge": ("00b890a2152a2ea41c350eed7add4fcc60a5807e"),
            "artifact_space_validation_merge": (
                "9113beaf84bd59f87e8d6706d2d995403f6fea16"
            ),
            "inference_bundle_merge": ("e7a44750fadcfb6af2fb74cbd4ecbe0a6df5e943"),
            "holdout_mutation_invariance_merge": (
                "7bcc6854b56f52d0dea076763db047dfc2d83976"
            ),
            "strict_artifact_runtime_types_merge": _MINIMUM_MAIN_SHA,
        },
        "source": {
            "dataset_id": _SOURCE_DATASET_ID,
            "dataset_artifact_digest": _SOURCE_DATASET_ARTIFACT_DIGEST,
            "study_digest": _SOURCE_STUDY_DIGEST,
        },
        "factor": "PPO_FIT_ONLY_FEATURE_STANDARDIZATION",
        "common": {
            "symbols": [
                "BTCUSDT",
                "ETHUSDT",
                "BNBUSDT",
                "XRPUSDT",
                "ADAUSDT",
            ],
            "seeds": [0, 1, 2, 3, 4],
            "caller_total_timesteps": 262_144,
            "training_layout": "sequential",
            "rollout_steps_per_env": None,
            "initial_capital": 10_000.0,
            "gross_budget": 0.1,
            "risk_config": None,
            "settle_terminal_position": True,
            "execution_policy": _execution_policy(),
            "development_window": {
                "start_inclusive": "2023-01-01T00:00:00Z",
                "stop_exclusive": "2025-01-01T00:00:00Z",
                "intervals": 17_544,
            },
            "fit_scope": (
                "source_study_baseline_config.feature_indices/fit_symbol_indices/fit_cutoff"
            ),
            "observation": ppo_observation_contract_payload(),
        },
        "runtime": {
            "python": "3.12",
            "stable_baselines3": "2.3.2",
            "torch": "2.4.1",
            "gymnasium": "0.29.1",
            "device": "cpu",
            "torch_num_threads": 1,
        },
        "artifact": {
            "schema": "ppo_inference_bundle_v1",
            "raw_normalizer": None,
            "normalized_requires_fitted_normalizer": True,
            "feature_schema_must_match": True,
            "policy_bytes_must_match": True,
        },
        "arms": {
            CONTROL_ARM: {
                "normalize_features": False,
                "normalizer_required": False,
            },
            CANDIDATE_ARM: {
                "normalize_features": True,
                "normalizer_required": True,
            },
        },
        "known_common_limitations": {
            "training_account_scope": "one_active_symbol_per_account_episode",
            "evaluation_account_scope": "directional_shared_cash_account",
            "account_scope_factor_isolated_elsewhere": True,
        },
        "decision": {
            "relative": {
                "minimum_paired_seed_wins": 4,
                "median_paired_total_return_delta_strictly_positive": True,
                "candidate_intervals_exact": 17_544,
                "candidate_max_drawdown_lte": 0.2,
                "candidate_termination_count_lte_control": True,
                "failure_decision": "KEEP_BASELINE",
            },
            "absolute": {
                "positive_full_return": True,
                "positive_2023_return": True,
                "positive_2024_return": True,
                "max_drawdown_lte": 0.2,
                "complete_intervals": 17_544,
                "no_termination": True,
                "terminal_flat": True,
                "minimum_seed_passes": 4,
                "positive_family_median_full_and_year_returns": True,
                "stresses": [
                    {"cost_multiplier": 2.0},
                    {"latency_bars": 1},
                ],
                "relative_pass_absolute_fail": "RELATIVE_IMPROVEMENT_ONLY",
                "full_pass": "PROSPECTIVE_PAPER_REQUIRED",
            },
        },
        "stop_rule": {
            "result_dependent_rescue_allowed": False,
            "forbidden_rescue_dimensions": [
                "feature_set",
                "normalization_variant",
                "training_layout",
                "risk",
                "reward",
                "network",
                "entropy",
                "learning_rate",
                "seed_roster",
                "timesteps",
                "evaluation_window",
            ],
        },
        "boundaries": {
            "reused_development_evidence": True,
            "historical_645_663_same_economics_control": False,
            "unused_data_accessed": False,
            "final_test_accessed": False,
            "production_eligible": False,
            "live_trading_authorized": False,
        },
    }


def ppo_normalization_protocol_bytes() -> bytes:
    """Serialize the frozen protocol canonically."""

    return canonical_json_bytes(expected_ppo_normalization_protocol())


def load_ppo_normalization_protocol(raw: bytes) -> dict[str, object]:
    """Load only the exact frozen canonical protocol."""

    payload = json.loads(raw)
    expected = expected_ppo_normalization_protocol()
    if payload != expected:
        raise ValueError("protocol differs from the frozen preregistration")
    if raw != canonical_json_bytes(expected):
        raise ValueError("protocol bytes are not canonical")
    return expected


__all__ = [
    "CANDIDATE_ARM",
    "CONTROL_ARM",
    "PPO_NORMALIZATION_PROTOCOL_SCHEMA",
    "expected_ppo_normalization_protocol",
    "load_ppo_normalization_protocol",
    "ppo_normalization_protocol_bytes",
]
