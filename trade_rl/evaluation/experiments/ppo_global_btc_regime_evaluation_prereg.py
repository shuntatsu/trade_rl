"""Result-blind evaluation authority for the sealed PPO global-BTC factor."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "ppo_global_btc_regime_evaluation_prereg_v1"
_ISSUE_NUMBER = 609
_FACTOR_ISSUE_NUMBER = 605
_IMPLEMENTATION_ISSUE_NUMBER = 607
_FACTOR_PREREG_HEAD = "d84235185c5e79327ea7ced7073f0026970ba46b"
_FACTOR_PREREG_PROTOCOL_DIGEST = (
    "65a803034fca8180c165728022221d7b3c2fa5dcdd8349d3ca52c3a6adc72228"
)
_FACTOR_PREREG_SEAL_RUN_ID = 35_078_049_585
_FACTOR_PREREG_SEAL_ARTIFACT_ID = 10_439_281_380
_FACTOR_PREREG_SEAL_ARTIFACT_DIGEST = (
    "sha256:b030ecc638c440fafcbee27cc45fe0eb0b242309c5bb6ddaaaf32980894e2859"
)
_FACTOR_PREREG_FRESH_ARTIFACT_ID = 10_438_184_487
_FACTOR_PREREG_FRESH_ARTIFACT_DIGEST = (
    "sha256:1a405428a709d8994a0f70d16697be249a664f57afaa74732102c2a8ea32fc69"
)
_IMPLEMENTATION_HEAD = "1ae494ce182189c5be61e8007dc956f912d8b9d5"
_IMPLEMENTATION_TREE = "4f14c5a06505b8ff9811b473bf6dea296453a13b"
_IMPLEMENTATION_INDEX_DIGEST = (
    "d5390a0025a92e4e2a1e987b7b7ac8ee6ef3974c5c8247b0256ae70c0d2d6e10"
)
_IMPLEMENTATION_SEAL_RUN_ID = 35_084_553_267
_IMPLEMENTATION_SEAL_ARTIFACT_ID = 10_441_133_239
_IMPLEMENTATION_SEAL_ARTIFACT_DIGEST = (
    "sha256:8f8a3319f9213bf999b3ce2d911882fd9276daf575a7cde99ec33de9e8208b37"
)
_IMPLEMENTATION_FRESH_ARTIFACT_ID = 10_441_194_709
_IMPLEMENTATION_FRESH_ARTIFACT_DIGEST = (
    "sha256:39f578321699d3deba1c1cc67ee9d5401c7a45573088d63b2c346f09721c7f52"
)
_SOURCE_DATASET_ID = "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
_SOURCE_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
_SOURCE_STUDY_DIGEST = (
    "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
)
_BASELINE_EVIDENCE_FINGERPRINT = (
    "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
)
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_PPO_SEEDS = (0, 1, 2, 3, 4)
_UNAFFECTED_STRATEGY_NAMES = (
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "cash",
    "constant_long",
    "constant_short",
)
_DEVELOPMENT_ACCEPTANCE_GATE_METRICS = (
    "positive_factor_effect_symbols",
    "cross_symbol_median_factor_effect_gt_zero",
    "positive_cross_symbol_median_seeds",
    "no_new_termination",
    "unaffected_raw_returns_equal",
)


@dataclass(frozen=True, slots=True)
class PPOGlobalBTCRegimeEvaluationProtocol:
    """Immutable result-blind decision authority for PR 609."""

    schema_version: str = _SCHEMA_VERSION
    issue_number: int = _ISSUE_NUMBER
    factor_issue_number: int = _FACTOR_ISSUE_NUMBER
    implementation_issue_number: int = _IMPLEMENTATION_ISSUE_NUMBER

    factor_prereg_head: str = _FACTOR_PREREG_HEAD
    factor_prereg_protocol_digest: str = _FACTOR_PREREG_PROTOCOL_DIGEST
    factor_prereg_seal_run_id: int = _FACTOR_PREREG_SEAL_RUN_ID
    factor_prereg_seal_artifact_id: int = _FACTOR_PREREG_SEAL_ARTIFACT_ID
    factor_prereg_seal_artifact_digest: str = _FACTOR_PREREG_SEAL_ARTIFACT_DIGEST
    factor_prereg_fresh_artifact_id: int = _FACTOR_PREREG_FRESH_ARTIFACT_ID
    factor_prereg_fresh_artifact_digest: str = _FACTOR_PREREG_FRESH_ARTIFACT_DIGEST

    implementation_head: str = _IMPLEMENTATION_HEAD
    implementation_tree: str = _IMPLEMENTATION_TREE
    implementation_index_digest: str = _IMPLEMENTATION_INDEX_DIGEST
    implementation_seal_run_id: int = _IMPLEMENTATION_SEAL_RUN_ID
    implementation_seal_artifact_id: int = _IMPLEMENTATION_SEAL_ARTIFACT_ID
    implementation_seal_artifact_digest: str = _IMPLEMENTATION_SEAL_ARTIFACT_DIGEST
    implementation_fresh_artifact_id: int = _IMPLEMENTATION_FRESH_ARTIFACT_ID
    implementation_fresh_artifact_digest: str = _IMPLEMENTATION_FRESH_ARTIFACT_DIGEST

    source_dataset_id: str = _SOURCE_DATASET_ID
    source_dataset_artifact_digest: str = _SOURCE_DATASET_ARTIFACT_DIGEST
    source_study_digest: str = _SOURCE_STUDY_DIGEST
    baseline_evidence_fingerprint: str = _BASELINE_EVIDENCE_FINGERPRINT

    controlled_factor: str = "FEATURE_SET"
    semantic_factor: str = "ppo_global_btc_regime_context"
    baseline_observation_schema: str = "ppo_observation_v2"
    candidate_observation_schema: str = "ppo_observation_v3_global_btc_regime"
    comparison_schema: str = "controlled_evidence_comparison_v2"
    symbols: tuple[str, ...] = _SYMBOLS
    ppo_seeds: tuple[int, ...] = _PPO_SEEDS
    ppo_total_timesteps: int = 100_000
    n_bootstrap: int = 2_000
    bootstrap_seed: int = 1_729
    unaffected_strategy_names: tuple[str, ...] = _UNAFFECTED_STRATEGY_NAMES

    unaffected_raw_return_invariance_required: bool = True
    only_candidate_ppo_observation_may_change: bool = True
    source_dataset_may_change: bool = False
    development_window_may_change: bool = False
    ppo_hyperparameters_may_change: bool = False
    execution_economics_may_change: bool = False

    factor_effect_statistic: str = "per_symbol_median_matched_seed_excess_total_return"
    candidate_return_statistic: str = (
        "per_symbol_median_candidate_total_return_across_matched_seeds"
    )
    seed_robustness_statistic: str = "per_seed_cross_symbol_median_excess_total_return"
    positive_threshold: float = 0.0
    positive_comparison: str = "strictly_greater_than"
    termination_rule: str = (
        "candidate_must_not_introduce_new_hard_or_economic_termination"
    )
    acceptance_rule_conjunction: str = "all_conditions_required"

    development_acceptance_scope: str = "robust_factor_improvement"
    development_acceptance_gate_metrics: tuple[str, ...] = (
        _DEVELOPMENT_ACCEPTANCE_GATE_METRICS
    )
    accept_requires_positive_factor_effect_symbols: int = 5
    accept_requires_positive_cross_symbol_median_seeds: int = 5
    accept_requires_cross_symbol_median_factor_effect_gt_zero: bool = True
    accept_requires_no_new_termination: bool = True
    accept_requires_all_unaffected_raw_returns_equal: bool = True
    candidate_profitability_statistics_are_diagnostic_only: bool = True
    development_acceptance_establishes_profitability: bool = False
    operational_eligibility_established: bool = False
    valid_non_accept_decision: str = "KEEP_BASELINE"
    invalid_decision: str = "INVALID"
    bootstrap_statistics_are_diagnostic_only: bool = True
    no_result_dependent_rescue: bool = True
    no_post_result_threshold_change: bool = True
    no_post_result_metric_substitution: bool = True

    economic_execution_authorized: bool = False
    economic_result_inspected: bool = False
    final_test_access_authorized: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False
    merge_authorized: bool = False

    def __post_init__(self) -> None:
        canonical = _canonical_field_values()
        for field in fields(self):
            actual = getattr(self, field.name)
            expected = canonical[field.name]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"{field.name} differs from preregistered Issue 609 authority"
                )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            payload[field.name] = list(value) if isinstance(value, tuple) else value
        return payload

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _canonical_field_values() -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "issue_number": _ISSUE_NUMBER,
        "factor_issue_number": _FACTOR_ISSUE_NUMBER,
        "implementation_issue_number": _IMPLEMENTATION_ISSUE_NUMBER,
        "factor_prereg_head": _FACTOR_PREREG_HEAD,
        "factor_prereg_protocol_digest": _FACTOR_PREREG_PROTOCOL_DIGEST,
        "factor_prereg_seal_run_id": _FACTOR_PREREG_SEAL_RUN_ID,
        "factor_prereg_seal_artifact_id": _FACTOR_PREREG_SEAL_ARTIFACT_ID,
        "factor_prereg_seal_artifact_digest": _FACTOR_PREREG_SEAL_ARTIFACT_DIGEST,
        "factor_prereg_fresh_artifact_id": _FACTOR_PREREG_FRESH_ARTIFACT_ID,
        "factor_prereg_fresh_artifact_digest": _FACTOR_PREREG_FRESH_ARTIFACT_DIGEST,
        "implementation_head": _IMPLEMENTATION_HEAD,
        "implementation_tree": _IMPLEMENTATION_TREE,
        "implementation_index_digest": _IMPLEMENTATION_INDEX_DIGEST,
        "implementation_seal_run_id": _IMPLEMENTATION_SEAL_RUN_ID,
        "implementation_seal_artifact_id": _IMPLEMENTATION_SEAL_ARTIFACT_ID,
        "implementation_seal_artifact_digest": _IMPLEMENTATION_SEAL_ARTIFACT_DIGEST,
        "implementation_fresh_artifact_id": _IMPLEMENTATION_FRESH_ARTIFACT_ID,
        "implementation_fresh_artifact_digest": _IMPLEMENTATION_FRESH_ARTIFACT_DIGEST,
        "source_dataset_id": _SOURCE_DATASET_ID,
        "source_dataset_artifact_digest": _SOURCE_DATASET_ARTIFACT_DIGEST,
        "source_study_digest": _SOURCE_STUDY_DIGEST,
        "baseline_evidence_fingerprint": _BASELINE_EVIDENCE_FINGERPRINT,
        "controlled_factor": "FEATURE_SET",
        "semantic_factor": "ppo_global_btc_regime_context",
        "baseline_observation_schema": "ppo_observation_v2",
        "candidate_observation_schema": "ppo_observation_v3_global_btc_regime",
        "comparison_schema": "controlled_evidence_comparison_v2",
        "symbols": _SYMBOLS,
        "ppo_seeds": _PPO_SEEDS,
        "ppo_total_timesteps": 100_000,
        "n_bootstrap": 2_000,
        "bootstrap_seed": 1_729,
        "unaffected_strategy_names": _UNAFFECTED_STRATEGY_NAMES,
        "unaffected_raw_return_invariance_required": True,
        "only_candidate_ppo_observation_may_change": True,
        "source_dataset_may_change": False,
        "development_window_may_change": False,
        "ppo_hyperparameters_may_change": False,
        "execution_economics_may_change": False,
        "factor_effect_statistic": (
            "per_symbol_median_matched_seed_excess_total_return"
        ),
        "candidate_return_statistic": (
            "per_symbol_median_candidate_total_return_across_matched_seeds"
        ),
        "seed_robustness_statistic": (
            "per_seed_cross_symbol_median_excess_total_return"
        ),
        "positive_threshold": 0.0,
        "positive_comparison": "strictly_greater_than",
        "termination_rule": (
            "candidate_must_not_introduce_new_hard_or_economic_termination"
        ),
        "acceptance_rule_conjunction": "all_conditions_required",
        "development_acceptance_scope": "robust_factor_improvement",
        "development_acceptance_gate_metrics": _DEVELOPMENT_ACCEPTANCE_GATE_METRICS,
        "accept_requires_positive_factor_effect_symbols": 5,
        "accept_requires_positive_cross_symbol_median_seeds": 5,
        "accept_requires_cross_symbol_median_factor_effect_gt_zero": True,
        "accept_requires_no_new_termination": True,
        "accept_requires_all_unaffected_raw_returns_equal": True,
        "candidate_profitability_statistics_are_diagnostic_only": True,
        "development_acceptance_establishes_profitability": False,
        "operational_eligibility_established": False,
        "valid_non_accept_decision": "KEEP_BASELINE",
        "invalid_decision": "INVALID",
        "bootstrap_statistics_are_diagnostic_only": True,
        "no_result_dependent_rescue": True,
        "no_post_result_threshold_change": True,
        "no_post_result_metric_substitution": True,
        "economic_execution_authorized": False,
        "economic_result_inspected": False,
        "final_test_access_authorized": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }


def canonical_ppo_global_btc_regime_evaluation_protocol() -> (
    PPOGlobalBTCRegimeEvaluationProtocol
):
    return PPOGlobalBTCRegimeEvaluationProtocol()


def load_ppo_global_btc_regime_evaluation_protocol(
    path: Path,
) -> PPOGlobalBTCRegimeEvaluationProtocol:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    raw = path.read_bytes()
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("preregistered protocol JSON is malformed") from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("preregistered protocol must be a JSON object")

    expected = canonical_ppo_global_btc_regime_evaluation_protocol().to_payload()
    if set(decoded) != set(expected):
        raise ValueError("preregistered protocol keys are unknown or missing")
    if raw != canonical_json_bytes(decoded):
        raise ValueError("preregistered protocol must use canonical JSON bytes")
    if raw != canonical_json_bytes(expected):
        raise ValueError("protocol values differ from preregistered authority")
    return canonical_ppo_global_btc_regime_evaluation_protocol()


__all__ = [
    "PPOGlobalBTCRegimeEvaluationProtocol",
    "canonical_ppo_global_btc_regime_evaluation_protocol",
    "load_ppo_global_btc_regime_evaluation_protocol",
]
