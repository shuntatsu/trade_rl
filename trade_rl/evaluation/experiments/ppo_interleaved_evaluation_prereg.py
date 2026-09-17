"""Result-blind authority for calibrated PPO interleaved-layout evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "ppo_interleaved_evaluation_prereg_v1"
_ISSUE_NUMBER = 629
_CAPABILITY_ISSUE_NUMBER = 621
_CAPABILITY_PR_NUMBER = 624
_CURRENT_MAIN_HEAD = "d18434799651cfc6c07e0840c40600dcf1dfa763"
_IMPLEMENTATION_HEAD = "cddef3532dd582d0f1066a61006f64265f0cabbe"
_IMPLEMENTATION_CI_RUN_ID = 35_162_829_156
_IMPLEMENTATION_CORE_JOB_ID = 105_017_294_291
_IMPLEMENTATION_GUIDE_JOB_ID = 105_017_294_531

_SUCCESSOR_BUNDLE_RUN_ID = 34_803_217_815
_SUCCESSOR_BUNDLE_ARTIFACT_ID = 10_331_899_302
_SUCCESSOR_BUNDLE_ARTIFACT_DIGEST = (
    "89e899427f23fa29c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
)
_DATASET_ID = "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
_DATASET_ARTIFACT_DIGEST = (
    "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
)
_STUDY_DIGEST = "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"
_EXECUTION_OVERLAY = (
    "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity"
)
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_PPO_SEEDS = (0, 1, 2, 3, 4)


def _bounded_count(value: object, *, field: str, upper: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= upper
    ):
        raise ValueError(f"{field} must be an integer within [0, {upper}]")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PPOInterleavedEvaluationProtocol:
    """Immutable pre-result authority for Issue 629."""

    schema_version: str = _SCHEMA_VERSION
    issue_number: int = _ISSUE_NUMBER
    capability_issue_number: int = _CAPABILITY_ISSUE_NUMBER
    capability_pr_number: int = _CAPABILITY_PR_NUMBER
    current_main_head: str = _CURRENT_MAIN_HEAD
    implementation_head: str = _IMPLEMENTATION_HEAD
    implementation_ci_run_id: int = _IMPLEMENTATION_CI_RUN_ID
    implementation_core_job_id: int = _IMPLEMENTATION_CORE_JOB_ID
    implementation_guide_job_id: int = _IMPLEMENTATION_GUIDE_JOB_ID

    successor_bundle_run_id: int = _SUCCESSOR_BUNDLE_RUN_ID
    successor_bundle_artifact_id: int = _SUCCESSOR_BUNDLE_ARTIFACT_ID
    successor_bundle_artifact_digest: str = _SUCCESSOR_BUNDLE_ARTIFACT_DIGEST
    dataset_id: str = _DATASET_ID
    dataset_artifact_digest: str = _DATASET_ARTIFACT_DIGEST
    study_digest: str = _STUDY_DIGEST
    execution_overlay: str = _EXECUTION_OVERLAY
    symbols: tuple[str, ...] = _SYMBOLS
    fit_cutoff: str = "2023-01-01T00:00:00Z"
    evaluation_start: str = "2023-01-01T00:00:00Z"
    evaluation_stop_exclusive: str = "2025-01-01T00:00:00Z"

    controlled_factor: str = "PPO_TRAINING_LAYOUT"
    semantic_factor: str = "ppo_interleaved_fixed_symbol_schedule_384"
    observation_schema: str = "ppo_observation_v2"
    ppo_seeds: tuple[int, ...] = _PPO_SEEDS
    ppo_total_timesteps: int = 100_000
    policy_name: str = "MlpPolicy"
    ent_coef: float = 0.0

    baseline_training_layout: str = "sequential"
    baseline_rollout_steps_per_env: None = None
    candidate_training_layout: str = "interleaved"
    candidate_rollout_steps_per_env: int = 384
    candidate_n_envs: int = 5
    candidate_aggregate_rollout_steps: int = 1_920
    reference_sequential_n_steps: int = 2_048
    ppo_batch_size: int = 64
    rollout_selection_rule: str = (
        "closest_batch_divisible_aggregate_at_or_below_sequential_default"
    )
    stable_baselines3_version: str = "2.3.2"
    torch_version: str = "2.4.1"

    total_timesteps_may_change: bool = False
    observation_may_change: bool = False
    reward_may_change: bool = False
    ppo_hyperparameters_may_change: bool = False
    execution_economics_may_change: bool = False
    symbol_roster_may_change: bool = False
    seed_roster_may_change: bool = False
    development_window_may_change: bool = False
    deterministic_execution_required: bool = True
    stochastic_slippage_allowed: bool = False
    actual_model_num_timesteps_is_diagnostic_only: bool = True
    rollout_boundary_granularity_preregistered: bool = True

    baseline_must_follow_protocol_seal: bool = True
    baseline_must_be_sealed_before_candidate: bool = True
    candidate_settings_may_change_after_baseline: bool = False
    baseline_result_may_select_candidate_settings: bool = False
    baseline_training_authorized: bool = False
    candidate_training_authorized: bool = False
    economic_result_inspected: bool = False

    factor_effect_statistic: str = (
        "per_symbol_median_matched_seed_excess_total_return"
    )
    seed_robustness_statistic: str = "per_seed_cross_symbol_median_excess_total_return"
    candidate_profitability_statistic: str = (
        "cross_symbol_median_of_per_symbol_median_candidate_total_return"
    )
    candidate_seed_profitability_statistic: str = (
        "per_seed_cross_symbol_median_candidate_total_return"
    )
    drawdown_statistic: str = "per_symbol_median_max_drawdown_across_matched_seeds"
    positive_threshold: float = 0.0
    stage_a_required_positive_factor_effect_symbols: int = 5
    stage_a_required_positive_seed_median_excess: int = 5
    stage_a_requires_cross_symbol_median_factor_effect_gt_zero: bool = True
    stage_a_requires_no_new_termination: bool = True
    stage_b_requires_cross_symbol_median_candidate_return_gt_zero: bool = True
    stage_b_required_positive_seed_candidate_medians: int = 4
    stage_b_required_drawdown_nonworse_symbols: int = 4
    stage_b_requires_no_new_termination: bool = True
    invalid_decision: str = "INVALID"
    stage_a_failure_decision: str = "KEEP_SEQUENTIAL_BASELINE"
    stage_b_failure_decision: str = "PROMOTE_RESEARCH_REFERENCE_ONLY"
    stage_b_success_decision: str = "PROMOTE_TO_SHARED_CASH_EVALUATION"

    policy_mode_diagnostics_required: bool = True
    policy_mode_diagnostics_are_decision_metrics: bool = False
    bootstrap_statistics_are_diagnostic_only: bool = True
    no_result_dependent_rescue: bool = True
    no_post_result_rollout_change: bool = True
    no_post_result_seed_or_symbol_change: bool = True
    no_post_result_hyperparameter_change: bool = True

    final_test_accessed: bool = False
    final_test_authorized: bool = False
    shared_cash_profitability_established: bool = False
    operational_eligibility_established: bool = False
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
                    f"{field.name} differs from preregistered Issue 629 authority"
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
        field.name: field.default
        for field in fields(PPOInterleavedEvaluationProtocol)
    }


def canonical_ppo_interleaved_evaluation_protocol() -> (
    PPOInterleavedEvaluationProtocol
):
    return PPOInterleavedEvaluationProtocol()


def classify_ppo_interleaved_evaluation(
    *,
    evidence_valid: bool,
    positive_factor_effect_symbols: int,
    cross_symbol_median_factor_effect: float,
    positive_seed_median_excess: int,
    new_termination_cells: int,
    cross_symbol_median_candidate_return: float,
    positive_seed_candidate_medians: int,
    drawdown_nonworse_symbols: int,
) -> str:
    """Replay the frozen two-stage Issue 629 decision rule."""

    if type(evidence_valid) is not bool:
        raise ValueError("evidence_valid must be a boolean")
    positive_effects = _bounded_count(
        positive_factor_effect_symbols,
        field="positive_factor_effect_symbols",
        upper=5,
    )
    factor_median = _finite_float(
        cross_symbol_median_factor_effect,
        field="cross_symbol_median_factor_effect",
    )
    positive_seed_effects = _bounded_count(
        positive_seed_median_excess,
        field="positive_seed_median_excess",
        upper=5,
    )
    terminations = _bounded_count(
        new_termination_cells,
        field="new_termination_cells",
        upper=25,
    )
    candidate_median = _finite_float(
        cross_symbol_median_candidate_return,
        field="cross_symbol_median_candidate_return",
    )
    positive_candidate_seeds = _bounded_count(
        positive_seed_candidate_medians,
        field="positive_seed_candidate_medians",
        upper=5,
    )
    drawdown_nonworse = _bounded_count(
        drawdown_nonworse_symbols,
        field="drawdown_nonworse_symbols",
        upper=5,
    )

    if not evidence_valid:
        return "INVALID"

    robust_factor = (
        positive_effects == 5
        and factor_median > 0.0
        and positive_seed_effects == 5
        and terminations == 0
    )
    if not robust_factor:
        return "KEEP_SEQUENTIAL_BASELINE"

    profitability_and_risk = (
        candidate_median > 0.0
        and positive_candidate_seeds >= 4
        and drawdown_nonworse >= 4
        and terminations == 0
    )
    if profitability_and_risk:
        return "PROMOTE_TO_SHARED_CASH_EVALUATION"
    return "PROMOTE_RESEARCH_REFERENCE_ONLY"


def load_ppo_interleaved_evaluation_protocol(
    path: Path,
) -> PPOInterleavedEvaluationProtocol:
    """Load only exact canonical bytes for the frozen protocol."""

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

    expected = canonical_ppo_interleaved_evaluation_protocol().to_payload()
    if set(decoded) != set(expected):
        raise ValueError("preregistered protocol keys are unknown or missing")
    if raw != canonical_json_bytes(decoded):
        raise ValueError("preregistered protocol must use canonical JSON bytes")
    if raw != canonical_json_bytes(expected):
        raise ValueError("protocol values differ from preregistered authority")
    return canonical_ppo_interleaved_evaluation_protocol()


__all__ = [
    "PPOInterleavedEvaluationProtocol",
    "canonical_ppo_interleaved_evaluation_protocol",
    "classify_ppo_interleaved_evaluation",
    "load_ppo_interleaved_evaluation_protocol",
]
