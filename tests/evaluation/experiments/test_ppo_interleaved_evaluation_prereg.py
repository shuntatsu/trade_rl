from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.ppo_interleaved_evaluation_prereg import (
    PPOInterleavedEvaluationProtocol,
    canonical_ppo_interleaved_evaluation_protocol,
    classify_ppo_interleaved_evaluation,
    load_ppo_interleaved_evaluation_protocol,
)


def test_canonical_protocol_freezes_result_blind_interleaved_factor() -> None:
    protocol = canonical_ppo_interleaved_evaluation_protocol()

    assert protocol.issue_number == 629
    assert protocol.capability_issue_number == 621
    assert protocol.capability_pr_number == 624
    assert protocol.current_main_head == "d18434799651cfc6c07e0840c40600dcf1dfa763"
    assert protocol.implementation_head == "cddef3532dd582d0f1066a61006f64265f0cabbe"
    assert protocol.implementation_ci_run_id == 35162829156
    assert protocol.implementation_core_job_id == 105017294291
    assert protocol.implementation_guide_job_id == 105017294531

    assert protocol.successor_bundle_run_id == 34803217815
    assert protocol.successor_bundle_artifact_id == 10331899302
    assert (
        protocol.successor_bundle_artifact_digest
        == "89e899427f23fa29c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
    )
    assert (
        protocol.dataset_id
        == "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
    )
    assert (
        protocol.dataset_artifact_digest
        == "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
    )
    assert (
        protocol.study_digest
        == "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"
    )
    assert protocol.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    assert protocol.ppo_seeds == (0, 1, 2, 3, 4)
    assert protocol.ppo_total_timesteps == 100_000

    assert protocol.controlled_factor == "PPO_TRAINING_LAYOUT"
    assert protocol.semantic_factor == "ppo_interleaved_fixed_symbol_schedule_384"
    assert protocol.observation_schema == "ppo_observation_v2"
    assert protocol.baseline_training_layout == "sequential"
    assert protocol.baseline_rollout_steps_per_env is None
    assert protocol.candidate_training_layout == "interleaved"
    assert protocol.candidate_rollout_steps_per_env == 384
    assert protocol.candidate_n_envs == 5
    assert protocol.candidate_aggregate_rollout_steps == 1_920
    assert protocol.reference_sequential_n_steps == 2_048
    assert protocol.ppo_batch_size == 64
    assert protocol.total_timesteps_may_change is False
    assert protocol.execution_economics_may_change is False
    assert protocol.observation_may_change is False
    assert protocol.reward_may_change is False
    assert protocol.ppo_hyperparameters_may_change is False
    assert protocol.deterministic_execution_required is True
    assert protocol.stochastic_slippage_allowed is False
    assert protocol.actual_model_num_timesteps_is_diagnostic_only is True
    assert protocol.rollout_boundary_granularity_preregistered is True

    assert protocol.baseline_must_follow_protocol_seal is True
    assert protocol.baseline_must_be_sealed_before_candidate is True
    assert protocol.candidate_settings_may_change_after_baseline is False
    assert protocol.no_result_dependent_rescue is True
    assert protocol.no_post_result_rollout_change is True

    assert protocol.stage_a_required_positive_factor_effect_symbols == 5
    assert protocol.stage_a_required_positive_seed_median_excess == 5
    assert protocol.stage_b_required_positive_seed_candidate_medians == 4
    assert protocol.stage_b_required_drawdown_nonworse_symbols == 4

    assert protocol.final_test_accessed is False
    assert protocol.final_test_authorized is False
    assert protocol.shared_cash_profitability_established is False
    assert protocol.operational_eligibility_established is False
    assert protocol.production_eligible is False
    assert protocol.live_trading_authorized is False
    assert protocol.merge_authorized is False

    payload = protocol.to_payload()
    forbidden_tokens = ("total_return", "excess_return", "result_artifact")
    serialized_keys = " ".join(payload).lower()
    assert not any(token in serialized_keys for token in forbidden_tokens)
    assert len(protocol.digest) == 64


def test_protocol_is_immutable_and_rejects_posthoc_parameter_changes() -> None:
    protocol = canonical_ppo_interleaved_evaluation_protocol()

    with pytest.raises(ValueError, match="candidate_rollout_steps_per_env"):
        replace(protocol, candidate_rollout_steps_per_env=448)
    with pytest.raises(ValueError, match="ppo_seeds"):
        replace(protocol, ppo_seeds=(0, 1, 2, 3, 5))
    with pytest.raises(ValueError, match="ppo_total_timesteps"):
        replace(protocol, ppo_total_timesteps=120_000)
    with pytest.raises(ValueError, match="shared_cash_profitability_established"):
        replace(protocol, shared_cash_profitability_established=True)


def test_rollout_choice_is_mechanical_and_nearest_allowed_at_or_below_default() -> None:
    protocol = canonical_ppo_interleaved_evaluation_protocol()

    allowed = tuple(range(protocol.ppo_batch_size, 2_049, protocol.ppo_batch_size))
    feasible = tuple(
        value
        for value in allowed
        if (value * protocol.candidate_n_envs) % protocol.ppo_batch_size == 0
        and value * protocol.candidate_n_envs <= protocol.reference_sequential_n_steps
    )
    chosen = protocol.candidate_rollout_steps_per_env
    assert chosen is not None
    assert chosen in feasible
    assert chosen * protocol.candidate_n_envs == 1_920
    assert (
        min(
            feasible,
            key=lambda value: (
                protocol.reference_sequential_n_steps
                - value * protocol.candidate_n_envs
            ),
        )
        == chosen
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                evidence_valid=True,
                positive_factor_effect_symbols=5,
                cross_symbol_median_factor_effect=0.01,
                positive_seed_median_excess=5,
                new_termination_cells=0,
                cross_symbol_median_candidate_return=0.02,
                positive_seed_candidate_medians=4,
                drawdown_nonworse_symbols=4,
            ),
            "PROMOTE_TO_SHARED_CASH_EVALUATION",
        ),
        (
            dict(
                evidence_valid=True,
                positive_factor_effect_symbols=5,
                cross_symbol_median_factor_effect=0.01,
                positive_seed_median_excess=5,
                new_termination_cells=0,
                cross_symbol_median_candidate_return=-0.01,
                positive_seed_candidate_medians=5,
                drawdown_nonworse_symbols=5,
            ),
            "PROMOTE_RESEARCH_REFERENCE_ONLY",
        ),
        (
            dict(
                evidence_valid=True,
                positive_factor_effect_symbols=4,
                cross_symbol_median_factor_effect=0.01,
                positive_seed_median_excess=5,
                new_termination_cells=0,
                cross_symbol_median_candidate_return=0.02,
                positive_seed_candidate_medians=5,
                drawdown_nonworse_symbols=5,
            ),
            "KEEP_SEQUENTIAL_BASELINE",
        ),
        (
            dict(
                evidence_valid=False,
                positive_factor_effect_symbols=5,
                cross_symbol_median_factor_effect=0.01,
                positive_seed_median_excess=5,
                new_termination_cells=0,
                cross_symbol_median_candidate_return=0.02,
                positive_seed_candidate_medians=5,
                drawdown_nonworse_symbols=5,
            ),
            "INVALID",
        ),
    ],
)
def test_frozen_two_stage_decision_rule(
    kwargs: dict[str, bool | int | float], expected: str
) -> None:
    assert classify_ppo_interleaved_evaluation(**kwargs) == expected


def test_decision_rule_fails_closed_on_bool_count_alias_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="positive_factor_effect_symbols"):
        classify_ppo_interleaved_evaluation(
            evidence_valid=True,
            positive_factor_effect_symbols=True,  # type: ignore[arg-type]
            cross_symbol_median_factor_effect=0.01,
            positive_seed_median_excess=5,
            new_termination_cells=0,
            cross_symbol_median_candidate_return=0.02,
            positive_seed_candidate_medians=5,
            drawdown_nonworse_symbols=5,
        )
    with pytest.raises(ValueError, match="cross_symbol_median_factor_effect"):
        classify_ppo_interleaved_evaluation(
            evidence_valid=True,
            positive_factor_effect_symbols=5,
            cross_symbol_median_factor_effect=float("nan"),
            positive_seed_median_excess=5,
            new_termination_cells=0,
            cross_symbol_median_candidate_return=0.02,
            positive_seed_candidate_medians=5,
            drawdown_nonworse_symbols=5,
        )


def test_loader_requires_exact_canonical_protocol_bytes(tmp_path: Path) -> None:
    protocol = canonical_ppo_interleaved_evaluation_protocol()
    path = tmp_path / "protocol.json"
    path.write_bytes(canonical_json_bytes(protocol.to_payload()))

    loaded = load_ppo_interleaved_evaluation_protocol(path)
    assert isinstance(loaded, PPOInterleavedEvaluationProtocol)
    assert loaded == protocol

    mutated = protocol.to_payload()
    mutated["candidate_rollout_steps_per_env"] = 448
    path.write_bytes(canonical_json_bytes(mutated))
    with pytest.raises(ValueError, match="protocol values"):
        load_ppo_interleaved_evaluation_protocol(path)
