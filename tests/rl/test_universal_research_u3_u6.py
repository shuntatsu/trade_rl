from __future__ import annotations

import numpy as np
import pytest

from trade_rl.evaluation.universal_zero_shot import (
    UniversalZeroShotPair,
    passes_zero_shot_gate,
    summarize_zero_shot_pairs,
    zero_shot_bootstrap,
)
from trade_rl.integrations.binance_universal import binance_universal_feature_specs
from trade_rl.learning.universal_bc import (
    CriticWarmStartPhase,
    SymbolBalancedBatchSampler,
    UniversalTeacherArtifact,
)
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    apply_architecture_to_training_config,
    architecture_spec,
)
from trade_rl.rl.universal_normalization import SymbolBalancedStandardNormalizer
from trade_rl.workflows.universal_research import (
    FullResearchAlgorithm,
    UniversalFullResearchPlan,
    UniversalResearchManifest,
    validate_full_research_inputs,
)


def test_universal_binance_contract_has_206_target_local_features() -> None:
    features = binance_universal_feature_specs(
        base_timeframe="15m",
        feature_timeframes=("1h", "4h", "1d"),
    )
    assert len(features) == 206
    assert not any("relative_return_to_btc" in feature.name for feature in features)
    assert not any("rolling_beta_to_btc" in feature.name for feature in features)


def test_symbol_balanced_normalizer_uses_only_fold_and_weights_symbols_equally() -> (
    None
):
    normalizer = SymbolBalancedStandardNormalizer.fit(
        {
            "AAAUSDT": np.asarray([[0.0], [2.0], [9_999.0], [9_999.0]]),
            "BBBUSDT": np.asarray([[100.0], [102.0], [-9_999.0], [-9_999.0]]),
        },
        train_symbols=("AAAUSDT", "BBBUSDT"),
        feature_schema_digest="features-v1",
        catalog_digest="catalog-v1",
        split_manifest_digest="split-v1",
        fold_train_range=(0, 2),
        max_samples_per_symbol=2,
    )
    assert normalizer.sample_count_per_symbol == 2
    assert normalizer.train_symbols == ("AAAUSDT", "BBBUSDT")
    assert normalizer.mean[0] == pytest.approx(51.0)
    transformed = normalizer.transform(np.asarray([[-1_000.0], [1_000.0]]))
    assert transformed.min() >= -10.0
    assert transformed.max() <= 10.0


def test_symbol_balanced_normalizer_handles_missing_values_per_feature() -> None:
    normalizer = SymbolBalancedStandardNormalizer.fit(
        {
            "A": np.asarray([[0.0, np.nan], [2.0, 10.0], [4.0, 12.0]]),
            "B": np.asarray([[100.0, 20.0], [102.0, np.nan], [104.0, 24.0]]),
        },
        train_symbols=("A", "B"),
        feature_schema_digest="features-v1",
        catalog_digest="catalog-v1",
        split_manifest_digest="split-v1",
        fold_train_range=(0, 3),
        max_samples_per_symbol=2,
    )
    assert normalizer.sample_count_per_feature == (2, 2)
    assert np.isfinite(normalizer.mean).all()
    assert np.isfinite(normalizer.std).all()


def test_symbol_balanced_normalizer_rejects_validation_symbol_in_fit_scope() -> None:
    with pytest.raises(ValueError, match="exactly match train_symbols"):
        SymbolBalancedStandardNormalizer.fit(
            {
                "TRAIN": np.ones((3, 2)),
                "VALIDATION": np.ones((3, 2)),
            },
            train_symbols=("TRAIN",),
            feature_schema_digest="features-v1",
            catalog_digest="catalog-v1",
            split_manifest_digest="split-v1",
            fold_train_range=(0, 3),
        )


def test_symbol_balanced_batch_sampler_is_deterministic_and_balanced() -> None:
    sampler = SymbolBalancedBatchSampler(
        sample_indices={"A": tuple(range(6)), "B": tuple(range(10, 16))},
        seed=7,
    )
    first = sampler.batch(batch_size=4, batch_index=3)
    second = sampler.batch(batch_size=4, batch_index=3)
    assert first == second
    assert [symbol for symbol, _ in first].count("A") == 2
    assert [symbol for symbol, _ in first].count("B") == 2


def test_universal_teacher_artifact_rejects_non_train_symbol() -> None:
    with pytest.raises(ValueError, match="teacher symbols must equal train symbols"):
        UniversalTeacherArtifact.create(
            teacher_digest="teacher",
            train_symbols=("A", "B"),
            teacher_symbols=("A", "B", "C"),
            normalizer_digest="normalizer",
            feature_schema_digest="features",
        )


def test_critic_warm_start_phase_order_is_explicit() -> None:
    assert CriticWarmStartPhase.ordered() == (
        CriticWarmStartPhase.CRITIC_ONLY,
        CriticWarmStartPhase.JOINT_FINE_TUNE,
    )


def test_architecture_candidates_have_fixed_universal_contract() -> None:
    expected = {
        UniversalArchitectureName.U_SMALL_DIRECT: (192, "shared_target_v1"),
        UniversalArchitectureName.U_MEDIUM_DIRECT: (256, "shared_target_v1"),
        UniversalArchitectureName.U_MEDIUM_GATE: (256, "hierarchical_gate_target_v1"),
        UniversalArchitectureName.U_LARGE_DIRECT: (336, "shared_target_v1"),
    }
    for name, (d_model, head) in expected.items():
        spec = architecture_spec(name)
        assert spec.d_model == d_model
        assert spec.actor_head == head
        assert spec.action_shape == (1,)
        assert spec.sequence_dropout == 0.0


def test_architecture_candidate_projects_into_training_config() -> None:
    base = ResidualTrainingConfig(timesteps=128, gamma=1.0, seeds=(1,))
    resolved = apply_architecture_to_training_config(
        base,
        UniversalArchitectureName.U_MEDIUM_GATE,
    )
    assert resolved.observation_encoder == "hierarchical_sequence_v2"
    assert resolved.sequence_tcn_capacity == "compact"
    assert resolved.sequence_d_model == 256
    assert resolved.sequence_timeframe_attention_heads == 4
    assert resolved.sequence_timeframe_attention_layers == 1
    assert resolved.sequence_dropout == 0.0
    assert resolved.policy_actor_head == "hierarchical_gate_target_v1"
    assert resolved.policy_net_arch == (256, 128)
    assert resolved.value_net_arch == (256, 128)


def _positive_zero_shot_pairs() -> tuple[UniversalZeroShotPair, ...]:
    return (
        UniversalZeroShotPair("X", 0, 0, 0.05, 0.01),
        UniversalZeroShotPair("Y", 0, 0, 0.04, 0.01),
        UniversalZeroShotPair("X", 1, 1, 0.03, 0.01),
        UniversalZeroShotPair("Y", 1, 1, 0.02, 0.01),
    )


def test_zero_shot_gate_uses_symbol_seed_and_safety_worst_cases() -> None:
    pairs = _positive_zero_shot_pairs()
    summary = summarize_zero_shot_pairs(pairs)
    bootstrap = zero_shot_bootstrap(pairs, n_bootstrap=200, seed=9, block_size=2)
    assert summary.worst_symbol_excess_return > 0.0
    assert summary.worst_seed_excess_return > 0.0
    assert passes_zero_shot_gate(summary, bootstrap=bootstrap)
    with pytest.raises(ValueError, match="bootstrap evidence"):
        passes_zero_shot_gate(summary, bootstrap=None)


def test_zero_shot_bootstrap_reports_paired_excess_lower_bound() -> None:
    result = zero_shot_bootstrap(
        _positive_zero_shot_pairs(),
        n_bootstrap=200,
        seed=9,
        block_size=2,
    )
    assert result.lower_ci > 0.0
    assert result.block_size == 2


def test_full_research_validation_fails_closed_on_missing_pair() -> None:
    manifest = UniversalResearchManifest(
        catalog_digest="catalog",
        split_manifest_digest="split",
        normalizer_digest="normalizer",
        feature_schema_digest="features",
        seed_manifest_digest="seeds",
        architecture_name=UniversalArchitectureName.U_MEDIUM_DIRECT,
        checkpoint_digest="checkpoint",
        cost_model_digest="costs",
        required_pairs=("candidate:baseline:fold0:seed0",),
        completed_pairs=(),
    )
    with pytest.raises(ValueError, match="paired deliverables"):
        validate_full_research_inputs(manifest)


def test_u6_requires_zero_shot_selection_before_full_algorithms() -> None:
    with pytest.raises(ValueError, match="zero-shot gate"):
        UniversalFullResearchPlan.create(
            selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
            zero_shot_gate_passed=False,
            algorithms=(
                FullResearchAlgorithm.PPO,
                FullResearchAlgorithm.LAGRANGIAN,
                FullResearchAlgorithm.DISCOUNTED,
            ),
        )


def test_u6_full_plan_closes_required_algorithm_comparison() -> None:
    plan = UniversalFullResearchPlan.create(
        selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
        zero_shot_gate_passed=True,
        algorithms=(
            FullResearchAlgorithm.PPO,
            FullResearchAlgorithm.LAGRANGIAN,
            FullResearchAlgorithm.DISCOUNTED,
        ),
    )
    assert set(plan.algorithms) == set(FullResearchAlgorithm)
