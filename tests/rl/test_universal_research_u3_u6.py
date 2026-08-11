from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.universal_bc import (
    CriticWarmStartPhase,
    SymbolBalancedBatchSampler,
    UniversalTeacherArtifact,
)
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    architecture_spec,
)
from trade_rl.rl.universal_normalization import SymbolBalancedStandardNormalizer
from trade_rl.workflows.universal_research import (
    UniversalResearchManifest,
    validate_full_research_inputs,
)


def test_symbol_balanced_normalizer_weights_symbols_equally_and_clips() -> None:
    normalizer = SymbolBalancedStandardNormalizer.fit(
        {
            "AAAUSDT": np.asarray([[0.0], [2.0]], dtype=np.float64),
            "BBBUSDT": np.asarray([[100.0], [102.0], [104.0], [106.0]], dtype=np.float64),
        },
        train_symbols=("AAAUSDT", "BBBUSDT"),
        feature_schema_digest="features-v1",
        catalog_digest="catalog-v1",
        split_manifest_digest="split-v1",
        fold_train_range=(10, 20),
        max_samples_per_symbol=2,
    )
    # Per-symbol equal sampling gives samples [0,2,100,106], not row-count weighting.
    assert normalizer.sample_count_per_symbol == 2
    assert normalizer.train_symbols == ("AAAUSDT", "BBBUSDT")
    transformed = normalizer.transform(np.asarray([[-1_000.0], [1_000.0]]))
    assert transformed.min() >= -10.0
    assert transformed.max() <= 10.0


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
