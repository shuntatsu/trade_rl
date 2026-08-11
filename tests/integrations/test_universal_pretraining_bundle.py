from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.behavior_cloning import BehaviorCloningConfig
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.rl.training import ResidualTrainingConfig


def _dataset(label: str, offset: float) -> SupervisedPolicyDataset:
    observations = {
        "active": np.ones((6, 1), dtype=np.float32),
        "current_weights": np.zeros((6, 1), dtype=np.float32),
        "instrument_context": np.full((6, 1, 9), offset, dtype=np.float32),
        "global_state": np.full((6, 2), offset, dtype=np.float32),
    }
    return SupervisedPolicyDataset(
        observations=observations,
        actions=np.linspace(-0.5, 0.5, 6, dtype=np.float32)[:, None],
        dataset_id=content_digest((label, "dataset")),
        train_start=0,
        train_stop=7,
        environment_digest=content_digest((label, "env")),
        action_spec_digest=content_digest("generic-action"),
        teacher_config_digest=content_digest("teacher-config"),
    )


def _split() -> BehaviorCloningSplit:
    return BehaviorCloningSplit(
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        validation_indices=np.asarray([4, 5], dtype=np.int64),
        train_episode_ids=np.asarray([0], dtype=np.int64),
        validation_episode_ids=np.asarray([1], dtype=np.int64),
    )


def test_combine_symbol_teachers_preserves_exact_symbol_train_scope() -> None:
    from trade_rl.integrations.universal_pretraining import combine_symbol_teachers

    combined = combine_symbol_teachers(
        {
            "AAAUSDT": (_dataset("A", 1.0), _split(), np.arange(6, dtype=np.float32)),
            "BBBUSDT": (_dataset("B", 2.0), _split(), np.arange(6, dtype=np.float32) + 10),
        },
        train_symbols=("AAAUSDT", "BBBUSDT"),
        normalizer_digest=content_digest("normalizer"),
        feature_schema_digest=content_digest("features"),
    )

    assert combined.dataset.sample_count == 12
    assert set(combined.split.train_indices.tolist()) == {0, 1, 2, 3, 6, 7, 8, 9}
    assert set(combined.split.validation_indices.tolist()) == {4, 5, 10, 11}
    assert combined.symbol_sample_indices == {
        "AAAUSDT": (0, 1, 2, 3),
        "BBBUSDT": (6, 7, 8, 9),
    }
    assert combined.critic_targets.tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    ]
    assert len(combined.teacher_artifact.artifact_digest) == 64


def test_combine_symbol_teachers_rejects_non_train_symbol() -> None:
    from trade_rl.integrations.universal_pretraining import combine_symbol_teachers

    with pytest.raises(ValueError, match="exactly match train_symbols"):
        combine_symbol_teachers(
            {"AAAUSDT": (_dataset("A", 1.0), _split(), np.arange(6))},
            train_symbols=("AAAUSDT", "BBBUSDT"),
            normalizer_digest=content_digest("normalizer"),
            feature_schema_digest=content_digest("features"),
        )


def test_build_universal_pretraining_hook_runs_balanced_bc_then_critic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trade_rl.integrations import universal_pretraining as module

    combined = module.combine_symbol_teachers(
        {
            "AAAUSDT": (_dataset("A", 1.0), _split(), np.arange(6, dtype=np.float32)),
            "BBBUSDT": (_dataset("B", 2.0), _split(), np.arange(6, dtype=np.float32) + 10),
        },
        train_symbols=("AAAUSDT", "BBBUSDT"),
        normalizer_digest=content_digest("normalizer"),
        feature_schema_digest=content_digest("features"),
    )
    calls: list[str] = []

    def fake_bc(*args: object, **kwargs: object) -> object:
        calls.append("bc")
        assert kwargs["symbol_sample_indices"] == combined.symbol_sample_indices
        return SimpleNamespace(
            initial_mse=1.0,
            final_mse=0.5,
            digest=content_digest("bc-result"),
        )

    def fake_warm(*args: object, **kwargs: object) -> object:
        calls.append("critic")
        assert np.array_equal(kwargs["sample_indices"], combined.split.train_indices)
        return SimpleNamespace(
            actor_max_abs_drift_critic_only=0.0,
            actor_max_abs_drift_joint=0.01,
        )

    monkeypatch.setattr(module, "pretrain_universal_policy", fake_bc)
    monkeypatch.setattr(module, "warm_start_policy_actor_critic", fake_warm)

    config = ResidualTrainingConfig(
        timesteps=32,
        gamma=1.0,
        seeds=(7,),
        n_steps=8,
        batch_size=8,
        behavior_cloning_epochs=2,
        behavior_cloning_teacher="oracle",
        behavior_cloning_required_relative_improvement=0.1,
        behavior_cloning_critic_warm_start_steps=2,
        behavior_cloning_critic_joint_fine_tune_steps=2,
        behavior_cloning_critic_learning_rate=1e-3,
        behavior_cloning_critic_joint_actor_learning_rate_scale=0.1,
    )
    hook = module.build_universal_pretraining_hook(combined)
    evidence = hook(
        policy=object(),
        config=config,
        behavior_cloning_seed=13,
        member_seed=7,
        output_root=tmp_path,
    )

    assert calls == ["bc", "critic"]
    assert evidence["passed"] is True
    assert evidence["teacher_artifact_digest"] == combined.teacher_artifact.artifact_digest
    assert len(evidence["behavior_cloning_digest"]) == 64
    assert len(evidence["critic_warm_start_digest"]) == 64
