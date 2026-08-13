from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaTeacherHoldoutMetric,
    evaluate_causal_alpha_teacher_admission,
)
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch, OracleEpisodeContract
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.rl.training import ResidualTrainingConfig


def _dataset() -> SupervisedPolicyDataset:
    return SupervisedPolicyDataset(
        observations=np.arange(12, dtype=np.float32).reshape(6, 2),
        actions=np.linspace(-0.5, 0.5, 6, dtype=np.float32)[:, None],
        dataset_id=content_digest("dataset"),
        train_start=0,
        train_stop=7,
        environment_digest=content_digest("env"),
        action_spec_digest=content_digest("action"),
        teacher_config_digest=content_digest("teacher"),
    )


def _split() -> BehaviorCloningSplit:
    return BehaviorCloningSplit(
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        validation_indices=np.asarray([4, 5], dtype=np.int64),
        train_episode_ids=np.asarray([0], dtype=np.int64),
        validation_episode_ids=np.asarray([1], dtype=np.int64),
    )


def _batch() -> EpisodeOracleBatch:
    contract = OracleEpisodeContract(
        dataset_id=content_digest("dataset"),
        episode_index=1,
        start=4,
        stop=7,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )
    return EpisodeOracleBatch(
        dataset_id=content_digest("dataset"),
        teacher_config_digest=content_digest("teacher"),
        sampling_config_digest=content_digest("sampling"),
        contracts=(contract,),
        targets=(np.asarray([[0.2], [0.2]], dtype=np.float32),),
    )


def _config() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=32,
        gamma=1.0,
        seeds=(7,),
        n_steps=8,
        batch_size=8,
        behavior_cloning_epochs=2,
        behavior_cloning_teacher="causal_alpha_ridge",
        behavior_cloning_required_relative_improvement=0.1,
        behavior_cloning_validation_fraction=0.1,
        behavior_cloning_critic_warm_start_steps=2,
        behavior_cloning_joint_warm_start_steps=2,
        behavior_cloning_critic_warm_start_learning_rate=1e-3,
        behavior_cloning_joint_warm_start_actor_lr_scale=0.1,
    )


def test_pretraining_hook_reuses_stored_teacher_admission_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trade_rl.integrations import universal_pretraining as module

    combined = module.combine_symbol_teachers(
        {"AAAUSDT": (_dataset(), _split(), np.arange(6, dtype=np.float32))},
        train_symbols=("AAAUSDT",),
        normalizer_digest=content_digest("normalizer"),
        feature_schema_digest=content_digest("features"),
    )
    admission = evaluate_causal_alpha_teacher_admission(
        (
            CausalAlphaTeacherHoldoutMetric(
                symbol="AAAUSDT",
                gross_return=0.02,
                net_return=0.01,
                turnover_per_day=0.1,
                total_execution_cost=0.2,
                trade_count=4,
                maximum_drawdown=0.03,
            ),
        )
    )
    bundle = replace(
        combined,
        episode_batches={"AAAUSDT": _batch()},
        causal_teacher_selection_evidence={
            "schema_version": "causal_alpha_selection_evidence_v1",
            "artifact_digest": content_digest("selection"),
            "selected_candidate_digest": content_digest("candidate"),
        },
        causal_teacher_admission_evidence=admission.to_payload(),
        causal_teacher_package_evidence={
            "schema_version": "universal_causal_alpha_teacher_package_evidence_v1",
            "artifact_digest": content_digest("package"),
        },
        causal_teacher_episode_hours=720.0,
    )

    monkeypatch.setattr(
        module,
        "evaluate_episode_action_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stored teacher admission must not replay holdouts")
        ),
        raising=False,
    )

    class ReachedBC(RuntimeError):
        pass

    monkeypatch.setattr(
        module,
        "pretrain_universal_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ReachedBC("bc reached")),
    )
    hook = module.build_universal_pretraining_hook(
        bundle,
        symbol_environment_factories={"AAAUSDT": lambda: object()},
    )

    with pytest.raises(ReachedBC, match="bc reached"):
        hook(
            policy=object(),
            config=_config(),
            behavior_cloning_seed=13,
            member_seed=7,
            output_root=tmp_path,
        )

    assert (tmp_path / "causal-teacher-selection.json").is_file()
    assert (tmp_path / "causal-teacher-admission.json").is_file()
    assert (tmp_path / "causal-teacher-package.json").is_file()
