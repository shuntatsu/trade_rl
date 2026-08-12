from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.rl.training import ResidualTrainingConfig


def _dataset(label: str) -> SupervisedPolicyDataset:
    return SupervisedPolicyDataset(
        observations=np.arange(12, dtype=np.float32).reshape(6, 2),
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


def _batch(label: str) -> EpisodeOracleBatch:
    dataset_id = content_digest((label, "dataset"))
    contract = OracleEpisodeContract(
        dataset_id=dataset_id,
        episode_index=1,
        start=4,
        stop=7,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )
    return EpisodeOracleBatch(
        dataset_id=dataset_id,
        teacher_config_digest=content_digest("teacher-config"),
        sampling_config_digest=content_digest((label, "sampling")),
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


def test_failed_teacher_holdout_stops_before_behavior_cloning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trade_rl.integrations import universal_pretraining as module

    symbols = ("AAAUSDT", "BBBUSDT")
    combined = module.combine_symbol_teachers(
        {
            "AAAUSDT": (_dataset("A"), _split(), np.arange(6, dtype=np.float32)),
            "BBBUSDT": (
                _dataset("B"),
                _split(),
                np.arange(6, dtype=np.float32) + 10,
            ),
        },
        train_symbols=symbols,
        normalizer_digest=content_digest("normalizer"),
        feature_schema_digest=content_digest("features"),
    )
    bundle = replace(
        combined,
        episode_batches={"AAAUSDT": _batch("A"), "BBBUSDT": _batch("B")},
        causal_teacher_selection_evidence={
            "schema_version": "causal_alpha_selection_evidence_v1",
            "artifact_digest": content_digest("selection-evidence"),
            "selected_candidate_digest": content_digest("candidate"),
        },
        causal_teacher_episode_hours=720.0,
    )
    calls: list[str] = []

    def replay(_factory, contract, *, actions):
        calls.append(f"holdout:{contract.dataset_id}")
        assert actions.shape == (2, 1)
        return SimpleNamespace(
            performance=SimpleNamespace(
                gross_return=-0.01,
                net_return=-0.02,
                turnover_total=2.0,
                cost_total=0.5,
                trade_count=2,
                maximum_drawdown=-0.1,
            )
        )

    monkeypatch.setattr(module, "evaluate_episode_action_path", replay)
    monkeypatch.setattr(
        module,
        "pretrain_universal_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("BC must not run after failed teacher admission")
        ),
    )
    hook = module.build_universal_pretraining_hook(
        bundle,
        symbol_environment_factories={symbol: lambda: object() for symbol in symbols},
    )

    with pytest.raises(RuntimeError, match="causal teacher admission"):
        hook(
            policy=object(),
            config=_config(),
            behavior_cloning_seed=13,
            member_seed=7,
            output_root=tmp_path,
        )

    assert len(calls) == 2
    selection_path = tmp_path / "causal-teacher-selection.json"
    admission_path = tmp_path / "causal-teacher-admission.json"
    assert selection_path.is_file()
    assert admission_path.is_file()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    assert selection["selected_candidate_digest"] == content_digest("candidate")
    assert admission["passed"] is False
    assert admission["negative_gross_symbol_count"] == 2


def test_teacher_holdout_is_not_replayed_for_legacy_teacher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trade_rl.integrations import universal_pretraining as module

    combined = module.combine_symbol_teachers(
        {"AAAUSDT": (_dataset("A"), _split(), np.arange(6, dtype=np.float32))},
        train_symbols=("AAAUSDT",),
        normalizer_digest=content_digest("normalizer"),
        feature_schema_digest=content_digest("features"),
    )
    monkeypatch.setattr(
        module,
        "evaluate_episode_action_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy teacher must not run causal teacher admission")
        ),
        raising=False,
    )
    assert combined.causal_teacher_selection_evidence is None
