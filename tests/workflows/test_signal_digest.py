from __future__ import annotations

from dataclasses import replace

import numpy as np

from test_support.training_config import complete_execution_config
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.signals import write_signal_artifact
from trade_rl.workflows.market_walk_forward import (
    MarketWalkForwardConfig,
    NamedCandidateRun,
)
from trade_rl.workflows.training_run import TrainingRunConfig
from trade_rl.workflows.walk_forward import WalkForwardWorkflowConfig
from trade_rl.workflows.walk_forward_evaluation import resolve_signal_digest


def _run(alpha_path: object) -> TrainingRunConfig:
    return TrainingRunConfig.from_mapping(
        {
            "schema_version": "training_run_config_v4",
            "training": {
                "timesteps": 8,
                "gamma": 0.99,
                "seeds": [0],
                "n_steps": 8,
                "batch_size": 8,
                "policy_actor_head": "standard_continuous_v1",
                "hierarchical_gate_temperature": 1.0,
                "behavior_cloning_gate_loss_weight": 1.0,
                "behavior_cloning_target_loss_weight": 1.0,
                "behavior_cloning_composed_loss_weight": 1.0,
                "behavior_cloning_gate_change_threshold": 0.05,
                "behavior_cloning_max_positive_class_weight": 20.0,
                "behavior_cloning_min_gate_precision": 0.0,
                "behavior_cloning_min_gate_recall": 0.0,
                "behavior_cloning_max_active_target_rmse": 1.0,
                "behavior_cloning_min_activity_ratio": 0.0,
                "behavior_cloning_max_activity_ratio": 1.0,
                "behavior_cloning_min_causal_holdout_trades": 0,
                "behavior_cloning_max_causal_holdout_regret": 0.0,
                "behavior_cloning_causal_holdout_bootstrap_resamples": 2_000,
                "behavior_cloning_causal_holdout_confidence_level": 0.95,
            },
            "environment": {
                "episode_bars": 4,
                "decision_every": 1,
                "initial_capital": 1_000.0,
                "require_full_reward_preroll": True,
            },
            "execution": complete_execution_config(),
            "risk": {},
            "reward": {},
            "trend": {"fast_lookback": 1, "base_lookback": 2, "slow_lookback": 3},
            "action": {"alpha_enabled": True, "n_factors": 0},
            "alpha_artifact": str(alpha_path),
        }
    )


def test_signal_digest_binds_signal_artifact_content(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path)) / "alpha"
    artifact_digest = write_signal_artifact(
        root,
        kind="alpha",
        dataset_id="d" * 64,
        fit_start=0,
        fit_stop=2,
        names=("BTC",),
        values=np.zeros((10, 1)),
    )
    run = _run(root)
    config = MarketWalkForwardConfig(
        workflow=WalkForwardWorkflowConfig(
            n_bars=100,
            train_bars=30,
            checkpoint_bars=10,
            selection_bars=10,
            test_bars=10,
            purge_bars=1,
            max_folds=1,
        ),
        candidates=(NamedCandidateRun("candidate", run),),
    )

    resolved = resolve_signal_digest(config, dataset_id="d" * 64)

    assert resolved == content_digest(
        {
            "alpha_artifact_digest": artifact_digest,
            "factor_artifact_digest": None,
            "schema_version": "causal_signal_identity_v1",
            "trend": config.candidates[0].run.trend,
        }
    )
    assert resolved != config.signal_digest


def test_signal_digest_rejects_candidate_artifact_mismatch(tmp_path: object) -> None:
    from pathlib import Path

    import pytest

    roots = []
    for index in range(2):
        root = Path(str(tmp_path)) / f"alpha-{index}"
        write_signal_artifact(
            root,
            kind="alpha",
            dataset_id="d" * 64,
            fit_start=0,
            fit_stop=2,
            names=("BTC",),
            values=np.full((10, 1), float(index)),
        )
        roots.append(root)
    first = _run(roots[0])
    second = replace(_run(roots[1]), training=replace(first.training, seeds=(1,)))
    config = MarketWalkForwardConfig(
        workflow=WalkForwardWorkflowConfig(
            n_bars=100,
            train_bars=30,
            checkpoint_bars=10,
            selection_bars=10,
            test_bars=10,
            purge_bars=1,
            max_folds=1,
        ),
        candidates=(
            NamedCandidateRun("first", first),
            NamedCandidateRun("second", second),
        ),
    )

    with pytest.raises(ValueError, match="signal artifacts"):
        resolve_signal_digest(config, dataset_id="d" * 64)
