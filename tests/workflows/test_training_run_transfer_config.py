from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.training_config import complete_execution_config
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.training_run import TrainingRunConfig, _training_backend


def _mapping() -> dict[str, object]:
    return {
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
        "action": {"alpha_enabled": False, "n_factors": 0},
    }


def test_training_config_parses_and_resolves_transfer_checkpoint_mapping(
    tmp_path: Path,
) -> None:
    raw = _mapping()
    raw["transfer_checkpoints"] = {"0": "transfer/stage-000"}

    config = TrainingRunConfig.from_mapping(raw).resolve_artifact_paths(tmp_path)

    assert config.transfer_checkpoints == ((0, tmp_path / "transfer/stage-000"),)


def test_training_config_rejects_transfer_seed_outside_ensemble() -> None:
    raw = _mapping()
    raw["transfer_checkpoints"] = {"7": "transfer/stage-000"}

    with pytest.raises(ValueError, match="transfer checkpoint seed"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_resume_and_transfer_for_same_seed() -> None:
    raw = _mapping()
    raw["resume_checkpoints"] = {"0": "resume/stage-000"}
    raw["transfer_checkpoints"] = {"0": "transfer/stage-001"}

    with pytest.raises(ValueError, match="resume and transfer"):
        TrainingRunConfig.from_mapping(raw)


def test_transfer_checkpoint_does_not_change_candidate_recipe_identity() -> None:
    raw = _mapping()
    uninterrupted = TrainingRunConfig.from_mapping(raw)
    raw["transfer_checkpoints"] = {"0": "transfer/stage-000"}
    transferred = TrainingRunConfig.from_mapping(raw)

    assert content_digest(uninterrupted.candidate_digest_payload()) == content_digest(
        transferred.candidate_digest_payload()
    )


def test_transfer_checkpoint_digest_is_bound_into_run_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.rl.training_run_config as training_run_config_module

    raw = _mapping()
    raw["transfer_checkpoints"] = {"0": "transfer/stage-000"}
    config = TrainingRunConfig.from_mapping(raw)
    monkeypatch.setattr(
        training_run_config_module,
        "load_checkpoint_manifest",
        lambda _: SimpleNamespace(digest="a" * 64),
    )

    payload = config.digest_payload()

    assert payload["resume_checkpoint_digests"] == {}
    assert payload["transfer_checkpoint_digests"] == {"0": "a" * 64}


def test_training_backend_receives_transfer_checkpoint_mapping(tmp_path: Path) -> None:
    raw = _mapping()
    raw["transfer_checkpoints"] = {"0": str(tmp_path / "stage-000")}
    config = TrainingRunConfig.from_mapping(raw)

    backend = _training_backend(lambda: object(), config)

    assert backend.resume_checkpoint_artifacts == {}
    assert backend.transfer_checkpoint_artifacts == {
        0: tmp_path / "stage-000",
    }
