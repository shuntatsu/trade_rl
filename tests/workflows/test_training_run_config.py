from __future__ import annotations

import pytest

from tests.support.training_config import complete_execution_config
from trade_rl.workflows.training_run import TrainingRunConfig


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
        "action": {"alpha_enabled": True, "n_factors": 0},
    }


def test_training_config_requires_alpha_artifact_when_action_enables_alpha() -> None:
    with pytest.raises(ValueError, match="alpha artifact"):
        TrainingRunConfig.from_mapping(_mapping())


def test_training_config_accepts_alpha_artifact_path() -> None:
    raw = _mapping()
    raw["alpha_artifact"] = "artifacts/alpha"
    config = TrainingRunConfig.from_mapping(raw)
    assert config.alpha_artifact is not None
    assert config.alpha_artifact.as_posix() == "artifacts/alpha"


def test_training_config_parses_git_dirty_and_includes_it_in_identity() -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["git_commit"] = "a" * 40
    raw["git_dirty"] = False

    config = TrainingRunConfig.from_mapping(raw)

    assert config.git_dirty is False
    assert config.digest_payload()["git_dirty"] is False


def test_training_config_rejects_non_boolean_git_dirty() -> None:
    raw = _mapping()
    raw["alpha_artifact"] = "artifacts/alpha"
    raw["git_dirty"] = "false"

    with pytest.raises(ValueError, match="git_dirty must be a boolean or null"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_from_json_resolves_signal_artifact_paths(
    tmp_path: object,
) -> None:
    import json
    from pathlib import Path

    root = Path(str(tmp_path))
    raw = _mapping()
    raw["alpha_artifact"] = "signals/alpha"
    path = root / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = TrainingRunConfig.from_json(path)

    assert config.alpha_artifact == root / "signals" / "alpha"


def test_training_config_identity_uses_signal_content_not_filesystem_path(
    tmp_path: object,
) -> None:
    from pathlib import Path

    import numpy as np

    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.artifacts.signals import write_signal_artifact

    root = Path(str(tmp_path))
    paths = (root / "first", root / "second")
    for path in paths:
        write_signal_artifact(
            path,
            kind="alpha",
            dataset_id="d" * 64,
            fit_start=0,
            fit_stop=2,
            names=("BTC",),
            values=np.zeros((4, 1)),
        )
    raw = _mapping()
    raw["alpha_artifact"] = str(paths[0])
    first = TrainingRunConfig.from_mapping(raw)
    raw["alpha_artifact"] = str(paths[1])
    second = TrainingRunConfig.from_mapping(raw)

    assert content_digest(first.digest_payload()) == content_digest(
        second.digest_payload()
    )
    assert first.digest_payload()["alpha_artifact_digest"] is not None


def _workflow_dataset():
    import numpy as np

    from trade_rl.data.market import MarketDataset

    n_bars = 32
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.ones((n_bars, 1), dtype=np.float64)
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("BTC",),
        timestamps=timestamps,
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.ones((n_bars, 1), dtype=np.float64),
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_authoritative_training_factory_requires_complete_reward_preroll() -> None:
    from trade_rl.workflows.training_run import _environment_factory

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["reward"] = {
        "baseline_window_hours": 4.0,
        "baseline_minimum_history_hours": 4.0,
    }
    config = TrainingRunConfig.from_mapping(raw)

    env = _environment_factory(_workflow_dataset(), config)()

    assert config.environment.require_full_reward_preroll is True
    assert env.config.require_full_reward_preroll is True


def test_walk_forward_environment_requires_complete_reward_preroll() -> None:
    from trade_rl.workflows.walk_forward_evaluation import build_market_environment

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["reward"] = {
        "baseline_window_hours": 4.0,
        "baseline_minimum_history_hours": 4.0,
    }
    config = TrainingRunConfig.from_mapping(raw)

    env = build_market_environment(
        _workflow_dataset(),
        config,
        normalizer=None,
        sequence_normalizer=None,
        episode_bars=4,
        liquidate_on_end=False,
    )

    assert env.config.require_full_reward_preroll is True


def test_normalizer_fit_begins_after_complete_reward_preroll() -> None:
    from trade_rl.evaluation.walk_forward.folds import IndexRange
    from trade_rl.workflows._market_walk_forward_core import _fit_normalizer

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["reward"] = {
        "baseline_window_hours": 4.0,
        "baseline_minimum_history_hours": 4.0,
    }
    config = TrainingRunConfig.from_mapping(raw)

    normalizer = _fit_normalizer(
        _workflow_dataset(), IndexRange(start=0, stop=24), config
    )

    assert normalizer.train_end > normalizer.train_start


def test_sequence_training_rejects_flat_export_and_declares_native_serving_supported() -> (
    None
):
    from trade_rl.workflows.training_run import _serving_support_payload

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["training"] = {
        **raw["training"],  # type: ignore[arg-type]
        "policy": "MultiInputPolicy",
        "observation_encoder": "hierarchical_sequence_v2",
        "policy_actor_head": "hierarchical_gate_target_v1",
    }
    raw["environment"] = {
        **raw["environment"],  # type: ignore[arg-type]
        "structured_sequence_observation": True,
        "sequence_windows": [["15m", 1], ["1h", 1], ["4h", 1], ["1d", 1]],
    }
    raw["exports"] = {"onnx": True, "torchscript": False}

    with pytest.raises(ValueError, match="structured sequence policies"):
        TrainingRunConfig.from_mapping(raw)

    raw["exports"] = {"onnx": False, "torchscript": False}
    config = TrainingRunConfig.from_mapping(raw)
    support = _serving_support_payload(config)
    assert support == {
        "loader_schema": "sb3_policy_loader_v3",
        "observation_mode": "structured_sequence",
        "runtime": "native_sb3_structured_sequence_v1",
        "schema_version": "serving_support_v2",
        "status": "supported",
    }


def test_training_config_parses_seed_checkpoint_resume_mapping(tmp_path) -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["resume_checkpoints"] = {"0": "resume/step-1"}
    config = TrainingRunConfig.from_mapping(raw).resolve_artifact_paths(tmp_path)
    assert config.resume_checkpoints == ((0, tmp_path / "resume/step-1"),)


def test_resume_checkpoint_does_not_change_candidate_recipe_identity() -> None:
    from trade_rl.artifacts.hashing import content_digest

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    uninterrupted = TrainingRunConfig.from_mapping(raw)
    raw["resume_checkpoints"] = {"0": "resume/step-1"}
    resumed = TrainingRunConfig.from_mapping(raw)

    assert content_digest(uninterrupted.candidate_digest_payload()) == content_digest(
        resumed.candidate_digest_payload()
    )


def test_training_config_rejects_resume_seed_outside_ensemble() -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["resume_checkpoints"] = {"7": "resume/step-1"}
    with pytest.raises(ValueError, match="resume checkpoint seed"):
        TrainingRunConfig.from_mapping(raw)


def test_training_dataset_reference_declares_unshifted_ichimoku_alignment() -> None:
    from trade_rl.workflows.training_run import _feature_alignment_payload

    names = (
        "15m__log_return_1bar",
        "15m__ichimoku_tenkan_distance_9bar",
        "1h__ichimoku_cloud_position_9_26_52",
    )
    assert _feature_alignment_payload(names) == {
        "15m__ichimoku_tenkan_distance_9bar": "unshifted_decision_time",
        "1h__ichimoku_cloud_position_9_26_52": "unshifted_decision_time",
    }


def test_training_config_requires_explicit_schema_version() -> None:
    raw = _mapping()
    raw.pop("schema_version")
    with pytest.raises(ValueError, match="missing required fields.*schema_version"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_unknown_top_level_field() -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["schema_verison"] = "training_run_config_v4"
    with pytest.raises(ValueError, match="unknown fields.*schema_verison"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_unknown_export_field() -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["exports"] = {"torchcript": True}
    with pytest.raises(ValueError, match="exports.*unknown fields.*torchcript"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_misspelled_training_field() -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    training = dict(raw["training"])  # type: ignore[arg-type]
    training["observation_encodr"] = "flat_mlp"
    raw["training"] = training
    with pytest.raises(
        ValueError, match="training.*unknown fields.*observation_encodr"
    ):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_shadow_reward_inside_environment() -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    environment = dict(raw["environment"])  # type: ignore[arg-type]
    environment["reward_config"] = {}
    raw["environment"] = environment
    with pytest.raises(ValueError, match="environment.*unknown fields.*reward_config"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_resolves_typed_encoder_and_cuda_modes() -> None:
    from trade_rl.rl.training_modes import CudaRuntimeMode, ObservationEncoder

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    training = dict(raw["training"])  # type: ignore[arg-type]
    training.update(
        {
            "observation_encoder": "flat_mlp",
            "cuda_runtime_mode": "deterministic",
        }
    )
    raw["training"] = training

    config = TrainingRunConfig.from_mapping(raw)

    assert config.training.observation_encoder is ObservationEncoder.FLAT_MLP
    assert config.training.cuda_runtime_mode is CudaRuntimeMode.DETERMINISTIC
    assert config.training.digest_payload()["observation_encoder"] == "flat_mlp"
    assert config.training.digest_payload()["cuda_runtime_mode"] == "deterministic"


def _structured_mapping() -> dict[str, object]:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    training = dict(raw["training"])  # type: ignore[arg-type]
    training.update(
        {
            "policy": "MultiInputPolicy",
            "observation_encoder": "hierarchical_sequence_v2",
            "policy_actor_head": "hierarchical_gate_target_v1",
        }
    )
    raw["training"] = training
    environment = dict(raw["environment"])  # type: ignore[arg-type]
    environment.update(
        {
            "structured_sequence_observation": True,
            "sequence_windows": [["15m", 1], ["1h", 1], ["4h", 1], ["1d", 1]],
        }
    )
    raw["environment"] = environment
    return raw


def test_training_config_accepts_structured_torchscript_for_sequence_policy() -> None:
    raw = _structured_mapping()
    raw["exports"] = {"structured_torchscript": True}

    config = TrainingRunConfig.from_mapping(raw)

    assert config.export_structured_torchscript is True
    assert config.digest_payload()["export_structured_torchscript"] is True


def test_training_config_rejects_structured_torchscript_for_flat_policy() -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["exports"] = {"structured_torchscript": True}

    with pytest.raises(ValueError, match="requires hierarchical_sequence_v2"):
        TrainingRunConfig.from_mapping(raw)


def test_candidate_identity_excludes_export_transport_and_git_provenance() -> None:
    from copy import deepcopy

    from trade_rl.artifacts.hashing import content_digest

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    baseline = TrainingRunConfig.from_mapping(raw)

    changed_raw = deepcopy(raw)
    changed_raw["exports"] = {"onnx": True, "tolerance": 0.0001}
    changed_raw["git_commit"] = "b" * 40
    changed_raw["git_dirty"] = True
    changed = TrainingRunConfig.from_mapping(changed_raw)

    assert content_digest(baseline.candidate_digest_payload()) == content_digest(
        changed.candidate_digest_payload()
    )
    assert content_digest(baseline.digest_payload()) != content_digest(
        changed.digest_payload()
    )


def test_candidate_identity_changes_when_learning_recipe_changes() -> None:
    from copy import deepcopy

    from trade_rl.artifacts.hashing import content_digest

    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    baseline = TrainingRunConfig.from_mapping(raw)

    changed_raw = deepcopy(raw)
    training = dict(changed_raw["training"])  # type: ignore[arg-type]
    training["timesteps"] = 16
    changed_raw["training"] = training
    changed = TrainingRunConfig.from_mapping(changed_raw)

    assert content_digest(baseline.candidate_digest_payload()) != content_digest(
        changed.candidate_digest_payload()
    )


def test_candidate_identity_payload_contains_only_authored_recipe_fields() -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["exports"] = {"onnx": True, "tolerance": 0.0001}
    raw["git_commit"] = "c" * 40
    raw["git_dirty"] = False
    config = TrainingRunConfig.from_mapping(raw)

    candidate = config.candidate_digest_payload()
    for excluded in (
        "export_onnx",
        "export_structured_torchscript",
        "export_tolerance",
        "export_torchscript",
        "git_commit",
        "git_dirty",
        "resume_checkpoint_digests",
        "transfer_checkpoint_digests",
    ):
        assert excluded not in candidate
