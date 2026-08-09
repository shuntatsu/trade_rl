from __future__ import annotations

import pytest

from trade_rl.rl.algorithm_configs import PPOConfig, SACConfig, build_algorithm_config
from trade_rl.rl.training import ResidualTrainingConfig


def _base(**overrides):
    values = {"timesteps": 100, "gamma": 0.99, "seeds": (1,)}
    values.update(overrides)
    return ResidualTrainingConfig(**values)


def test_build_ppo_config_exposes_only_ppo_parameters() -> None:
    config = build_algorithm_config(_base())

    assert isinstance(config, PPOConfig)
    assert not hasattr(config, "buffer_size")


def test_build_sac_config_exposes_only_off_policy_parameters() -> None:
    config = build_algorithm_config(_base(algorithm="sac"))

    assert isinstance(config, SACConfig)
    assert not hasattr(config, "n_epochs")


def test_typed_config_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_algorithm_config(_base(), algorithm="unknown")


def test_vector_environment_width_is_validated_and_digested() -> None:
    config = _base(n_envs=4, n_steps=8, batch_size=8)

    assert config.n_envs == 4
    assert config.digest_payload()["n_envs"] == 4

    with pytest.raises(ValueError, match="n_envs must be a positive integer"):
        _base(n_envs=0)


def test_ppo_batch_and_timestep_rounding_use_complete_vector_rollout() -> None:
    config = _base(
        timesteps=10,
        n_steps=4,
        n_envs=2,
        batch_size=8,
    )

    assert config.rounded_timesteps == 16


def test_vector_environment_width_preserves_existing_positional_arguments() -> None:
    config = ResidualTrainingConfig(8, 0.99, (0,), 3e-4, 8, 8)

    assert config.batch_size == 8
    assert config.n_envs == 1


def test_behavior_cloning_teacher_and_quality_gate_are_validated_and_digested() -> None:
    config = _base(
        behavior_cloning_epochs=1,
        behavior_cloning_teacher="trend_baseline",
        behavior_cloning_required_relative_improvement=0.05,
        behavior_cloning_gate_prediction_threshold=0.49,
    )

    assert config.digest_payload()["behavior_cloning_teacher"] == "trend_baseline"
    assert config.digest_payload()[
        "behavior_cloning_required_relative_improvement"
    ] == pytest.approx(0.05)
    assert config.digest_payload()[
        "behavior_cloning_gate_prediction_threshold"
    ] == pytest.approx(0.49)

    with pytest.raises(ValueError, match="behavior_cloning_teacher"):
        _base(behavior_cloning_teacher="future_oracle")
    with pytest.raises(
        ValueError, match="behavior_cloning_required_relative_improvement"
    ):
        _base(behavior_cloning_required_relative_improvement=1.0)
    with pytest.raises(ValueError, match="gate_prediction_threshold"):
        _base(behavior_cloning_gate_prediction_threshold=1.0)


def test_behavior_cloning_seed_is_optional_validated_and_digested() -> None:
    inherited = _base(behavior_cloning_epochs=1)
    fixed = _base(behavior_cloning_epochs=1, behavior_cloning_seed=0)

    assert inherited.behavior_cloning_seed is None
    assert inherited.digest_payload()["behavior_cloning_seed"] is None
    assert fixed.behavior_cloning_seed == 0
    assert fixed.digest_payload()["behavior_cloning_seed"] == 0

    with pytest.raises(ValueError, match="behavior_cloning_seed"):
        _base(behavior_cloning_epochs=1, behavior_cloning_seed=-1)
