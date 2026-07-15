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
