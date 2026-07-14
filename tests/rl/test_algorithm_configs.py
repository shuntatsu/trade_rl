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
