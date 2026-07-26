from __future__ import annotations

import pytest

from trade_rl.rl.algorithm_configs import (
    LagrangianPPOConfig,
    build_algorithm_config,
)
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.training import ResidualTrainingConfig


_CANONICAL_SUPPORT = (1, 20, 1, 20, 1, 1, 1)


def _lagrangian_values() -> dict[str, object]:
    count = len(CONSTRAINT_COST_NAMES)
    return {
        "lagrangian_budgets": (0.01,) * count,
        "lagrangian_dual_learning_rates": (0.05,) * count,
        "lagrangian_ema_betas": (0.9,) * count,
        "lagrangian_initial_multipliers": (0.0,) * count,
        "lagrangian_max_multipliers": (10.0,) * count,
        "lagrangian_warmup_rollouts": (0,) * count,
        "lagrangian_update_interval_rollouts": (1,) * count,
        "lagrangian_minimum_completed_episodes": _CANONICAL_SUPPORT,
        "lagrangian_probe_episodes": 2,
        "lagrangian_probe_max_steps_per_episode": 16,
    }


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 10,
        "gamma": 1.0,
        "seeds": (0,),
        "algorithm": "lagrangian_ppo",
        "n_steps": 4,
        "n_envs": 2,
        "batch_size": 8,
        "n_epochs": 1,
        "asset_set_encoder": False,
        "device": "cpu",
        **_lagrangian_values(),
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_lagrangian_algorithm_builds_typed_configuration() -> None:
    source = _config()
    config = build_algorithm_config(source)

    assert isinstance(config, LagrangianPPOConfig)
    assert config.cost_schema.names == CONSTRAINT_COST_NAMES
    assert config.lagrangian_schema.names == CONSTRAINT_COST_NAMES
    assert tuple(
        spec.minimum_completed_episodes for spec in config.lagrangian_schema.specs
    ) == _CANONICAL_SUPPORT
    assert config.probe_episodes == 2
    assert config.probe_max_steps_per_episode == 16
    assert config.actor_composition_mode == "raw_lagrangian_then_sb3_normalize_v1"


def test_lagrangian_algorithm_uses_complete_ppo_rollout_rounding() -> None:
    assert _config().rounded_timesteps == 16


def test_lagrangian_configuration_requires_complete_vectors() -> None:
    with pytest.raises(ValueError, match="lagrangian_budgets"):
        _config(lagrangian_budgets=(0.01,))
    with pytest.raises(ValueError, match="lagrangian_minimum_completed_episodes"):
        _config(lagrangian_minimum_completed_episodes=(1,) * 6)
    with pytest.raises(ValueError, match="lagrangian_probe_episodes"):
        _config(lagrangian_probe_episodes=0)
    with pytest.raises(ValueError, match="lagrangian_probe_max_steps_per_episode"):
        _config(lagrangian_probe_max_steps_per_episode=0)


def test_non_lagrangian_algorithm_rejects_lagrangian_settings() -> None:
    with pytest.raises(
        ValueError,
        match="lagrangian_budgets.*inactive|inactive.*lagrangian_budgets",
    ):
        _config(algorithm="ppo")


def test_lagrangian_training_identity_tracks_every_semantic_field() -> None:
    baseline = _config().digest_payload()
    variations = (
        _config(lagrangian_budgets=(0.02,) * 7),
        _config(lagrangian_dual_learning_rates=(0.1,) * 7),
        _config(lagrangian_ema_betas=(0.8,) * 7),
        _config(lagrangian_initial_multipliers=(0.1,) * 7),
        _config(lagrangian_max_multipliers=(20.0,) * 7),
        _config(lagrangian_warmup_rollouts=(1,) * 7),
        _config(lagrangian_update_interval_rollouts=(2,) * 7),
        _config(lagrangian_minimum_completed_episodes=(2, 20, 2, 20, 2, 2, 2)),
        _config(lagrangian_probe_episodes=3),
        _config(lagrangian_probe_max_steps_per_episode=32),
    )

    assert all(item.digest_payload() != baseline for item in variations)


def test_ordinary_ppo_identity_omits_inactive_lagrangian_contract() -> None:
    config = ResidualTrainingConfig(
        timesteps=8,
        gamma=1.0,
        seeds=(0,),
        n_steps=4,
        batch_size=4,
        asset_set_encoder=False,
    )

    assert "lagrangian" not in config.digest_payload()
