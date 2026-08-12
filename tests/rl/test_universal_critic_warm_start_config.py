from __future__ import annotations

import pytest

from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 128,
        "gamma": 1.0,
        "seeds": (7,),
        "n_steps": 8,
        "batch_size": 8,
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_universal_critic_warm_start_is_inactive_by_default() -> None:
    config = _config()

    assert config.behavior_cloning_critic_warm_start_steps == 0
    assert config.behavior_cloning_joint_warm_start_steps == 0
    assert config.behavior_cloning_critic_warm_start_enabled is False


def test_universal_critic_warm_start_requires_both_phases() -> None:
    with pytest.raises(ValueError, match="warm-start phases"):
        _config(
            behavior_cloning_epochs=2,
            behavior_cloning_critic_warm_start_steps=10,
        )


def test_universal_critic_warm_start_accepts_causal_trend_returns() -> None:
    config = _config(
        behavior_cloning_epochs=2,
        behavior_cloning_teacher="trend_baseline",
        behavior_cloning_critic_warm_start_steps=10,
        behavior_cloning_joint_warm_start_steps=5,
    )

    assert config.behavior_cloning_critic_warm_start_enabled is True


def test_universal_critic_warm_start_requires_behavior_cloning() -> None:
    with pytest.raises(ValueError, match="behavior cloning"):
        _config(
            behavior_cloning_critic_warm_start_steps=10,
            behavior_cloning_joint_warm_start_steps=5,
        )


def test_universal_critic_warm_start_validates_learning_rates() -> None:
    with pytest.raises(ValueError, match="critic warm-start learning rate"):
        _config(
            behavior_cloning_epochs=2,
            behavior_cloning_critic_warm_start_steps=10,
            behavior_cloning_joint_warm_start_steps=5,
            behavior_cloning_critic_warm_start_learning_rate=0.0,
        )
    with pytest.raises(ValueError, match="joint actor learning-rate scale"):
        _config(
            behavior_cloning_epochs=2,
            behavior_cloning_critic_warm_start_steps=10,
            behavior_cloning_joint_warm_start_steps=5,
            behavior_cloning_joint_warm_start_actor_lr_scale=1.1,
        )


def test_universal_critic_warm_start_enabled_contract() -> None:
    config = _config(
        behavior_cloning_epochs=2,
        behavior_cloning_critic_warm_start_steps=10,
        behavior_cloning_joint_warm_start_steps=5,
        behavior_cloning_critic_warm_start_learning_rate=3e-4,
        behavior_cloning_joint_warm_start_actor_lr_scale=0.1,
    )

    assert config.behavior_cloning_critic_warm_start_enabled is True
