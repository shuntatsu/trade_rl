from pathlib import Path

import pytest

from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.workflows.universal_lagrangian_mechanics import (
    build_lagrangian_mechanics_config,
)

CONFIG = Path("examples/binance-multitimeframe/universal-u6-lagrangian.json")


def test_mechanics_config_preserves_reward_and_exercises_dual_requirements() -> None:
    base = TrainingRunConfig.from_json(CONFIG)

    mechanics = build_lagrangian_mechanics_config(
        base,
        episode_hours=8.0,
        timesteps=1_024,
    )

    assert mechanics.reward == base.reward
    assert mechanics.environment.episode_hours == 8.0
    assert base.environment.episode_hours == 720.0
    assert mechanics.training.timesteps == 1_024
    assert mechanics.training.lagrangian_probe_max_steps_per_episode == 32
    assert 1_024 // 32 >= max(
        mechanics.training.lagrangian_minimum_completed_episodes
    )
    assert 1_024 // mechanics.training.n_steps > max(
        mechanics.training.lagrangian_warmup_rollouts
    )


def test_mechanics_config_rejects_smoke_that_cannot_complete_constraints() -> None:
    base = TrainingRunConfig.from_json(CONFIG)

    with pytest.raises(ValueError, match="warmup|completed-episode"):
        build_lagrangian_mechanics_config(
            base,
            episode_hours=8.0,
            timesteps=base.training.n_steps,
        )
