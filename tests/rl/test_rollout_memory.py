from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.rl.rollout_memory import estimate_ppo_rollout_buffer_bytes
from trade_rl.rl.training import ResidualTrainingConfig


def test_dict_rollout_memory_matches_sb3_float32_storage_contract() -> None:
    observation_space = spaces.Dict(
        {
            "a": spaces.Box(-1, 1, shape=(2, 3), dtype=np.float16),
            "b": spaces.Box(0, 1, shape=(5,), dtype=np.uint8),
        }
    )
    # SB3 DictRolloutBuffer coerces every observation array to float32.
    elements_per_step = 6 + 5 + 3 + 6  # observations + actions + six scalar arrays
    assert (
        estimate_ppo_rollout_buffer_bytes(
            observation_space,
            n_steps=4,
            n_envs=2,
            action_dim=3,
        )
        == elements_per_step * 4 * 2 * 4
    )


def test_training_config_rejects_invalid_rollout_memory_cap() -> None:
    with pytest.raises(ValueError, match="max_rollout_buffer_bytes"):
        ResidualTrainingConfig(
            timesteps=8,
            gamma=0.99,
            seeds=(0,),
            n_steps=8,
            batch_size=8,
            max_rollout_buffer_bytes=0,
        )


def test_production_sequence_rollout_fits_configured_memory_cap() -> None:
    counts = {"15m": 59, "1h": 59, "4h": 55, "1d": 53}
    lengths = {"15m": 96, "1h": 168, "4h": 120, "1d": 60}
    observation_spaces: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(
            -np.inf, np.inf, shape=(3, 4 * 226), dtype=np.float32
        ),
        "asset_state": spaces.Box(-np.inf, np.inf, shape=(3, 18), dtype=np.float32),
        "global_state": spaces.Box(-np.inf, np.inf, shape=(35,), dtype=np.float32),
        "active": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
    }
    for timeframe, count in counts.items():
        shape = (3, lengths[timeframe], count)
        observation_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
            -np.inf, np.inf, shape=shape, dtype=np.float32
        )
        observation_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype=np.uint8
        )
        observation_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 1.0, shape=shape, dtype=np.float16
        )
    estimated = estimate_ppo_rollout_buffer_bytes(
        spaces.Dict(observation_spaces), n_steps=128, n_envs=4, action_dim=3
    )
    assert estimated == 473_122_816
    assert estimated < 805_306_368
