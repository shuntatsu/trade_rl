from __future__ import annotations

from typing import Any

import pytest

from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: Any) -> ResidualTrainingConfig:
    values: dict[str, Any] = {
        "timesteps": 2_048,
        "gamma": 1.0,
        "seeds": (0,),
        "behavior_cloning_min_causal_holdout_episodes": 1,
        "behavior_cloning_min_causal_holdout_net_return_lower_bound": -1.0,
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_config_rejects_invalid_minimum_causal_holdout_episode_count(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="behavior_cloning_min_causal_holdout_episodes",
    ):
        _config(behavior_cloning_min_causal_holdout_episodes=value)


@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), float("-inf"), -1.0001),
)
def test_config_rejects_invalid_causal_net_return_lower_bound_floor(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="behavior_cloning_min_causal_holdout_net_return_lower_bound",
    ):
        _config(behavior_cloning_min_causal_holdout_net_return_lower_bound=value)


def test_config_defaults_preserve_legacy_bc_admission_contract() -> None:
    resolved = ResidualTrainingConfig(
        timesteps=2_048,
        gamma=1.0,
        seeds=(0,),
    )

    assert resolved.behavior_cloning_min_causal_holdout_episodes == 1
    assert resolved.behavior_cloning_min_causal_holdout_net_return_lower_bound == -1.0
