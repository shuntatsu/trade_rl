from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.rl.training import ResidualTrainingConfig


def _config() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=128,
        gamma=1.0,
        seeds=(0,),
        behavior_cloning_epochs=1,
        behavior_cloning_teacher="oracle",
    )


def test_training_config_accepts_explicit_causal_alpha_teacher() -> None:
    config = replace(_config(), behavior_cloning_teacher="causal_alpha_ridge")
    assert config.behavior_cloning_teacher == "causal_alpha_ridge"


def test_training_config_rejects_unknown_teacher() -> None:
    with pytest.raises(ValueError, match="behavior_cloning_teacher"):
        replace(_config(), behavior_cloning_teacher="causal_alpha_magic")
