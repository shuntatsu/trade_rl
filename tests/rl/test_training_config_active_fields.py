from __future__ import annotations

import pytest

from trade_rl.rl.algorithm_configs import SACConfig, build_algorithm_config
from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 16,
        "gamma": 0.99,
        "seeds": (0,),
        "n_steps": 2_048,
        "batch_size": 64,
        "n_epochs": 10,
        "asset_set_encoder": False,
        "device": "cpu",
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_off_policy_typed_config_carries_distinct_critic_architecture() -> None:
    config = build_algorithm_config(
        _config(
            algorithm="sac",
            policy_net_arch=(32, 16),
            value_net_arch=(64, 48),
            learning_starts=0,
        )
    )

    assert isinstance(config, SACConfig)
    assert config.policy_net_arch == (32, 16)
    assert config.value_net_arch == (64, 48)


def test_td3_rejects_sde_settings_that_sb3_td3_cannot_consume() -> None:
    with pytest.raises(ValueError, match="TD3.*SDE|SDE.*TD3"):
        _config(
            algorithm="td3",
            learning_starts=0,
            use_sde=True,
        )

    with pytest.raises(ValueError, match="TD3.*SDE|SDE.*TD3"):
        _config(
            algorithm="td3",
            learning_starts=0,
            sde_sample_freq=4,
        )


def test_ppo_rejects_non_default_off_policy_only_parameters() -> None:
    with pytest.raises(ValueError, match="inactive.*buffer_size|buffer_size.*PPO"):
        _config(buffer_size=128)


def test_off_policy_rejects_non_default_ppo_only_parameters() -> None:
    with pytest.raises(ValueError, match="inactive.*n_epochs|n_epochs.*SAC"):
        _config(
            algorithm="sac",
            learning_starts=0,
            n_epochs=2,
        )


def test_disabled_sequence_encoder_rejects_non_default_sequence_parameters() -> None:
    with pytest.raises(ValueError, match="sequence_d_model.*sequence_encoder"):
        _config(sequence_d_model=64)
    with pytest.raises(ValueError, match="sequence_capacity.*sequence_encoder"):
        _config(sequence_capacity="compact")


def test_compact_sequence_capacity_is_explicit_and_identity_bound() -> None:
    standard = _config(sequence_encoder=True, policy="MultiInputPolicy")
    compact = _config(
        sequence_encoder=True,
        sequence_capacity="compact",
        policy="MultiInputPolicy",
    )

    assert compact.sequence_capacity == "compact"
    assert compact.digest_payload() != standard.digest_payload()
    with pytest.raises(ValueError, match="sequence_capacity"):
        _config(
            sequence_encoder=True,
            sequence_capacity="tiny",
            policy="MultiInputPolicy",
        )


def test_disabled_asset_set_encoder_rejects_non_default_embedding_parameters() -> None:
    with pytest.raises(ValueError, match="asset_embedding_dim.*asset_set_encoder"):
        _config(asset_embedding_dim=32)


def test_disabled_behavior_cloning_rejects_non_default_cloning_parameters() -> None:
    with pytest.raises(ValueError, match="behavior_cloning_patience.*behavior cloning"):
        _config(behavior_cloning_patience=5)
