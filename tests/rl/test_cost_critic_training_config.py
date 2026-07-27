from __future__ import annotations

import pytest

from trade_rl.rl.cost_learning import CONSTRAINT_COST_NAMES
from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 10,
        "gamma": 1.0,
        "seeds": (0,),
        "algorithm": "cost_critic_ppo",
        "n_steps": 4,
        "n_envs": 2,
        "batch_size": 8,
        "n_epochs": 1,
        "observation_encoder": "invalid_legacy_combination"
        if (False) and (False)
        else "hierarchical_sequence_v2"
        if (False)
        else "asset_set"
        if (False)
        else "flat_mlp",
        "device": "cpu",
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_cost_critic_algorithm_builds_typed_ppo_view() -> None:
    from trade_rl.rl.algorithm_configs import (
        CostCriticPPOConfig,
        build_algorithm_config,
    )

    source = _config(
        cost_learning_rate=5e-4,
        cost_n_epochs=3,
        cost_batch_size=4,
        cost_continuous_hidden_dims=(64, 32),
        cost_event_hidden_dims=(48, 24),
        cost_max_grad_norm=0.75,
        cost_continuous_gae_lambda=0.97,
        cost_event_gae_lambda=1.0,
        cost_value_loss_coefficient=0.5,
        cost_auxiliary_event_loss_coefficient=0.25,
    )

    config = build_algorithm_config(source)

    assert isinstance(config, CostCriticPPOConfig)
    assert config.n_steps == 4
    assert config.cost_learning_rate == pytest.approx(5e-4)
    assert config.cost_n_epochs == 3
    assert config.cost_batch_size == 4
    assert config.cost_continuous_hidden_dims == (64, 32)
    assert config.cost_event_hidden_dims == (48, 24)
    assert config.cost_max_grad_norm == pytest.approx(0.75)
    assert config.cost_schema.names == CONSTRAINT_COST_NAMES
    assert all(spec.gamma == 1.0 for spec in config.cost_schema.specs)
    assert all(
        spec.gae_lambda == pytest.approx(1.0)
        for spec in config.cost_schema.specs
        if spec.name in {"drawdown_stop_event", "forced_liquidation_event"}
    )
    assert config.cost_schema["daily_turnover"].gae_lambda == pytest.approx(0.97)
    assert config.cost_schema["daily_turnover"].value_loss_coefficient == pytest.approx(
        0.5
    )
    assert config.cost_schema[
        "forced_liquidation_event"
    ].auxiliary_event_loss_coefficient == pytest.approx(0.25)


def test_cost_critic_algorithm_uses_complete_ppo_rollout_rounding() -> None:
    config = _config()

    assert config.rounded_timesteps == 16


def test_cost_critic_algorithm_supports_sequence_encoder_and_bc_warm_start() -> None:
    config = _config(
        policy="MultiInputPolicy",
        observation_encoder=(
            "invalid_legacy_combination"
            if (True) and (False)
            else "hierarchical_sequence_v2"
            if (True)
            else "asset_set"
            if (False)
            else "flat_mlp"
        ),
        sequence_d_model=128,
        sequence_timeframe_attention_heads=4,
        sequence_asset_attention_heads=4,
        sequence_timeframe_attention_layers=1,
        sequence_asset_attention_layers=1,
        sequence_tcn_capacity="compact",
        behavior_cloning_epochs=1,
    )

    assert config.observation_encoder == "hierarchical_sequence_v2"
    assert config.behavior_cloning_epochs == 1


def test_ordinary_ppo_rejects_non_default_cost_critic_settings() -> None:
    with pytest.raises(
        ValueError,
        match="cost_learning_rate.*inactive|inactive.*cost_learning_rate",
    ):
        _config(algorithm="ppo", cost_learning_rate=1e-4)


def test_cost_critic_training_identity_tracks_schema_and_architecture() -> None:
    baseline = _config()
    changed_lambda = _config(cost_event_gae_lambda=1.0)
    changed_width = _config(cost_event_hidden_dims=(256, 64))

    assert baseline.digest_payload() != changed_lambda.digest_payload()
    assert baseline.digest_payload() != changed_width.digest_payload()


def test_cost_critic_settings_fail_closed() -> None:
    with pytest.raises(ValueError, match="cost_n_epochs"):
        _config(cost_n_epochs=0)
    with pytest.raises(ValueError, match="cost_batch_size"):
        _config(cost_batch_size=9)
    with pytest.raises(ValueError, match="cost_event_gae_lambda"):
        _config(cost_event_gae_lambda=1.1)
    with pytest.raises(ValueError, match="cost_event_hidden_dims"):
        _config(cost_event_hidden_dims=())
