from __future__ import annotations

import pytest

pytest.importorskip("stable_baselines3")

from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.a2c import (
    A2CFitMetadata,
    A2CIntentStrategy,
    fit_a2c_strategy,
)
from trade_rl.strategies.rl.a2c_artifact import (
    load_a2c_inference_bundle,
    save_a2c_inference_bundle,
)
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def test_real_cpu_a2c_fit_records_effective_steps_and_scope() -> None:
    dataset = pooled_market()
    strategy = fit_a2c_strategy(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=1.0,
        total_timesteps=11,
        seed=7,
    )
    assert strategy.policy.num_timesteps == 15
    assert strategy.fit_metadata is not None
    assert strategy.fit_metadata.requested_timesteps == 11
    assert strategy.fit_metadata.effective_timesteps == 15
    assert strategy.fit_metadata.step_rounding == "ceil_to_complete_rollout"
    assert strategy.fit_metadata.fit_symbols == ("BTCUSDT", "ETHUSDT")
    assert strategy.fit_metadata.required_coverage_timesteps == 6
    assert strategy.fit_metadata.nominal_full_episodes_per_symbol == 2


def test_real_cpu_a2c_fit_bundle_load_roundtrip(tmp_path) -> None:
    dataset = pooled_market()
    strategy = fit_a2c_strategy(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=11,
        seed=17,
    )

    root = tmp_path / "a2c-bundle"
    digest = save_a2c_inference_bundle(
        root,
        strategy,
        feature_names=dataset.feature_names,
    )
    loaded = load_a2c_inference_bundle(
        root,
        expected_digest=digest,
        feature_names=dataset.feature_names,
    )
    observation = StrategyObservation(
        index=2,
        timestamp=dataset.timestamps[2],
        symbol=dataset.symbols[0],
        features=dataset.features[2, 0],
        feature_available=dataset.feature_available[2, 0],
        global_features=dataset.global_features[2],
        global_feature_available=dataset.global_feature_available[2],
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
        feature_staleness=dataset.resolved_array("feature_staleness")[2, 0],
    )

    assert loaded.fit_metadata == strategy.fit_metadata
    assert loaded.fit_metadata is not None
    assert loaded.policy.num_timesteps == loaded.fit_metadata.effective_timesteps
    assert loaded.policy.num_timesteps == strategy.policy.num_timesteps
    assert loaded.decide(observation) is strategy.decide(observation)


def test_real_sb3_ppo_policy_cannot_be_serialized_as_a2c(tmp_path) -> None:
    import stable_baselines3

    dataset = pooled_market()
    policy = stable_baselines3.PPO(
        "MlpPolicy",
        PPOTradingEnv(
            dataset,
            feature_indices=(0,),
            start_index=0,
            stop_index=3,
            gross_budget=0.1,
        ),
        n_steps=5,
        batch_size=5,
        n_epochs=1,
        device="cpu",
        verbose=0,
    )
    metadata = A2CFitMetadata(
        requested_timesteps=5,
        effective_timesteps=5,
        rollout_steps=5,
        step_rounding="ceil_to_complete_rollout",
        seed=19,
        fit_symbol_indices=(0,),
        fit_symbols=(dataset.symbols[0],),
        episode_steps=3,
        required_coverage_timesteps=3,
        nominal_full_episodes_per_symbol=1,
    )
    strategy = A2CIntentStrategy(
        policy,
        feature_indices=(0,),
        feature_names=dataset.feature_names,
        fit_metadata=metadata,
    )
    root = tmp_path / "mislabelled-ppo-model"

    with pytest.raises(ValueError, match="A2C policy family"):
        save_a2c_inference_bundle(
            root,
            strategy,
            feature_names=dataset.feature_names,
        )

    assert not root.exists()
