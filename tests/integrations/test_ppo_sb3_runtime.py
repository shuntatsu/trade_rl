from __future__ import annotations

from pathlib import Path

import pytest

from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import (
    PPOIntentStrategy,
    PPOTradingEnv,
    fit_ppo_strategy,
)
from trade_rl.strategies.rl.ppo_artifact import (
    load_normalized_ppo,
    save_normalized_ppo,
)


def _env() -> PPOTradingEnv:
    return PPOTradingEnv(
        pooled_market(),
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
    )


def _observation(symbol_index: int = 0) -> StrategyObservation:
    dataset = pooled_market()
    global_available = dataset.resolved_array("global_feature_available")[0]
    return StrategyObservation(
        index=0,
        timestamp=dataset.timestamps[0],
        symbol=dataset.symbols[symbol_index],
        features=dataset.features[0, symbol_index],
        feature_available=dataset.feature_available[0, symbol_index],
        feature_staleness=dataset.resolved_array("feature_staleness")[0, symbol_index],
        global_features=dataset.global_features[0],
        global_feature_available=global_available,
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


def test_real_sb3_accepts_environment_and_runs_sequential_rollout() -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    env_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    torch = pytest.importorskip("torch")

    assert stable_baselines3.__version__ == "2.3.2"
    assert torch.__version__.split("+", 1)[0] == "2.4.1"
    torch.set_num_threads(1)

    env_checker.check_env(_env(), warn=True)

    model = stable_baselines3.PPO(
        "MlpPolicy",
        _env(),
        n_steps=8,
        batch_size=8,
        seed=17,
        ent_coef=0.0,
        verbose=0,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )
    model.learn(total_timesteps=16)

    strategy = PPOIntentStrategy(model, feature_indices=(0,))
    decision = strategy.decide(_observation())

    assert decision in {
        PositionIntent.SHORT,
        PositionIntent.FLAT,
        PositionIntent.LONG,
    }
    assert model.num_timesteps == 16


def test_real_sb3_interleaved_normalized_fit_roundtrips_bundle(
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    dataset = pooled_market()

    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
        seed=23,
        training_layout="interleaved",
        rollout_steps_per_env=32,
        normalize_features=True,
    )
    assert strategy.feature_normalizer is not None

    root = tmp_path / "normalized-ppo"
    digest = save_normalized_ppo(root, strategy)
    loaded = load_normalized_ppo(
        root,
        expected_digest=digest,
        feature_names=dataset.feature_names,
    )

    before = strategy.decide(_observation())
    after = loaded.decide(_observation())

    assert after is before
    assert loaded.feature_normalizer == strategy.feature_normalizer
