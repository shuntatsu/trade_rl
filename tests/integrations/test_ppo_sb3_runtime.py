from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.data.market import MarketDataset
from trade_rl.simulation import ExecutionCostConfig
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
    assert model.device.type == "cpu"
    assert model.num_timesteps == 16


def test_real_sb3_interleaved_normalized_fit_roundtrips_bundle(
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
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
    assert torch.get_num_threads() == 1

    root = tmp_path / "normalized-ppo"
    digest = save_normalized_ppo(root, strategy)
    torch.set_num_threads(2)
    loaded = load_normalized_ppo(
        root,
        expected_digest=digest,
        feature_names=dataset.feature_names,
    )

    before = strategy.decide(_observation())
    after = loaded.decide(_observation())

    assert after is before
    assert loaded.feature_normalizer == strategy.feature_normalizer
    assert loaded.policy.device.type == "cpu"
    assert torch.get_num_threads() == 1


def test_real_interleaved_fit_is_parameter_deterministic_for_same_seed() -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    dataset = pooled_market()

    def fit():
        return fit_ppo_strategy(
            dataset,
            feature_indices=(0,),
            fit_symbol_indices=(0, 1),
            start_index=0,
            stop_index=3,
            gross_budget=0.1,
            total_timesteps=64,
            seed=31,
            training_layout="interleaved",
            rollout_steps_per_env=32,
        )

    first = fit()
    second = fit()
    first_state = first.policy.policy.state_dict()
    second_state = second.policy.policy.state_dict()

    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name]), name


def test_real_sequential_fit_is_parameter_deterministic_for_same_seed() -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    dataset = pooled_market()

    def fit():
        return fit_ppo_strategy(
            dataset,
            feature_indices=(0,),
            fit_symbol_indices=(0, 1),
            start_index=0,
            stop_index=3,
            gross_budget=0.1,
            total_timesteps=1,
            seed=35,
        )

    first = fit()
    second = fit()
    first_state = first.policy.policy.state_dict()
    second_state = second.policy.policy.state_dict()

    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name]), name


def test_real_sequential_fit_records_default_rollout_rounded_timesteps() -> None:
    pytest.importorskip("stable_baselines3")
    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=1,
        seed=37,
    )

    assert strategy.policy.num_timesteps == 2048


def test_real_interleaved_fit_records_vector_rollout_rounded_timesteps() -> None:
    pytest.importorskip("stable_baselines3")
    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=65,
        seed=41,
        training_layout="interleaved",
        rollout_steps_per_env=32,
    )

    assert strategy.policy.num_timesteps == 128


def _strong_uptrend_market(n_bars: int = 64) -> MarketDataset:
    open_price = np.empty((n_bars, 1), dtype=np.float64)
    close = np.empty((n_bars, 1), dtype=np.float64)
    open_price[0, 0] = 100.0
    close[0, 0] = 100.0
    for index in range(1, n_bars):
        open_price[index, 0] = close[index - 1, 0]
        close[index, 0] = open_price[index, 0] * 1.05
    return MarketDataset(
        dataset_id="7" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.ones((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_real_ppo_learns_trivial_causal_long_signal() -> None:
    pytest.importorskip("stable_baselines3")
    dataset = _strong_uptrend_market()
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        start_index=0,
        stop_index=dataset.n_bars - 1,
        gross_budget=0.5,
        total_timesteps=4096,
        seed=53,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    observation = StrategyObservation(
        index=0,
        timestamp=dataset.timestamps[0],
        symbol=dataset.symbols[0],
        features=dataset.features[0, 0],
        feature_available=dataset.feature_available[0, 0],
        feature_staleness=dataset.resolved_array("feature_staleness")[0, 0],
        global_features=dataset.global_features[0],
        global_feature_available=dataset.resolved_array("global_feature_available")[0],
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )

    assert strategy.decide(observation) is PositionIntent.LONG
    torch = pytest.importorskip("torch")
    for name, parameter in strategy.policy.policy.named_parameters():
        assert torch.isfinite(parameter).all(), name


def test_real_raw_ppo_model_roundtrips_deterministic_intent(tmp_path: Path) -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=64,
        seed=59,
        training_layout="interleaved",
        rollout_steps_per_env=32,
    )
    model_path = tmp_path / "model.zip"
    strategy.policy.save(str(model_path))
    loaded = stable_baselines3.PPO.load(str(model_path), device="cpu")

    before = strategy.decide(_observation())
    after = PPOIntentStrategy(loaded, feature_indices=(0,)).decide(_observation())

    assert after is before
    assert loaded.device.type == "cpu"


def _strong_downtrend_market(n_bars: int = 64) -> MarketDataset:
    dataset = _strong_uptrend_market(n_bars)
    open_price = np.empty_like(dataset.open)
    close = np.empty_like(dataset.close)
    open_price[0, 0] = 100.0
    close[0, 0] = 100.0
    for index in range(1, n_bars):
        open_price[index, 0] = close[index - 1, 0]
        close[index, 0] = open_price[index, 0] * 0.95
    return MarketDataset(
        dataset_id="8" * 64,
        symbols=dataset.symbols,
        timestamps=dataset.timestamps,
        features=dataset.features,
        global_features=dataset.global_features,
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=dataset.volume,
        funding_rate=dataset.funding_rate,
        tradable=dataset.tradable,
        feature_available=dataset.feature_available,
        feature_names=dataset.feature_names,
        global_feature_names=dataset.global_feature_names,
        periods_per_year=dataset.periods_per_year,
    )


def test_real_ppo_learns_trivial_causal_short_signal() -> None:
    pytest.importorskip("stable_baselines3")
    torch = pytest.importorskip("torch")
    dataset = _strong_downtrend_market()
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        start_index=0,
        stop_index=dataset.n_bars - 1,
        gross_budget=0.5,
        total_timesteps=4096,
        seed=61,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    observation = StrategyObservation(
        index=0,
        timestamp=dataset.timestamps[0],
        symbol=dataset.symbols[0],
        features=dataset.features[0, 0],
        feature_available=dataset.feature_available[0, 0],
        feature_staleness=dataset.resolved_array("feature_staleness")[0, 0],
        global_features=dataset.global_features[0],
        global_feature_available=dataset.resolved_array("global_feature_available")[0],
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )

    assert strategy.decide(observation) is PositionIntent.SHORT
    for name, parameter in strategy.policy.policy.named_parameters():
        assert torch.isfinite(parameter).all(), name
