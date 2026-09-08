from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.ppo import (
    PPOIntentStrategy,
    PPOTradingEnv,
    fit_ppo_strategy,
)


@dataclass
class SequenceStrategy:
    intents: tuple[PositionIntent, ...]
    index: int = 0

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        del observation
        intent = self.intents[self.index]
        self.index += 1
        return intent


class FakePolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def predict(self, observation: np.ndarray, *, deterministic: bool = True):
        assert observation.ndim == 1
        assert deterministic is True
        return np.asarray(self.action), None


class FakePPO(FakePolicy):
    last: FakePPO | None = None

    def __init__(self, policy: str, env: object, **kwargs: object) -> None:
        super().__init__(2)
        self.policy = policy
        self.env = env
        self.kwargs = kwargs
        self.learn_timesteps: int | None = None
        FakePPO.last = self

    def learn(self, total_timesteps: int) -> FakePPO:
        self.learn_timesteps = total_timesteps
        return self


def market() -> MarketDataset:
    close = np.asarray([[100.0], [100.0], [110.0], [121.0]])
    return MarketDataset(
        dataset_id="4" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.arange(4, dtype=np.float32).reshape(4, 1, 1),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((4, 1), 1_000_000.0),
        funding_rate=np.zeros((4, 1)),
        tradable=np.ones((4, 1), dtype=np.bool_),
        feature_available=np.ones((4, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def pooled_market() -> MarketDataset:
    close = np.asarray(
        [
            [100.0, 200.0],
            [100.0, 200.0],
            [110.0, 190.0],
            [121.0, 180.0],
        ]
    )
    features = np.zeros((4, 2, 1), dtype=np.float32)
    features[:, 0, 0] = np.arange(4, dtype=np.float32)
    features[:, 1, 0] = 100.0 + np.arange(4, dtype=np.float32)
    return MarketDataset(
        dataset_id="5" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((4, 2), 1_000_000.0),
        funding_rate=np.zeros((4, 2)),
        tradable=np.ones((4, 2), dtype=np.bool_),
        feature_available=np.ones((4, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def observation() -> StrategyObservation:
    return StrategyObservation(
        index=0,
        timestamp=np.datetime64("2026-01-01", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([0.5]),
        feature_available=np.asarray([True]),
        global_features=np.asarray([0.0]),
        global_feature_available=np.asarray([True]),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


def test_policy_action_mapping_is_short_flat_long() -> None:
    assert (
        PPOIntentStrategy(FakePolicy(0), feature_indices=(0,)).decide(observation())
        is PositionIntent.SHORT
    )
    assert (
        PPOIntentStrategy(FakePolicy(1), feature_indices=(0,)).decide(observation())
        is PositionIntent.FLAT
    )
    assert (
        PPOIntentStrategy(FakePolicy(2), feature_indices=(0,)).decide(observation())
        is PositionIntent.LONG
    )


def test_env_reward_and_quantity_hold_match_canonical_replay() -> None:
    dataset = market()
    intents = (PositionIntent.LONG, PositionIntent.LONG, PositionIntent.FLAT)
    replay = run_single_symbol_replay(
        dataset,
        SequenceStrategy(intents),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    env.reset(seed=7)
    rewards: list[float] = []
    for action in (2, 2, 1):
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        assert truncated is False
    assert terminated is True

    expected_rewards = [math.log1p(value) for value in replay.returns.values]
    np.testing.assert_allclose(rewards, expected_rewards)
    assert env.book.quantities == replay.book.quantities
    assert env.book.portfolio_value == replay.book.portfolio_value


def test_pooled_env_cycles_symbols_without_symbol_identity_in_observation() -> None:
    env = PPOTradingEnv(
        pooled_market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    first_observation, _ = env.reset(seed=3)
    assert env.active_symbol_index == 0
    assert first_observation.shape == (4,)

    second_observation, _ = env.reset()
    assert env.active_symbol_index == 1
    assert second_observation.shape == (4,)
    _, _, _, _, _ = env.step(2)
    assert env.book.quantities[0] == 0.0
    assert env.book.quantities[1] > 0.0

    env.reset()
    assert env.active_symbol_index == 0


def test_fit_uses_small_teacher_free_standard_ppo(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3",
        SimpleNamespace(PPO=FakePPO),
    )
    strategy = fit_ppo_strategy(
        market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        total_timesteps=256,
        seed=11,
    )

    fitted = FakePPO.last
    assert fitted is not None
    assert fitted.policy == "MlpPolicy"
    assert fitted.kwargs["policy_kwargs"] == {
        "net_arch": {"pi": [64, 64], "vf": [64, 64]}
    }
    assert fitted.kwargs["seed"] == 11
    assert fitted.kwargs["ent_coef"] == 0.0
    assert fitted.learn_timesteps == 256
    assert isinstance(strategy, PPOIntentStrategy)
