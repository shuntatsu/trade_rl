from __future__ import annotations

import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.rl.ppo import (
    PPOIntentStrategy,
    PPOTradingEnv,
    fit_ppo_strategy,
)


class FakePPO:
    last: FakePPO | None = None

    def __init__(self, policy: str, env: object, **kwargs: object) -> None:
        self.policy = policy
        self.env = env
        self.kwargs = kwargs
        self.learn_timesteps: int | None = None
        FakePPO.last = self

    def learn(self, total_timesteps: int) -> FakePPO:
        self.learn_timesteps = total_timesteps
        return self

    def predict(self, observation: np.ndarray, *, deterministic: bool = True):
        assert deterministic is True
        return np.asarray(2), None


class FakeDummyVecEnv:
    last: FakeDummyVecEnv | None = None

    def __init__(self, env_fns: list[Callable[[], object]]) -> None:
        self.envs = [cast(Any, env_fn()) for env_fn in env_fns]
        self.num_envs = len(self.envs)
        FakeDummyVecEnv.last = self


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
        dataset_id="6" * 64,
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


def install_fake_sb3(monkeypatch: pytest.MonkeyPatch) -> None:
    FakePPO.last = None
    FakeDummyVecEnv.last = None
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=FakePPO))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda threads: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3.common",
        SimpleNamespace(),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3.common.vec_env",
        SimpleNamespace(DummyVecEnv=FakeDummyVecEnv),
    )


def interleaved_fit(**overrides: object) -> PPOIntentStrategy:
    kwargs: dict[str, object] = {
        "dataset": pooled_market(),
        "feature_indices": (0,),
        "fit_symbol_indices": (0, 1),
        "start_index": 0,
        "stop_index": 3,
        "gross_budget": 0.5,
        "total_timesteps": 256,
        "seed": 11,
        "training_layout": "interleaved",
        "rollout_steps_per_env": 32,
    }
    kwargs.update(overrides)
    return cast(Any, fit_ppo_strategy)(**kwargs)


def test_default_fit_preserves_single_env_and_default_rollout_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)

    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        total_timesteps=256,
        seed=11,
    )

    fitted = FakePPO.last
    assert fitted is not None
    assert isinstance(fitted.env, PPOTradingEnv)
    assert fitted.env.symbol_indices == (0, 1)
    assert "n_steps" not in fitted.kwargs
    assert "batch_size" not in fitted.kwargs
    assert fitted.kwargs["policy_kwargs"] == {
        "net_arch": {"pi": [64, 64], "vf": [64, 64]}
    }
    assert fitted.kwargs["seed"] == 11
    assert fitted.kwargs["ent_coef"] == 0.0
    assert fitted.kwargs["device"] == "cpu"
    assert fitted.learn_timesteps == 256
    assert isinstance(strategy, PPOIntentStrategy)


def test_default_fit_preserves_stochastic_execution_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    cost = ExecutionCostConfig(slippage_std=0.01)

    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        total_timesteps=256,
        seed=11,
        execution_cost=cost,
    )

    fitted = FakePPO.last
    assert fitted is not None
    assert isinstance(fitted.env, PPOTradingEnv)
    assert fitted.env.execution_cost is cost
    assert isinstance(strategy, PPOIntentStrategy)


def test_interleaved_fit_uses_one_fixed_env_per_fit_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)

    strategy = fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        fit_symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        total_timesteps=256,
        seed=11,
        training_layout="interleaved",
        rollout_steps_per_env=32,
    )

    vector = FakeDummyVecEnv.last
    fitted = FakePPO.last
    assert vector is not None
    assert fitted is not None
    assert fitted.env is vector
    assert vector.num_envs == 2
    assert [env.symbol_indices for env in vector.envs] == [(0,), (1,)]
    for expected_symbol, env in zip(("BTCUSDT", "ETHUSDT"), vector.envs, strict=True):
        first, first_info = env.reset(seed=7)
        second, second_info = env.reset()
        assert first.shape == second.shape == (5,)
        assert first_info["symbol"] == second_info["symbol"] == expected_symbol
    assert fitted.policy == "MlpPolicy"
    assert fitted.kwargs["policy_kwargs"] == {
        "net_arch": {"pi": [64, 64], "vf": [64, 64]}
    }
    assert fitted.kwargs["seed"] == 11
    assert fitted.kwargs["ent_coef"] == 0.0
    assert fitted.kwargs["n_steps"] == 32
    assert fitted.kwargs["batch_size"] == 64
    assert fitted.learn_timesteps == 256
    assert isinstance(strategy, PPOIntentStrategy)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training_layout": "unknown"}, "training_layout"),
        ({"training_layout": True}, "training_layout"),
        (
            {"training_layout": "sequential", "rollout_steps_per_env": 32},
            "sequential",
        ),
        ({"rollout_steps_per_env": None}, "rollout_steps_per_env"),
        ({"rollout_steps_per_env": 0}, "rollout_steps_per_env"),
        ({"rollout_steps_per_env": True}, "rollout_steps_per_env"),
        ({"rollout_steps_per_env": 31}, "divisible"),
        ({"fit_symbol_indices": (0, 0)}, "unique"),
    ],
)
def test_interleaved_fit_rejects_invalid_training_contract(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    install_fake_sb3(monkeypatch)

    with pytest.raises(ValueError, match=message):
        interleaved_fit(**overrides)


def test_interleaved_fit_rejects_stochastic_execution_slippage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)

    with pytest.raises(ValueError, match="deterministic execution"):
        interleaved_fit(execution_cost=ExecutionCostConfig(slippage_std=0.01))


def test_interleaved_fit_accepts_implicit_full_symbol_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)

    interleaved_fit(fit_symbol_indices=None)

    vector = FakeDummyVecEnv.last
    assert vector is not None
    assert [env.symbol_indices for env in vector.envs] == [(0,), (1,)]
