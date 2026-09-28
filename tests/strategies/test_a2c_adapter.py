from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.rl_family_comparison.contract import fixed_comparison_contract
from trade_rl.strategies import (
    A2CIntentStrategy,
    fit_a2c_strategy,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.a2c import effective_a2c_timesteps


class FakeTanh:
    pass


class FakeRMSprop:
    pass


class FakeFlattenExtractor:
    pass


class FakeA2C:
    last: FakeA2C | None = None

    def __init__(self, policy: str, env: object, **kwargs: Any) -> None:
        self.policy_name = policy
        self.env = env
        self.kwargs = kwargs
        self.num_timesteps = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.learn_argument: int | None = None
        self.action = 2
        FakeA2C.last = self

    def learn(self, total_timesteps: int) -> FakeA2C:
        self.learn_argument = total_timesteps
        rollout = self.kwargs["n_steps"]
        self.num_timesteps = math.ceil(total_timesteps / rollout) * rollout
        return self

    def predict(self, observation: np.ndarray, *, deterministic: bool = True):
        assert deterministic is True
        assert observation.ndim == 1
        return np.asarray(self.action), None


def market(
    *,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    bars: int = 5,
) -> MarketDataset:
    symbol_count = len(symbols)
    close = np.full((bars, symbol_count), 100.0, dtype=np.float64)
    close *= np.arange(1, bars + 1, dtype=np.float64).reshape(-1, 1)
    features = np.arange(bars * symbol_count, dtype=np.float32).reshape(
        bars, symbol_count, 1
    )
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=symbols,
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((bars, symbol_count), 1_000_000.0),
        funding_rate=np.zeros((bars, symbol_count)),
        tradable=np.ones((bars, symbol_count), dtype=np.bool_),
        feature_available=np.ones((bars, symbol_count, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def install_fake_sb3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(A2C=FakeA2C))
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3.common.torch_layers",
        SimpleNamespace(FlattenExtractor=FakeFlattenExtractor),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda value: None,
            nn=SimpleNamespace(Tanh=FakeTanh),
            optim=SimpleNamespace(RMSprop=FakeRMSprop),
        ),
    )


def test_a2c_intent_strategy_uses_shared_three_action_semantics() -> None:
    class Policy:
        def __init__(self, action: int) -> None:
            self.action = action
            self.deterministic: bool | None = None

        def predict(self, observation: np.ndarray, *, deterministic: bool = True):
            assert observation.shape == (5,)
            self.deterministic = deterministic
            return np.asarray(self.action), None

    observation = StrategyObservation(
        index=0,
        timestamp=np.datetime64("2026-01-01", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([0.5]),
        feature_available=np.asarray([True]),
        global_features=np.asarray([0.0]),
        global_feature_available=np.asarray([True]),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
        feature_staleness=np.asarray([0.0]),
    )

    policy = Policy(2)
    strategy = A2CIntentStrategy(
        policy,
        feature_indices=(0,),
        feature_names=("signal",),
    )

    assert strategy.decide(observation) is PositionIntent.LONG
    assert policy.deterministic is True
    assert strategy.feature_indices == (0,)
    assert strategy.feature_names == ("signal",)


@pytest.mark.parametrize("feature_name", [[[]], [{"nested": "name"}]])
def test_a2c_adapter_rejects_unhashable_feature_names_as_value_error(
    feature_name: object,
) -> None:
    class Policy:
        def predict(self, observation: np.ndarray, *, deterministic: bool = True):
            return np.asarray(1), None

    with pytest.raises(ValueError, match="feature_names"):
        A2CIntentStrategy(
            Policy(),
            feature_indices=(0,),
            feature_names=(feature_name,),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("feature_index", ([0], {"index": 0}))
def test_a2c_adapter_rejects_unhashable_feature_indices_as_value_error(
    feature_index: object,
) -> None:
    class Policy:
        def predict(self, observation: np.ndarray, *, deterministic: bool = True):
            return np.asarray(1), None

    with pytest.raises(ValueError, match="feature_indices"):
        A2CIntentStrategy(
            Policy(),
            feature_indices=(feature_index,),  # type: ignore[arg-type]
        )


def test_a2c_fit_uses_explicit_cpu_config_and_five_step_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)

    strategy = fit_a2c_strategy(
        market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=4,
        gross_budget=1.0,
        total_timesteps=6,
        seed=19,
    )

    model = FakeA2C.last
    assert model is not None
    algorithm = fixed_comparison_contract()["algorithm_settings"]["a2c"]
    policy_kwargs = model.kwargs["policy_kwargs"]
    optimizer_kwargs = policy_kwargs["optimizer_kwargs"]
    assert policy_kwargs["net_arch"] == {
        "pi": algorithm["network"],
        "vf": algorithm["network"],
    }
    assert {
        "rollout_steps": model.kwargs["n_steps"],
        "learning_rate": model.kwargs["learning_rate"],
        "gamma": model.kwargs["gamma"],
        "gae_lambda": model.kwargs["gae_lambda"],
        "entropy_coefficient": model.kwargs["ent_coef"],
        "value_coefficient": model.kwargs["vf_coef"],
        "max_gradient_norm": model.kwargs["max_grad_norm"],
        "rmsprop_epsilon": model.kwargs["rms_prop_eps"],
        "normalize_advantage": model.kwargs["normalize_advantage"],
        "optimizer": "RMSprop"
        if policy_kwargs["optimizer_class"] is FakeRMSprop
        else "<unexpected>",
        "network": policy_kwargs["net_arch"]["pi"],
        "activation": "Tanh"
        if policy_kwargs["activation_fn"] is FakeTanh
        else "<unexpected>",
        "use_rms_prop": model.kwargs["use_rms_prop"],
        "rmsprop_alpha": optimizer_kwargs["alpha"],
        "rmsprop_momentum": optimizer_kwargs["momentum"],
        "rmsprop_weight_decay": optimizer_kwargs["weight_decay"],
        "rmsprop_centered": optimizer_kwargs["centered"],
        "rmsprop_capturable": optimizer_kwargs["capturable"],
        "rmsprop_foreach": optimizer_kwargs["foreach"],
        "rmsprop_maximize": optimizer_kwargs["maximize"],
        "rmsprop_differentiable": optimizer_kwargs["differentiable"],
        "rmsprop_optimizer_epsilon": optimizer_kwargs["eps"],
        "stats_window_size": model.kwargs["stats_window_size"],
        "use_sde": model.kwargs["use_sde"],
        "shared_feature_extractor": policy_kwargs["share_features_extractor"],
        "orthogonal_initialization": policy_kwargs["ortho_init"],
        "feature_extractor": "FlattenExtractor"
        if policy_kwargs["features_extractor_class"] is FakeFlattenExtractor
        else "<unexpected>",
    } == algorithm
    assert isinstance(strategy, A2CIntentStrategy)
    assert model.policy_name == "MlpPolicy"
    assert model.learn_argument == 6
    assert model.num_timesteps == 10
    assert model.kwargs["n_steps"] == 5
    assert model.kwargs["seed"] == 19
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["learning_rate"] == 7e-4
    assert model.kwargs["gamma"] == 0.99
    assert model.kwargs["gae_lambda"] == 1.0
    assert model.kwargs["ent_coef"] == 0.0
    assert model.kwargs["vf_coef"] == 0.5
    assert model.kwargs["max_grad_norm"] == 0.5
    assert model.kwargs["rms_prop_eps"] == 1e-5
    assert model.kwargs["use_rms_prop"] is True
    assert model.kwargs["normalize_advantage"] is False
    assert model.kwargs["policy_kwargs"]["net_arch"] == {
        "pi": [64, 64],
        "vf": [64, 64],
    }
    assert model.kwargs["policy_kwargs"]["activation_fn"] is FakeTanh
    assert model.kwargs["policy_kwargs"]["optimizer_class"] is FakeRMSprop
    assert model.kwargs["policy_kwargs"]["optimizer_kwargs"] == {
        "alpha": 0.99,
        "eps": 1e-5,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "centered": False,
        "capturable": False,
        "foreach": None,
        "maximize": False,
        "differentiable": False,
    }
    assert strategy.fit_metadata is not None
    assert strategy.fit_metadata.requested_timesteps == 6
    assert strategy.fit_metadata.effective_timesteps == 10
    assert strategy.fit_metadata.fit_symbols == ("BTCUSDT", "ETHUSDT")
    assert strategy.fit_metadata.required_coverage_timesteps == 8
    assert strategy.fit_metadata.nominal_full_episodes_per_symbol == 1


def test_a2c_rounding_uses_integer_arithmetic_for_large_requests() -> None:
    requested = 10**400 + 1
    assert effective_a2c_timesteps(requested) == requested + 4


def test_a2c_sequential_coverage_rounds_to_its_rollout_size_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    FakeA2C.last = None

    with pytest.raises(ValueError, match="symbol coverage"):
        fit_a2c_strategy(
            market(),
            feature_indices=(0,),
            start_index=0,
            stop_index=4,
            gross_budget=1.0,
            total_timesteps=5,
        )

    assert FakeA2C.last is None


def test_a2c_single_symbol_coverage_is_checked_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    FakeA2C.last = None

    with pytest.raises(ValueError, match="symbol coverage"):
        fit_a2c_strategy(
            market(symbols=("BTCUSDT",), bars=9),
            feature_indices=(0,),
            start_index=0,
            stop_index=8,
            gross_budget=1.0,
            total_timesteps=1,
        )

    assert FakeA2C.last is None
