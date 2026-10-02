from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from tests.strategies.test_ppo_intent import (
    FakeAdam,
    FakeFlattenExtractor,
    FakePPO,
    FakeTanh,
    market,
)
from trade_rl.strategies.rl.ppo import fit_ppo_strategy
from trade_rl.strategies.rl.ppo_training import (
    PPO_DEFAULT_GAE_LAMBDA,
    PPO_DEFAULT_GAMMA,
    PPO_REWARD_SCHEMA,
    ppo_training_objective_contract_payload,
    validated_ppo_gamma,
)


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    FakePPO.last = None
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3",
        SimpleNamespace(PPO=FakePPO),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda threads: None,
            nn=SimpleNamespace(Tanh=FakeTanh),
            optim=SimpleNamespace(Adam=FakeAdam),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3.common.torch_layers",
        SimpleNamespace(FlattenExtractor=FakeFlattenExtractor),
    )


def test_ppo_training_objective_contract_binds_economic_reward_and_discount() -> None:
    assert ppo_training_objective_contract_payload() == {
        "schema": "ppo_training_objective_v1",
        "reward_schema": PPO_REWARD_SCHEMA,
        "reward_scope": "per_symbol_account_after_cost_log_return",
        "gamma": PPO_DEFAULT_GAMMA,
        "gae_lambda": PPO_DEFAULT_GAE_LAMBDA,
        "normalize_advantage": True,
    }


@pytest.mark.parametrize("value", [True, 0.0, -0.1, 1.0000001, math.nan, math.inf])
def test_ppo_gamma_rejects_values_outside_open_closed_unit_interval(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="ppo_gamma"):
        validated_ppo_gamma(value)


def test_fit_ppo_strategy_passes_explicit_gamma_without_changing_gae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)

    fit_ppo_strategy(
        market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        total_timesteps=256,
        seed=11,
        gamma=0.9975,
    )

    fitted = FakePPO.last
    assert fitted is not None
    assert fitted.kwargs["gamma"] == pytest.approx(0.9975)
    assert fitted.kwargs["gae_lambda"] == pytest.approx(PPO_DEFAULT_GAE_LAMBDA)
    assert (
        fitted.kwargs["normalize_advantage"]
        is ppo_training_objective_contract_payload(gamma=0.9975)[
            "normalize_advantage"
        ]
    )
