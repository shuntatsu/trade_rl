from dataclasses import replace
from enum import IntEnum
from typing import Any

import numpy as np
import pytest

from tests.evaluation.test_shared_cash_replay import _market
from tests.strategies.test_ppo_interleaved_training import (
    FakeDummyVecEnv,
    FakePPO,
    install_fake_sb3,
    pooled_market,
)
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.native_transition import execute_native_target_transition
from trade_rl.strategies.rl import ppo as ppo_module
from trade_rl.strategies.rl.ppo import PPOTradingEnv, fit_ppo_strategy


def test_explicit_training_risk_closes_after_drawdown_and_survives_reset() -> None:
    dataset = _market(np.array([[100.0], [100.0], [50.0], [50.0]]))
    config = PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        execution_cost=ExecutionCostConfig.zero(),
        risk_config=config,
    )
    for _ in range(2):
        env.reset(seed=0)
        env.step(2)
        env.step(2)
        assert env.book.max_drawdown == pytest.approx(0.25)
        env.step(2)
        assert np.all(env.book.quantities == 0)


def test_omitted_risk_and_explicit_none_have_identical_legacy_paths() -> None:
    kwargs: dict[str, Any] = dict(
        dataset=pooled_market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
    )
    default = PPOTradingEnv(**kwargs)
    explicit = PPOTradingEnv(**kwargs, risk_config=None)
    for _ in range(2):
        np.testing.assert_array_equal(
            default.reset(seed=7)[0], explicit.reset(seed=7)[0]
        )
        assert default.risk.config.drawdown_stop == 1.0
        for action in (2, 0, 1):
            a, b = default.step(action), explicit.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:4] == b[1:4]
            np.testing.assert_array_equal(
                default.book.quantities, explicit.book.quantities
            )


def test_native_transition_is_shared_only_by_configured_risk_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _market(np.full((4, 1), 100.0))
    default = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        execution_cost=ExecutionCostConfig.zero(),
    )
    explicit = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        execution_cost=ExecutionCostConfig.zero(),
        risk_config=PreTradeRiskConfig(
            max_gross=0.25,
            max_abs_weight=0.25,
            max_turnover=None,
        ),
    )
    native_calls: list[tuple[int, tuple[float, ...]]] = []

    def observe_native_transition(
        executor: object,
        book: object,
        pretrade_risk: object,
        target_weights: object,
        *,
        start_index: int,
        **kwargs: object,
    ) -> tuple[object, object, tuple[int, ...]]:
        native_calls.append(
            (start_index, tuple(float(value) for value in target_weights))
        )
        return execute_native_target_transition(
            executor,
            book,
            pretrade_risk,
            target_weights,
            start_index=start_index,
            **kwargs,
        )  # type: ignore[arg-type]

    monkeypatch.setattr(
        ppo_module,
        "execute_native_target_transition",
        observe_native_transition,
        raising=False,
    )
    default.reset(seed=7)
    explicit.reset(seed=7)
    default.step(2)
    explicit.step(2)

    assert native_calls == [(0, (0.5,))]
    assert default.book.weights[0] == pytest.approx(0.5)
    assert explicit.book.weights[0] == pytest.approx(0.25)


def test_configured_ppo_normalizes_integer_subclass_start_index() -> None:
    class StartIndex(IntEnum):
        FIRST = 0
        STOP = 3

    env = PPOTradingEnv(
        _market(np.full((4, 1), 100.0)),
        feature_indices=(0,),
        start_index=StartIndex.FIRST,
        stop_index=StartIndex.STOP,
        gross_budget=0.5,
        execution_cost=ExecutionCostConfig.zero(),
        risk_config=PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
        ),
    )

    assert type(env.start_index) is int
    assert type(env.stop_index) is int
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(2)

    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert env.index == 1


@pytest.mark.parametrize("layout", ["sequential", "interleaved"])
def test_fitter_propagates_explicit_risk_to_every_environment(
    layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_sb3(monkeypatch)
    config = PreTradeRiskConfig(max_gross=0.5, max_abs_weight=0.1, max_turnover=None)
    fit_ppo_strategy(
        pooled_market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        total_timesteps=128,
        seed=0,
        training_layout=layout,
        rollout_steps_per_env=32 if layout == "interleaved" else None,
        risk_config=config,
    )
    assert FakePPO.last is not None
    environments = (
        FakeDummyVecEnv.last.envs
        if layout == "interleaved" and FakeDummyVecEnv.last
        else [FakePPO.last.env]
    )
    for env in environments:
        env.reset(seed=0)
        assert env.risk.config == config
    if len(environments) > 1:
        assert environments[0].risk is not environments[1].risk


def test_invalid_risk_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="risk_config"):
        PPOTradingEnv(
            pooled_market(),
            feature_indices=(0,),
            start_index=0,
            stop_index=3,
            gross_budget=0.1,
            risk_config={},
        )  # type: ignore[arg-type]


def test_explicit_risk_cannot_exceed_executor_leverage_but_default_adapts() -> None:
    costs = replace(ExecutionCostConfig.zero(), max_leverage=0.2)
    kwargs: dict[str, Any] = dict(
        dataset=pooled_market(),
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.3,
        execution_cost=costs,
    )
    with pytest.raises(ValueError, match="max_gross.*max_leverage"):
        PPOTradingEnv(
            **kwargs, risk_config=PreTradeRiskConfig(max_gross=0.5, max_abs_weight=0.5)
        )
    default = PPOTradingEnv(**kwargs)
    default.reset()
    default.step(2)
    assert default.risk.config.max_gross == 0.2


@pytest.mark.parametrize("layout", ["sequential", "interleaved"])
def test_incompatible_risk_is_rejected_before_model_creation(
    layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_sb3(monkeypatch)
    with pytest.raises(ValueError, match="max_gross.*max_leverage"):
        fit_ppo_strategy(
            pooled_market(),
            feature_indices=(0,),
            start_index=0,
            stop_index=3,
            gross_budget=0.1,
            total_timesteps=64,
            seed=0,
            execution_cost=replace(ExecutionCostConfig.zero(), max_leverage=0.2),
            risk_config=PreTradeRiskConfig(max_gross=0.5, max_abs_weight=0.5),
            training_layout=layout,
            rollout_steps_per_env=32 if layout == "interleaved" else None,
        )
    assert FakePPO.last is None
