from dataclasses import replace
from datetime import UTC, datetime
from importlib import import_module

import numpy as np
import pytest

from tests.evaluation.test_forecast_allocation import setup
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_digest,
)


def capability():
    try:
        return import_module("trade_rl.evaluation.rl_allocation.env")
    except ModuleNotFoundError as error:
        if error.name.startswith("trade_rl.evaluation.rl_allocation"):
            pytest.fail("The new canonical allocation learning environment is missing")
        raise


def parameters(*, stop=8):
    from trade_rl.strategies.allocation_action import AllocationActionContract

    executor, _, _, current = setup()
    contract = AllocationActionContract(mode="direct", scale=0.5)
    dataset = executor.dataset
    costs = tuple(
        replace(
            current["estimates"],
            decision_time=dataset.timestamps[i],
            available_at=dataset.timestamps[i],
            horizon_end=dataset.timestamps[i] + np.timedelta64(1, "h"),
        )
        for i in range(6, stop)
    )
    clock = FinancialClockContract(
        decision_interval_seconds=3600,
        execution_interval_seconds=3600,
        reward_interval_seconds=3600,
        economic_horizon_seconds=(stop - 6) * 3600,
        rollout_steps=4,
        gamma=1.0,
        gae_lambda=0.95,
        reward_schema="equity_delta_v1",
    )
    objective = ObjectiveContract(
        capital=CapitalContract("independent_symbol", "USD", (1000.0,)),
        evaluation_start=datetime(2026, 1, 1, 6, tzinfo=UTC),
        evaluation_stop_exclusive=datetime(2026, 1, 1, stop, tzinfo=UTC),
        terminal_valuation="marked_continuation",
        economics_digest=MarketExecutor(
            dataset, executor.cost, insolvency_valuation="retain_debt"
        ).execution_policy_digest,
        risk_digest=content_digest(current["pretrade_risk"].config),
        deployment_recipe_digest=allocation_recipe_digest(
            contract,
            ("signal",),
            allocator=current["allocator"],
            expected_horizon_seconds=3600,
            runtime_profile=AllocationRuntimeProfile(
                MarketExecutor(
                    dataset, executor.cost, insolvency_valuation="retain_debt"
                ).execution_policy_digest,
                content_digest(current["pretrade_risk"].config),
                1000.0,
                "USD",
                3600,
                (stop - 6) * 3600,
            ),
        ),
    )
    return dict(
        dataset=dataset,
        stream=current["stream"],
        estimates=costs,
        bound=BoundObjectiveClock(objective, clock),
        action_contract=contract,
        allocator=current["allocator"],
        execution_cost=executor.cost,
        risk_config=current["pretrade_risk"].config,
        feature_indices=(0,),
        symbol_index=0,
        start_index=6,
        stop_index=stop,
        account_id="independent-S0",
    )


def test_actual_execution_calendar_and_bar_duration_must_match_declared_recipe():
    from trade_rl.data.contracts import MarketCalendarKind

    args = parameters()
    args["dataset"] = replace(
        args["dataset"], calendar_kind=MarketCalendarKind.SESSION, nominal_bar_hours=0.5
    )
    with pytest.raises(ValueError, match="recipe"):
        capability().AllocationTradingEnv(**args)


def test_actual_after_cost_equity_reward_telescopes_and_preserves_finite_horizon():
    env = capability().AllocationTradingEnv(**parameters())
    observation, _ = env.reset(seed=9)
    assert env.action_space.n == 4
    assert env.observation_space.contains(observation)
    assert observation[-1] == 1.0
    assert env.book.as_of_index == 6
    assert env.book.as_of_dataset_id == env.dataset.dataset_id
    _, first, terminated, truncated, info = env.step(3)
    # Target .5 at decision 100 buys 5 at next open 100, fee 1.
    # Canonical next close 110 gives equity 499 + 550 = 1049.
    assert first == pytest.approx(0.049)
    assert not terminated and not truncated
    assert env.index == env.book.as_of_index == 7
    assert env.book.cash == 499.0
    assert env.book.quantities[0] == 5.0
    assert info["execution"].book is env.book
    observation, second, terminated, truncated, info = env.step(2)
    # FLAT sells 5 at next open 110, fee 1.1; no extra terminal liquidation.
    assert second == pytest.approx(-0.0011)
    assert first + second == pytest.approx((1047.9 - 1000.0) / 1000.0)
    assert env.book.cash == pytest.approx(1047.9)
    assert env.book.quantities[0] == 0.0
    assert env.index == env.book.as_of_index == 8
    assert terminated and not truncated
    assert info["terminal_valuation"] == "marked_continuation"
    assert np.array_equal(observation, np.zeros_like(observation))
    with pytest.raises(RuntimeError, match="terminal"):
        env.step(2)


@pytest.mark.parametrize(
    "field,value", [("gamma", 0.99), ("reward_schema", "net_log_return_v1")]
)
def test_profit_learning_rejects_a_misaligned_runtime_clock(field, value):
    kwargs = parameters()
    kwargs["bound"] = replace(
        kwargs["bound"], clock=replace(kwargs["bound"].clock, **{field: value})
    )
    with pytest.raises(ValueError, match="equity.*gamma|profit.*clock"):
        capability().AllocationTradingEnv(**kwargs)


@pytest.mark.parametrize(
    "field", ["economics_digest", "risk_digest", "deployment_recipe_digest"]
)
def test_declarations_must_match_actual_runtime_profiles_and_action_recipe(field):
    kwargs = parameters()
    kwargs["bound"] = replace(
        kwargs["bound"],
        objective=replace(kwargs["bound"].objective, **{field: "0" * 64}),
    )
    with pytest.raises(ValueError, match="profile|recipe"):
        capability().AllocationTradingEnv(**kwargs)


def test_episode_boundary_does_not_liquidate_a_marked_continuation_position():
    env = capability().AllocationTradingEnv(**parameters(stop=7))
    env.reset(seed=3)
    _, reward, terminated, truncated, info = env.step(3)
    assert reward == pytest.approx(0.049)
    assert terminated and not truncated
    assert env.book.quantities[0] == 5.0
    assert env.book.fill_count == 1
    assert env.book.total_cost == 1.0
    assert info["execution"].next_index == 7


@pytest.mark.parametrize("field", ["action_contract", "allocator"])
def test_reset_cannot_admit_a_changed_action_recipe_with_a_cached_digest(field):
    env = capability().AllocationTradingEnv(**parameters())
    env.reset(seed=9)
    value = (
        replace(env.action_contract, scale=0.2)
        if field == "action_contract"
        else replace(env.allocator, upper_weight=0.25)
    )
    setattr(env, field, value)
    with pytest.raises(ValueError, match="recipe|profile"):
        env.reset(seed=9)


def test_declared_dates_outside_nanosecond_range_cannot_wrap_into_dataset_dates():
    kwargs = parameters()
    first = datetime(2500, 1, 1, 6, tzinfo=UTC)
    last = datetime(2500, 1, 1, 8, tzinfo=UTC)
    times = np.datetime64(first.replace(tzinfo=None), "ns") + (
        np.arange(kwargs["dataset"].n_bars) - 6
    ) * np.timedelta64(1, "h")
    kwargs["dataset"] = replace(kwargs["dataset"], timestamps=times, available_at=None)
    kwargs["bound"] = replace(
        kwargs["bound"],
        objective=replace(
            kwargs["bound"].objective,
            evaluation_start=first,
            evaluation_stop_exclusive=last,
        ),
    )
    with pytest.raises(ValueError, match="range|represent|overflow"):
        capability().AllocationTradingEnv(**kwargs)


def test_new_recipe_binds_the_actual_capital_and_economics_profile():
    env = capability().AllocationTradingEnv(**parameters())
    original = env.recipe_digest
    env.execution_cost = replace(env.execution_cost, fee_rate=0.1)
    with pytest.raises(ValueError, match="profile|recipe"):
        env.reset(seed=9)
    assert env.recipe_digest != original


def test_reset_cannot_use_cached_capital_after_the_bound_objective_changes():
    env = capability().AllocationTradingEnv(**parameters())
    env.bound = replace(
        env.bound,
        objective=replace(
            env.bound.objective,
            capital=CapitalContract("independent_symbol", "USD", (2000.0,)),
        ),
    )
    with pytest.raises(ValueError, match="bound|capital|recipe"):
        env.reset(seed=9)


def test_early_insolvency_keeps_signed_debt_in_the_same_book_and_profit_reward():
    kwargs = parameters(stop=9)
    kwargs["allocator"] = replace(kwargs["allocator"], lower_weight=-1.0)
    recipe = allocation_recipe_digest(
        kwargs["action_contract"],
        ("signal",),
        allocator=kwargs["allocator"],
        expected_horizon_seconds=3600,
        runtime_profile=AllocationRuntimeProfile(
            kwargs["bound"].objective.economics_digest,
            content_digest(kwargs["risk_config"]),
            1000.0,
            "USD",
            3600,
            10800,
        ),
    )
    kwargs["bound"] = replace(
        kwargs["bound"],
        objective=replace(kwargs["bound"].objective, deployment_recipe_digest=recipe),
    )
    dataset = kwargs["dataset"]
    close = dataset.close.copy()
    close[7:] = 400.0
    kwargs["dataset"] = replace(
        dataset,
        close=close,
        mark_price=close,
        index_price=close,
        high=np.maximum(dataset.open, close),
        low=np.minimum(dataset.open, close),
    )
    env = capability().AllocationTradingEnv(**kwargs)
    env.reset(seed=9)
    _, reward, terminated, truncated, _ = env.step(1)
    # Sell5 at100 pays fee1: cash1499. Mark liability2000; debt=-501.
    assert env.book.cash == env.book.portfolio_value == -501.0
    assert reward == -1.501
    assert env.index == 7 < env.stop_index
    assert terminated and not truncated
    assert env.book.quantities[0] == 0


def test_policy_seed_does_not_reseed_frozen_execution_randomness():
    kwargs = parameters()
    cost = replace(kwargs["execution_cost"], slippage_std=0.01, random_seed=41)
    kwargs["execution_cost"] = cost
    profile = MarketExecutor(
        kwargs["dataset"], cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    kwargs["bound"] = replace(
        kwargs["bound"],
        objective=replace(kwargs["bound"].objective, economics_digest=profile),
    )
    # Rebind the complete recipe to these declared stochastic economics.
    # Obtain the recipe independently via the pure lower contract.
    recipe = allocation_recipe_digest(
        kwargs["action_contract"],
        ("signal",),
        allocator=kwargs["allocator"],
        expected_horizon_seconds=3600,
        runtime_profile=AllocationRuntimeProfile(
            profile, content_digest(kwargs["risk_config"]), 1000.0, "USD", 3600, 7200
        ),
    )
    kwargs["bound"] = replace(
        kwargs["bound"],
        objective=replace(kwargs["bound"].objective, deployment_recipe_digest=recipe),
    )
    env = capability().AllocationTradingEnv(**kwargs)
    outcomes = []
    for seed in (1, 17):
        env.reset(seed=seed)
        outcomes.append(env.step(3)[1])
    assert outcomes[0] == outcomes[1]


def test_runtime_cannot_report_a_different_dataset_from_the_actual_executor():
    env = capability().AllocationTradingEnv(**parameters())
    env.reset(seed=9)
    env.dataset = replace(env.dataset, global_feature_names=("other",))
    with pytest.raises(ValueError, match="executor|Dataset"):
        env.step(3)
