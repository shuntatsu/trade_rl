"""Fixed synthetic learnability protocol, never market/economic selection."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module

import numpy as np
import pytest

from tests.evaluation.test_allocation_rl_env import parameters
from tests.simulation.test_stateful_execution_characterization import _normalize
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.forecasts.simple_prequential import (
    fit_prequential_simple_ridge,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_digest,
)

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")


def trainer():
    try:
        return import_module("trade_rl.evaluation.rl_allocation.training")
    except ModuleNotFoundError as error:
        if error.name == "trade_rl.evaluation.rl_allocation.training":
            pytest.fail("The actual allocation PPO training capability is missing")
        raise


def synthetic_episode(schedule_seed, *, price_response=0.01, fee=0.0005):
    # G2 protocol fixed before fitting: 16 causal sign decisions, .01 price
    # response, .0005 actual fee; four flat mature prefix labels fit a cash
    # forecast. Held-out schedules 100/101/102 are not used by RL fitting.
    rng = np.random.default_rng(schedule_seed)
    cue = rng.choice([-1.0, 1.0], size=23)
    close = np.full((23, 1), 100.0)
    for i in range(6, 22):
        close[i + 1, 0] = close[i, 0] * (1.0 + price_response * cue[i])
    opens = close.copy()
    opens[7:, 0] = close[6:-1, 0]
    shape = close.shape
    dataset = MarketDataset(
        dataset_id=content_digest({"synthetic_cue_episode_v1": schedule_seed}),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.timedelta64(schedule_seed if schedule_seed >= 100 else 0, "D")
        + np.arange(23) * np.timedelta64(1, "h"),
        features=np.stack([np.ones(23), cue], axis=1)[:, None, :].astype(np.float32),
        feature_names=("constant", "current_cue"),
        feature_available=np.ones((23, 1, 2), dtype=bool),
        global_features=np.zeros(shape),
        global_feature_names=("regime",),
        symbols=("S0",),
        volume_units=(VolumeUnit.BASE_ASSET,),
        periods_per_year=8760,
        open=opens,
        close=close,
        mark_price=close,
        index_price=close,
        high=np.maximum(opens, close),
        low=np.minimum(opens, close),
        volume=np.full(shape, 100_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
        split_factor=np.ones(shape),
    )
    times = dataset.timestamps
    block = ForecastBlock(
        times[5], times[5] + np.timedelta64(15, "m"), times[6], times[22]
    )
    stream = fit_prequential_simple_ridge(
        dataset, blocks=(block,), feature_indices=(0,), horizon_hours=1
    )
    assert all(packet.expected_simple_return == 0.0 for packet in stream.packets)
    costs = tuple(
        HorizonCostEstimates(
            "S0",
            times[i],
            times[i],
            times[i + 1],
            "declared-synthetic-fee",
            buy_cost=fee,
            sell_cost=fee,
        )
        for i in range(6, 22)
    )
    action = AllocationActionContract("direct", 0.5)
    allocator = AfterCostTargetAllocator()
    cost = replace(ExecutionCostConfig.zero(), fee_rate=fee, max_participation_rate=1.0)
    risk = PreTradeRiskConfig(max_abs_weight=1.0, max_turnover=None)
    objective = ObjectiveContract(
        CapitalContract("independent_symbol", "USD", (1000.0,)),
        datetime(2026, 1, 1, 6, tzinfo=UTC)
        + timedelta(days=schedule_seed if schedule_seed >= 100 else 0),
        datetime(2026, 1, 1, 22, tzinfo=UTC)
        + timedelta(days=schedule_seed if schedule_seed >= 100 else 0),
        "marked_continuation",
        MarketExecutor(
            dataset, cost, insolvency_valuation="retain_debt"
        ).execution_policy_digest,
        content_digest(risk),
        allocation_recipe_digest(
            action,
            ("current_cue",),
            allocator=allocator,
            expected_horizon_seconds=3600,
            runtime_profile=AllocationRuntimeProfile(
                MarketExecutor(
                    dataset, cost, insolvency_valuation="retain_debt"
                ).execution_policy_digest,
                content_digest(risk),
                1000.0,
                "USD",
                3600,
                16 * 3600,
            ),
        ),
    )
    clock = FinancialClockContract(
        3600, 3600, 3600, 16 * 3600, 32, 1.0, 0.95, "equity_delta_v1"
    )
    return AllocationTradingEnv(
        dataset,
        stream=stream,
        estimates=costs,
        bound=BoundObjectiveClock(objective, clock),
        action_contract=action,
        allocator=allocator,
        execution_cost=cost,
        risk_config=risk,
        feature_indices=(1,),
        symbol_index=0,
        start_index=6,
        stop_index=22,
        account_id="synthetic-independent-S0",
    )


def rollout(env, action):
    observation, _ = env.reset(seed=41)
    trace, reward, aligned = [], 0.0, 0
    while True:
        cue = (
            float(env.dataset.features[env.index, 0, 1])
            if env.dataset.n_features == 2
            else 0.0
        )
        observation, increment, done, truncated, info = env.step(action(observation))
        assert not truncated
        reward += increment
        aligned += int(float(env.book.weights[0]) * cue > 0)
        trace.append(_normalize(info))
        if done:
            return reward, aligned / len(trace), trace


@pytest.mark.parametrize("seed", [0, 7, 17])
def test_actual_ppo_learns_current_cue_and_frozen_reload_preserves_full_execution(
    seed, tmp_path
):
    module = trainer()
    env = synthetic_episode(10)
    untrained = module.build_allocation_ppo(env, seed=seed)
    initial = [
        rollout(
            synthetic_episode(s),
            lambda x: int(untrained.predict(x, deterministic=True)[0]),
        )[0]
        for s in (100, 101, 102)
    ]
    trained = module.fit_allocation_ppo(env, total_timesteps=4096, seed=seed)
    assert trained.manifest["training"]["actual_timesteps"] == 4096
    assert trained.manifest["training"]["requested_timesteps"] == 4096
    assert trained.manifest["training"]["clock"]["gamma"] == 1.0
    assert trained.manifest["training"]["source"]["decision_indices"] == list(
        range(6, 22)
    )
    assert trained.manifest["training"]["source"]["decision_counts"] == [256] * 16
    scores = []
    for schedule, initial_reward in zip((100, 101, 102), initial):
        check = synthetic_episode(schedule)
        result = rollout(
            check,
            lambda x: trained.action(x, runtime_recipe_digest=check.recipe_digest),
        )
        # Fixed acceptance before fitting: cue-conditioned EFFECTIVE exposure,
        # positive after-cost gain over cash, and improvement over initialization.
        assert result[1] >= 0.8
        assert result[0] > 0.025
        assert result[0] - initial_reward > 0.02
        scores.append(result)
    from trade_rl.strategies.rl.allocation_artifact import (
        load_allocation_policy,
        save_allocation_policy,
    )

    digest = save_allocation_policy(tmp_path / "policy", trained)
    loaded = load_allocation_policy(
        tmp_path / "policy",
        expected_digest=digest,
        expected_recipe_digest=env.recipe_digest,
    )
    for schedule, expected in zip((100, 101, 102), scores):
        check = synthetic_episode(schedule)
        assert (
            rollout(
                check,
                lambda x: loaded.action(x, runtime_recipe_digest=check.recipe_digest),
            )
            == expected
        )
    with pytest.raises(ValueError, match="recipe"):
        loaded.action(
            np.zeros(env.observation_space.shape), runtime_recipe_digest="0" * 64
        )


def test_rollout_boundary_bootstraps_but_declared_horizon_is_terminal():
    module = trainer()
    kwargs = parameters(stop=10)
    kwargs["bound"] = replace(
        kwargs["bound"], clock=replace(kwargs["bound"].clock, rollout_steps=2)
    )
    env = AllocationTradingEnv(**kwargs)
    model = module.build_allocation_ppo(env, seed=0)
    calls = []
    original = model.rollout_buffer.compute_returns_and_advantage

    def capture(last_values, dones):
        calls.append((last_values.detach().cpu().numpy().copy(), dones.copy()))
        original(last_values, dones)

    model.rollout_buffer.compute_returns_and_advantage = capture
    _, callback = model._setup_learn(4)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, 2)
    assert env.index == 8
    assert calls[-1][1].tolist() == [False]
    model.collect_rollouts(model.env, callback, model.rollout_buffer, 2)
    assert calls[-1][1].tolist() == [True]
    assert (
        env.index == 6
    )  # genuine terminal auto-reset by DummyVecEnv, not buffer reset


def test_training_budget_cannot_silently_round_to_more_transitions():
    with pytest.raises(ValueError, match="rollout|multiple"):
        trainer().fit_allocation_ppo(synthetic_episode(10), total_timesteps=33, seed=0)


@pytest.mark.parametrize("seed", [0, 7, 17])
def test_actual_ppo_avoids_fee_only_trading_on_held_out_cues(seed):
    # Separately fixed G2 protocol before its first fit: no price response,
    # .005 fee, unchanged 4096-step budget/network and all three policy seeds.
    trained = trainer().fit_allocation_ppo(
        synthetic_episode(10, price_response=0.0, fee=0.005),
        total_timesteps=4096,
        seed=seed,
    )
    for schedule in (100, 101, 102):
        check = synthetic_episode(schedule, price_response=0.0, fee=0.005)
        result = rollout(
            check,
            lambda x: trained.action(x, runtime_recipe_digest=check.recipe_digest),
        )
        assert result[0] == 0.0
        assert check.book.fill_count == 0


def test_observed_source_counts_cover_actual_terminal_resets_only():
    env = AllocationTradingEnv(**parameters(stop=7))
    policy = trainer().fit_allocation_ppo(env, total_timesteps=4, seed=0)
    source = policy.manifest["training"]["source"]
    assert source["start_index"] == 6 and source["stop_index"] == 7
    assert source["decision_indices"] == [6]
    assert source["decision_counts"] == [4]
    assert source["observation_indices"] == [6]


def test_actual_short_rollout_receipt_includes_nonterminal_critic_bootstrap():
    args = parameters(stop=10)
    args["bound"] = replace(
        args["bound"], clock=replace(args["bound"].clock, rollout_steps=2)
    )
    env = AllocationTradingEnv(**args)
    policy = trainer().fit_allocation_ppo(env, total_timesteps=2, seed=0)
    source = policy.manifest["training"]["source"]
    assert source["decision_indices"] == [6, 7]
    assert source["decision_counts"] == [1, 1]
    assert source["observation_indices"] == [6, 7, 8]
    assert not env._terminated and env.index == 8
