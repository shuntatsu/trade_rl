from dataclasses import replace
from inspect import signature

import numpy as np
import pytest

from tests.evaluation.test_allocation_rl_env import capability, parameters
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.evaluation.rl_allocation.training_source import allocation_training_source
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_digest,
)


@pytest.mark.parametrize(
    "field", ["global_features", "global_feature_available", "global_feature_names"]
)
def test_unused_global_inputs_do_not_enter_an_execution_consumption_receipt(field):
    env = capability().AllocationTradingEnv(**parameters())
    before = allocation_training_source(env)
    if field == "global_feature_names":
        value = ("unused-name",)
    else:
        value = getattr(env.dataset, field).copy()
        value[7, 0] = 7.0 if field == "global_features" else False
    env.dataset = replace(env.dataset, **{field: value})
    assert allocation_training_source(env) == before


def test_receipt_hashes_only_actual_sampled_rows_in_a_declared_episode_envelope():
    env = capability().AllocationTradingEnv(**parameters())
    try:
        before = allocation_training_source(
            env, decision_counts={6: 3}, observation_indices=(6,)
        )
    except TypeError:
        pytest.fail("The source receipt has no observed decision coverage")
    assert before["decision_indices"] == [6]
    assert before["decision_counts"] == [3]
    assert before["start_index"] == 6 and before["stop_index"] == 8
    close = env.dataset.close.copy()
    close[8:, 0] *= 2.0
    env.dataset = replace(
        env.dataset,
        close=close,
        mark_price=close,
        index_price=close,
        high=np.maximum(env.dataset.open, close),
        low=np.minimum(env.dataset.open, close),
    )
    assert (
        allocation_training_source(
            env, decision_counts={6: 3}, observation_indices=(6,)
        )
        == before
    )
    assert (
        allocation_training_source(
            env, decision_counts={6: 3, 7: 1}, observation_indices=(6, 7)
        )
        != before
    )


def test_nonterminal_bootstrap_observation_is_bound_separately_from_action_counts():
    assert "observation_indices" in signature(allocation_training_source).parameters
    args = parameters()
    env = capability().AllocationTradingEnv(**args)
    env.reset()
    observation, _, done, _, _ = env.step(0)
    assert not done and env.index == 7
    before = allocation_training_source(
        env, decision_counts={6: 1}, observation_indices=(6, 7)
    )
    values = env.dataset.features.copy()
    values[7, 0, 0] += 5.0
    args["dataset"] = replace(env.dataset, features=values)
    owner = env.stream.vintages[0]
    args["stream"] = replace(
        env.stream,
        packets=tuple(
            replace(
                packet,
                feature_values=tuple(float(x) for x in values[7, 0]),
                expected_simple_return=owner.model.predict(values[7, 0]),
            )
            if packet.as_of == env.dataset.timestamps[7]
            else packet
            for packet in env.stream.packets
        ),
    )
    changed = capability().AllocationTradingEnv(**args)
    changed.reset()
    new_observation = changed.step(0)[0]
    assert not np.array_equal(observation, new_observation)
    after = allocation_training_source(
        changed, decision_counts={6: 1}, observation_indices=(6, 7)
    )
    assert before["decision_counts"] == after["decision_counts"] == [1]
    assert before["observation_indices"] == after["observation_indices"] == [6, 7]
    assert before["feature_consumption_digest"] != after["feature_consumption_digest"]
    assert (
        before["execution_consumption_digest"] == after["execution_consumption_digest"]
    )


def test_effective_session_bar_duration_is_an_execution_consumption_input():
    args = parameters()
    receipts = []
    for hours in (1.0, 0.5):
        args["dataset"] = replace(
            args["dataset"],
            calendar_kind=MarketCalendarKind.SESSION,
            nominal_bar_hours=hours,
        )
        objective, clock = args["bound"].objective, args["bound"].clock
        profile = AllocationRuntimeProfile(
            objective.economics_digest,
            objective.risk_digest,
            objective.capital.initial_equities[0],
            objective.capital.currency,
            clock.decision_interval_seconds,
            clock.economic_horizon_seconds,
            calendar_kind=MarketCalendarKind.SESSION.value,
            execution_bar_hours=hours,
        )
        digest = allocation_recipe_digest(
            args["action_contract"],
            ("signal",),
            allocator=args["allocator"],
            expected_horizon_seconds=3600,
            runtime_profile=profile,
        )
        args["bound"] = replace(
            args["bound"],
            objective=replace(objective, deployment_recipe_digest=digest),
        )
        receipts.append(
            allocation_training_source(capability().AllocationTradingEnv(**args))
        )
    assert (
        receipts[0]["execution_consumption_digest"]
        != receipts[1]["execution_consumption_digest"]
    )


def test_unsupported_rollout_batch_geometry_rejects_before_loading_a_trainer(
    monkeypatch,
):
    from trade_rl.evaluation.rl_allocation import training

    args = parameters()
    args["bound"] = replace(
        args["bound"], clock=replace(args["bound"].clock, rollout_steps=65)
    )
    env = capability().AllocationTradingEnv(**args)

    def unexpected_import(name):
        pytest.fail(f"Invalid rollout was admitted to trainer dependency {name}")

    monkeypatch.setattr(training.importlib, "import_module", unexpected_import)
    with pytest.raises(ValueError, match="batch"):
        training.build_allocation_ppo(env)


@pytest.mark.parametrize("processing_capacity", [False, True])
def test_capacity_reference_rows_bind_only_the_actual_volume_source(
    processing_capacity,
):
    args = parameters()
    args["execution_cost"] = replace(
        args["execution_cost"], processing_bar_volume_capacity=processing_capacity
    )
    objective = args["bound"].objective
    economics = MarketExecutor(
        args["dataset"], args["execution_cost"], insolvency_valuation="retain_debt"
    ).execution_policy_digest
    recipe = allocation_recipe_digest(
        args["action_contract"],
        ("signal",),
        allocator=args["allocator"],
        expected_horizon_seconds=3600,
        runtime_profile=AllocationRuntimeProfile(
            economics, objective.risk_digest, 1000.0, "USD", 3600, 7200
        ),
    )
    args["bound"] = replace(
        args["bound"],
        objective=replace(
            objective, economics_digest=economics, deployment_recipe_digest=recipe
        ),
    )
    receipts, fills = [], []
    for volume in (0.01, 10000.0):
        values = args["dataset"].volume.copy()
        values[6, 0] = volume
        args["dataset"] = replace(args["dataset"], volume=values)
        env = capability().AllocationTradingEnv(**args)
        env.reset()
        env.step(3)
        fills.append(env.book.quantities[0])
        receipts.append(
            allocation_training_source(
                env, decision_counts={6: 1}, observation_indices=(6, 7)
            )
        )
    if processing_capacity:
        assert fills == [5.0, 5.0]
        assert receipts[0] == receipts[1]
    else:
        assert fills[0] < fills[1] == 5.0
        assert (
            receipts[0]["execution_consumption_digest"]
            != receipts[1]["execution_consumption_digest"]
        )
