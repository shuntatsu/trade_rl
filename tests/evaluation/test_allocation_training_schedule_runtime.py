from dataclasses import replace
from datetime import UTC, datetime

import pytest

from tests.evaluation.test_allocation_rl_env import parameters
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.objectives import BoundObjectiveClock
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_digest,
)


def capability():
    from trade_rl.evaluation.rl_allocation.training_schedule import (
        AllocationTrainingScheduleEnv,
        allocation_training_window,
    )

    return AllocationTrainingScheduleEnv, allocation_training_window


def env_for(start, stop):
    args = parameters(stop=10)
    dataset = args["dataset"]
    args["start_index"], args["stop_index"] = start, stop
    args["estimates"] = tuple(
        estimate
        for estimate in args["estimates"]
        if dataset.timestamps[start]
        <= estimate.decision_time
        < dataset.timestamps[stop]
    )
    clock = replace(
        args["bound"].clock,
        economic_horizon_seconds=(stop - start) * 3600,
        rollout_steps=2,
    )
    objective = args["bound"].objective
    profile = AllocationRuntimeProfile(
        objective.economics_digest,
        objective.risk_digest,
        objective.capital.initial_equities[0],
        objective.capital.currency,
        clock.decision_interval_seconds,
        clock.economic_horizon_seconds,
    )
    recipe = allocation_recipe_digest(
        args["action_contract"],
        ("signal",),
        allocator=args["allocator"],
        expected_horizon_seconds=3600,
        runtime_profile=profile,
    )

    def utc(index):
        stamp = dataset.timestamps[index].astype("datetime64[us]").astype(datetime)
        return stamp.replace(tzinfo=UTC)

    objective = replace(
        objective,
        evaluation_start=utc(start),
        evaluation_stop_exclusive=utc(stop),
        deployment_recipe_digest=recipe,
    )
    args["bound"] = BoundObjectiveClock(objective, clock)
    args["account_id"] = f"scheduled-{start}-{stop}"
    return AllocationTradingEnv(**args)


def declared(first, second, *, extra=()):
    from trade_rl.strategies.rl.allocation_training_schedule import (
        AllocationTrainingSchedule,
    )

    _, make_window = capability()
    windows = (
        make_window(first),
        make_window(second),
        *extra,
    )
    return AllocationTrainingSchedule(
        windows,
        train_window_ids=(windows[0].window_id, windows[1].window_id),
    )


def test_runtime_cycles_only_at_true_episode_reset_and_preserves_declared_order():
    Runtime, make_window = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    schedule = declared(first, second)
    runtime = Runtime(schedule, (second, first))

    observation, info = runtime.reset(seed=7)
    assert observation.shape == first.observation_space.shape
    assert info["training_window_id"] == make_window(first).window_id
    assert runtime.active_window_id == make_window(first).window_id

    _, _, done, truncated, info = runtime.step(0)
    assert not done and not truncated
    assert info["training_window_id"] == make_window(first).window_id
    assert runtime.usage_payload()["windows"][0] == {
        "window_id": make_window(first).window_id,
        "reset_count": 1,
        "decision_count": 1,
    }

    with pytest.raises(RuntimeError, match="terminal"):
        runtime.reset()

    _, _, done, truncated, _ = runtime.step(0)
    assert done and not truncated
    _, info = runtime.reset()
    assert info["training_window_id"] == make_window(second).window_id
    assert runtime.active_window_id == make_window(second).window_id

    runtime.step(0)
    _, _, done, _, _ = runtime.step(0)
    assert done
    runtime.reset()
    assert runtime.active_window_id == make_window(first).window_id
    assert runtime.usage_payload()["windows"] == [
        {
            "window_id": make_window(first).window_id,
            "reset_count": 2,
            "decision_count": 2,
        },
        {
            "window_id": make_window(second).window_id,
            "reset_count": 1,
            "decision_count": 2,
        },
    ]


def test_runtime_rollout_boundary_is_not_a_window_or_account_boundary():
    Runtime, make_window = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    runtime = Runtime(declared(first, second), (first, second))
    runtime.reset()
    first_id = runtime.active_window_id
    runtime.step(0)
    # A PPO rollout may end here. No runtime method is called and no reset occurs.
    assert runtime.active_window_id == first_id == make_window(first).window_id
    assert runtime.usage_payload()["windows"][0]["reset_count"] == 1
    assert first.index == 7


def test_runtime_rejects_missing_duplicate_or_changed_declared_sources():
    Runtime, make_window = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    schedule = declared(first, second)

    with pytest.raises(ValueError, match="exactly"):
        Runtime(schedule, (first,))
    with pytest.raises(ValueError, match="unique"):
        Runtime(schedule, (first, first))

    values = second.dataset.features.copy()
    values[8, 0, 0] += 1
    second.dataset = second.executor.dataset = replace(second.dataset, features=values)
    with pytest.raises(ValueError, match="schedule|source|exactly"):
        Runtime(schedule, (first, second))

    assert schedule.training_windows[0].window_id == make_window(first).window_id


def test_runtime_requires_exact_dataset_lineage_even_when_causal_window_id_matches():
    Runtime, make_window = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    first_window, second_window = make_window(first), make_window(second)
    relined = replace(first_window, dataset_id="f" * 64)
    assert relined.window_id == first_window.window_id

    from trade_rl.strategies.rl.allocation_training_schedule import (
        AllocationTrainingSchedule,
    )

    schedule = AllocationTrainingSchedule(
        (relined, second_window),
        train_window_ids=(relined.window_id, second_window.window_id),
    )
    with pytest.raises(ValueError, match="lineage"):
        Runtime(schedule, (first, second))


def test_runtime_rejects_incompatible_recipe_or_financial_clock():
    Runtime, _ = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    schedule = declared(first, second)

    second.action_contract = replace(second.action_contract, scale=0.25)
    with pytest.raises(ValueError, match="recipe"):
        Runtime(schedule, (first, second))

    first, second = env_for(6, 8), env_for(8, 10)
    schedule = declared(first, second)
    second.bound = replace(
        second.bound, clock=replace(second.bound.clock, gae_lambda=0.5)
    )
    with pytest.raises(ValueError, match="clock"):
        Runtime(schedule, (first, second))


def test_validation_window_can_be_declared_but_never_supplied_to_training_runtime():
    Runtime, make_window = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    second_window = make_window(second)
    validation = replace(
        make_window(second, role="validation"),
        dataset_id="f" * 64,
        start_index=20,
        stop_index=22,
        decision_start_ns=second_window.terminal_time_ns + 3_600_000_000_000,
        terminal_time_ns=second_window.terminal_time_ns + 10_800_000_000_000,
        source_digest="e" * 64,
    )
    schedule = declared(first, second, extra=(validation,))
    runtime = Runtime(schedule, (first, second))
    assert validation.window_id not in {
        row["window_id"] for row in runtime.usage_payload()["windows"]
    }


def test_usage_payload_binds_schedule_and_never_calls_window_count_experience_count():
    Runtime, _ = capability()
    first, second = env_for(6, 8), env_for(8, 10)
    schedule = declared(first, second)
    runtime = Runtime(schedule, (first, second))
    runtime.reset()
    runtime.step(0)
    payload = runtime.usage_payload()
    assert payload["schema"] == "allocation_training_schedule_usage_v1"
    assert payload["schedule_digest"] == schedule.digest
    assert "experience_count" not in payload
    assert content_digest(payload)
