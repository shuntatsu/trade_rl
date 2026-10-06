"""Actual SB3 consumption of the declared chronological allocation schedule."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

from tests.evaluation.test_allocation_rl_env import parameters
from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
from tests.strategies.test_allocation_protocol_receipt import protocol
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.objectives import BoundObjectiveClock
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.training_schedule import (
    AllocationTrainingScheduleEnv,
    allocation_training_window,
)
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_digest,
)
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
)


def env_for(start: int, stop: int) -> AllocationTradingEnv:
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
        rollout_steps=4,
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

    def utc(index: int) -> datetime:
        stamp = dataset.timestamps[index].astype("datetime64[us]").astype(datetime)
        return stamp.replace(tzinfo=UTC)

    args["bound"] = BoundObjectiveClock(
        replace(
            objective,
            evaluation_start=utc(start),
            evaluation_stop_exclusive=utc(stop),
            deployment_recipe_digest=recipe,
        ),
        clock,
    )
    args["account_id"] = f"scheduled-fit-{start}-{stop}"
    return AllocationTradingEnv(**opt_in(args))


def scheduled() -> tuple[
    AllocationTrainingScheduleEnv, tuple[AllocationTradingEnv, ...]
]:
    first, second = env_for(6, 8), env_for(8, 10)
    windows = (allocation_training_window(first), allocation_training_window(second))
    schedule = AllocationTrainingSchedule(
        windows,
        train_window_ids=tuple(window.window_id for window in windows),
    )
    return AllocationTrainingScheduleEnv(schedule, (second, first)), (first, second)


def capability():
    from trade_rl.evaluation.rl_allocation.scheduled_training import (
        fit_allocation_ppo_schedule,
    )

    return fit_allocation_ppo_schedule


def test_actual_schedule_fit_cycles_declared_accounts_and_records_reuse():
    env, _ = scheduled()
    fit = capability()(
        env,
        total_timesteps=8,
        seed=7,
        training_protocol=protocol(n_steps=4, batch_size=2, n_epochs=2),
    )
    receipt = fit.receipt
    assert fit.model.num_timesteps == receipt["actual_timesteps"] == 8
    assert receipt["schema"] == "allocation_ppo_schedule_fit_receipt_v1"
    assert receipt["schedule_digest"] == env.schedule.digest
    assert receipt["protocol"]["ppo"]["n_steps"] == 4
    # DummyVecEnv immediately prepares the next episode after a true terminal.
    # The final prepared reset is retained as runtime evidence even though no
    # actor decision consumes it before this fit returns.
    assert receipt["usage"]["windows"] == [
        {
            "window_id": env.schedule.train_window_ids[0],
            "reset_count": 3,
            "decision_count": 4,
        },
        {
            "window_id": env.schedule.train_window_ids[1],
            "reset_count": 2,
            "decision_count": 4,
        },
    ]
    assert sum(row["decision_count"] for row in receipt["usage"]["windows"]) == 8
    assert receipt["consumption"]["windows"] == [
        {
            "window_id": env.schedule.train_window_ids[0],
            "decision_indices": [6, 7],
            "decision_counts": [2, 2],
            "observation_indices": [6, 7],
        },
        {
            "window_id": env.schedule.train_window_ids[1],
            "decision_indices": [8, 9],
            "decision_counts": [2, 2],
            "observation_indices": [8, 9],
        },
    ]
    assert "experience_count" not in canonical_json_bytes(receipt).decode()


def test_schedule_fit_requires_explicit_protocol_and_exact_rollout_budget(monkeypatch):
    from trade_rl.evaluation.rl_allocation import scheduled_training as module

    env, _ = scheduled()
    monkeypatch.setattr(
        module,
        "construct_protocol_ppo",
        lambda *a, **k: pytest.fail("backend construction reached"),
    )
    with pytest.raises(ValueError, match="explicit"):
        module.fit_allocation_ppo_schedule(env, total_timesteps=8)
    with pytest.raises(ValueError, match="rollout"):
        module.fit_allocation_ppo_schedule(
            env,
            total_timesteps=6,
            training_protocol=protocol(n_steps=4, batch_size=2),
        )


def test_schedule_fit_revalidates_every_window_after_learning(monkeypatch):
    from trade_rl.evaluation.rl_allocation import scheduled_training as module

    env, children = scheduled()
    original = module.construct_protocol_ppo

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        learn = model.learn

        def altered(**learn_kwargs):
            result = learn(**learn_kwargs)
            child = children[1]
            values = child.dataset.features.copy()
            values[8, 0, 0] += 9
            child.dataset = child.executor.dataset = replace(
                child.dataset, features=values
            )
            return result

        model.learn = altered
        return model

    monkeypatch.setattr(module, "construct_protocol_ppo", build)
    with pytest.raises(ValueError, match="source|schedule"):
        module.fit_allocation_ppo_schedule(
            env,
            total_timesteps=4,
            seed=7,
            training_protocol=protocol(n_steps=4, batch_size=2, n_epochs=1),
        )


def test_schedule_fit_receipt_is_detached_from_mutable_runtime_usage():
    env, _ = scheduled()
    fit = capability()(
        env,
        total_timesteps=4,
        seed=0,
        training_protocol=protocol(n_steps=4, batch_size=2, n_epochs=1),
    )
    before = canonical_json_bytes(fit.receipt)
    env._usage[env.schedule.train_window_ids[0]]["decision_count"] += 100
    assert canonical_json_bytes(fit.receipt) == before


def test_schedule_fit_v4_reconstructs_frozen_prefix_once_and_records_it():
    from tests.evaluation.test_allocation_preprocessing_runtime import frozen_args

    child = AllocationTradingEnv(**frozen_args(stop=8))
    window = allocation_training_window(child)
    schedule = AllocationTrainingSchedule(
        (window,), train_window_ids=(window.window_id,)
    )
    env = AllocationTrainingScheduleEnv(schedule, (child,))
    fit = capability()(
        env,
        total_timesteps=2,
        seed=0,
        training_protocol=protocol(n_steps=2, batch_size=2, n_epochs=1),
    )
    receipt = fit.receipt
    assert receipt["preprocessing_fit"]["schema"] == (
        "allocation_ppo_preprocessing_fit_receipt_v1"
    )
    assert receipt["preprocessing_fit"]["declaration_digest"] == (
        child.feature_preprocessing.digest
    )
    assert receipt["consumption"]["windows"] == [
        {
            "window_id": window.window_id,
            "decision_indices": [6, 7],
            "decision_counts": [1, 1],
            "observation_indices": [6, 7],
        }
    ]
