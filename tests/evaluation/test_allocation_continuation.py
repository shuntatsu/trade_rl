from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime

import numpy as np
import pytest

from tests.evaluation.test_allocation_rl_env import parameters
from trade_rl.artifacts import content_digest
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.evaluation.objectives import BoundObjectiveClock
from trade_rl.evaluation.robustness.walk_forward.stitching import (
    FoldOOSResult,
    StitchMode,
    stitch_oos,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.rl.allocation_policy import (
    AllocationRuntimeProfile,
    allocation_recipe_digest,
)

_SHARED_DATASET = parameters(stop=10)["dataset"]


def capability():
    from trade_rl.evaluation.rl_allocation.continuation import (
        allocation_state_digest,
        export_allocation_continuation,
        resume_allocation_continuation,
    )

    return (
        allocation_state_digest,
        export_allocation_continuation,
        resume_allocation_continuation,
    )


def env_for(start, stop, *, cost=None, account_id="continuous-S0", dataset=None):
    from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv

    args = parameters(stop=10)
    source = _SHARED_DATASET if dataset is None else dataset
    args["dataset"] = source
    args["start_index"], args["stop_index"] = start, stop
    args["account_id"] = account_id
    args["estimates"] = tuple(
        estimate
        for estimate in args["estimates"]
        if source.timestamps[start] <= estimate.decision_time < source.timestamps[stop]
    )
    if cost is not None:
        args["execution_cost"] = cost
    execution = MarketExecutor(
        source, args["execution_cost"], insolvency_valuation="retain_debt"
    ).execution_policy_digest
    objective = args["bound"].objective
    clock = replace(
        args["bound"].clock,
        economic_horizon_seconds=(stop - start) * 3600,
        rollout_steps=max(1, stop - start),
    )
    profile = AllocationRuntimeProfile(
        execution,
        content_digest(args["risk_config"]),
        objective.capital.initial_equities[0],
        objective.capital.currency,
        clock.decision_interval_seconds,
        clock.economic_horizon_seconds,
        calendar_kind=MarketCalendarKind(source.calendar_kind).value,
        execution_bar_hours=source.bar_hours,
    )
    recipe = allocation_recipe_digest(
        args["action_contract"],
        ("signal",),
        allocator=args["allocator"],
        expected_horizon_seconds=3600,
        runtime_profile=profile,
    )

    def utc(index):
        stamp = source.timestamps[index].astype("datetime64[us]").astype(datetime)
        return stamp.replace(tzinfo=UTC)

    args["bound"] = BoundObjectiveClock(
        replace(
            objective,
            evaluation_start=utc(start),
            evaluation_stop_exclusive=utc(stop),
            economics_digest=execution,
            risk_digest=content_digest(args["risk_config"]),
            deployment_recipe_digest=recipe,
        ),
        clock,
    )
    return AllocationTradingEnv(**args)


def execution_prices(info):
    return tuple(
        event.execution_price
        for event in info["execution"].order_events
        if event.execution_price is not None
    )


def book_economics(book):
    values = {}
    for field in fields(book):
        if field.name in {"as_of_index", "as_of_dataset_id", "_exact_quantities"}:
            continue
        value = getattr(book, field.name)
        values[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
    values["exact_quantities"] = tuple(str(value) for value in book.exact_quantities)
    return values


def run(env, actions):
    rewards, prices = [], []
    for action in actions:
        _, reward, done, truncated, info = env.step(action)
        assert not truncated
        rewards.append(reward)
        prices.extend(execution_prices(info))
    return rewards, tuple(prices), done


def test_continuation_matches_uninterrupted_economics_and_execution_rng():
    _, export, resume = capability()
    base = parameters(stop=10)
    cost = replace(
        base["execution_cost"],
        slippage_std=0.01,
        random_seed=73,
        max_participation_rate=1.0,
    )
    whole = env_for(6, 10, cost=cost)
    first = env_for(6, 8, cost=cost)
    second = env_for(8, 10, cost=cost)

    whole.reset(seed=9)
    whole_rewards, whole_prices, done = run(whole, (3, 1, 3, 1))
    assert done

    first.reset(seed=9)
    left_rewards, left_prices, done = run(first, (3, 1))
    assert done
    handoff = export(first)
    _, info = resume(second, handoff)
    assert info["opening_state_digest"] == handoff.state_digest
    right_rewards, right_prices, done = run(second, (3, 1))
    assert done

    assert left_rewards + right_rewards == pytest.approx(whole_rewards)
    assert left_prices + right_prices == pytest.approx(whole_prices)
    assert book_economics(second.book) == pytest.approx(book_economics(whole.book))
    assert handoff.consumed


def test_handoff_preserves_pending_gtc_orders_and_exact_state_identity():
    digest, export, resume = capability()
    base = parameters(stop=10)
    cost = replace(
        base["execution_cost"],
        max_participation_rate=0.00001,
        processing_bar_volume_capacity=False,
    )
    first = env_for(6, 8, cost=cost)
    second = env_for(8, 10, cost=cost)
    opening, _ = first.reset(seed=3)
    assert first.observation_space.contains(opening)
    first.step(3)
    _, _, done, _, _ = first.step(3)
    assert done
    assert first.order_book.active_orders

    closing_digest = digest(first)
    active_before = first.order_book.active_orders
    handoff = export(first)
    observation, info = resume(second, handoff)
    assert second.observation_space.contains(observation)
    assert info["opening_state_digest"] == closing_digest == handoff.state_digest
    assert second.order_book.active_orders == active_before
    assert digest(second) == closing_digest
    assert second.book.as_of_index == first.stop_index == second.start_index


def test_actual_handoff_digests_satisfy_existing_continuous_stitch_contract():
    digest, export, resume = capability()
    first = env_for(6, 8)
    second = env_for(8, 10)
    first.reset()
    opening = digest(first)
    rewards_a, _, done = run(first, (3, 0))
    assert done
    boundary = export(first)
    resume(second, boundary)
    rewards_b, _, done = run(second, (0, 2))
    assert done
    closing = digest(second)

    stitched = stitch_oos(
        (
            FoldOOSResult(
                0,
                6,
                8,
                ReturnSeries(tuple(rewards_a), ReturnKind.DECISION_STEP, 8760),
                opening_state_digest=opening,
                closing_state_digest=boundary.state_digest,
            ),
            FoldOOSResult(
                1,
                8,
                10,
                ReturnSeries(tuple(rewards_b), ReturnKind.DECISION_STEP, 8760),
                opening_state_digest=boundary.state_digest,
                closing_state_digest=closing,
            ),
        ),
        mode=StitchMode.CONTINUOUS_ACCOUNT,
    )
    assert stitched.boundaries == ((6, 8), (8, 10))
    assert stitched.gaps == ()


@pytest.mark.parametrize(
    "change,match",
    [
        ("gap", "contiguous"),
        ("account", "account"),
        ("economics", "execution"),
    ],
)
def test_resume_rejects_gap_account_or_execution_profile_mismatch(change, match):
    _, export, resume = capability()
    first = env_for(6, 8)
    first.reset()
    run(first, (3, 0))
    handoff = export(first)

    if change == "gap":
        next_env = env_for(9, 10)
    elif change == "account":
        next_env = env_for(8, 10, account_id="different-account")
    else:
        base = parameters(stop=10)
        next_env = env_for(
            8,
            10,
            cost=replace(base["execution_cost"], fee_rate=0.003),
        )
    with pytest.raises(ValueError, match=match):
        resume(next_env, handoff)
    assert not handoff.consumed


def test_continuation_is_one_shot_and_cannot_branch_execution_rng():
    _, export, resume = capability()
    first = env_for(6, 8)
    second = env_for(8, 10)
    duplicate = env_for(8, 10)
    first.reset()
    run(first, (3, 0))
    handoff = export(first)
    resume(second, handoff)
    with pytest.raises(RuntimeError, match="consumed"):
        resume(duplicate, handoff)


def test_export_rejects_live_or_economically_terminated_accounts():
    _, export, _ = capability()
    live = env_for(6, 8)
    live.reset()
    with pytest.raises(RuntimeError, match="terminal"):
        export(live)

    terminated = env_for(6, 8)
    terminated.reset()
    terminated.book.terminate("drawdown_stop")
    terminated._terminated = True
    with pytest.raises(ValueError, match="economic"):
        export(terminated)


def test_same_terminal_cannot_export_two_execution_rng_branches():
    _, export, _ = capability()
    first = env_for(6, 8)
    first.reset()
    run(first, (3, 0))
    export(first)
    with pytest.raises(RuntimeError, match="already exported"):
        export(first)


def test_continuation_identity_is_immutable_after_export():
    _, export, _ = capability()
    first = env_for(6, 8)
    first.reset()
    run(first, (3, 0))
    handoff = export(first)
    with pytest.raises(FrozenInstanceError):
        handoff.next_index = 9  # type: ignore[misc]


def test_failed_resume_rolls_target_back_and_does_not_consume(monkeypatch):
    _, export, resume = capability()
    first = env_for(6, 8)
    target = env_for(8, 10)
    first.reset()
    run(first, (3, 0))
    handoff = export(first)

    def fail_observation():
        raise ValueError("synthetic observation failure")

    monkeypatch.setattr(target, "_observation", fail_observation)
    with pytest.raises(ValueError, match="synthetic observation failure"):
        resume(target, handoff)

    assert target._terminated
    assert not handoff.consumed
    assert not hasattr(target, "book")
    assert not hasattr(target, "order_book")
    assert not hasattr(target, "index")
    assert not hasattr(target, "decision")
