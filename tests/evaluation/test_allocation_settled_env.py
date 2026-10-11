"""Literal settled-account reward, native reserve and admission oracles."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from importlib import import_module, util

import numpy as np
import pytest

from tests.evaluation.test_allocation_nonrl_walk_forward import fixture
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.evaluation import allocation_decision
from trade_rl.evaluation import allocation_reserved_closing as reserved_closing
from trade_rl.evaluation.objectives import CapitalContract, ObjectiveContract
from trade_rl.evaluation.rl_allocation import training as marked_training
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_settled_clock import (
    SettledAllocationClockContract,
)


def capability(name):
    assert util.find_spec(name) is not None, "distinct settled owner is missing"
    return import_module(name)


def arguments(
    *,
    actor_steps=2,
    reserve_steps=2,
    fee=0.002,
    side=1,
    terminal_price=100.0,
    funding=False,
    cash_rate=0.0,
    borrow_rate=0.0,
    dividend=0.0,
    lot_size=0.0,
    actor_volume=None,
    partial=False,
    reserve_features=False,
    cost_changes=None,
):
    """Build declarations from the existing literal fixture, never a fit/reset."""
    BoundSettledAllocationClock = capability(
        "trade_rl.evaluation.rl_allocation.settled_binding"
    ).BoundSettledAllocationClock
    allocation_recipe_payload_v4 = capability(
        "trade_rl.strategies.rl.allocation_recipe_v4"
    ).allocation_recipe_payload_v4
    _, (marked,) = fixture(
        ranges=((6, 6 + actor_steps),),
        signals=(side, 0, 0, 0),
        fee=fee,
        mode="direct",
    )
    start, horizon = 6, actor_steps + reserve_steps
    prices = np.full((12, 1), 100.0)
    # Actor processing is at start+1; reserve first processes start+P+1.
    prices[start + actor_steps + 1 :] = terminal_price
    volume = marked.dataset.volume.copy()
    if actor_volume is not None:
        volume[start + 1 : start + actor_steps + 1] = actor_volume
    if partial:
        volume[start + actor_steps + 1 : start + horizon + 1] = 1.0
    available = marked.dataset.feature_available.copy()
    if not reserve_features:
        available[start + actor_steps :] = False
    dataset = replace(
        marked.dataset,
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        mark_price=prices,
        index_price=prices,
        volume=volume,
        feature_available=available,
        feature_staleness=np.where(available, 0.0, 1.0).astype(np.float32),
        funding_rate=np.zeros((12, 1)),
        funding_price_rate=prices * (0.001 if funding else 0.0),
        funding_due=np.full((12, 1), funding, dtype=bool),
        cash_rate=np.full(12, cash_rate),
        borrow_rate=np.full((12, 1), borrow_rate),
        dividend=np.full((12, 1), dividend),
        lot_size=np.full((12, 1), lot_size),
    )
    packets = []
    for index, packet in zip(range(6, 10), marked.stream.packets):
        if index < start + actor_steps:
            packets.append(
                replace(
                    packet,
                    decision_close=float(dataset.close[index, 0]),
                    feature_values=(float(dataset.features[index, 0, 0]),),
                    expected_simple_return=float(dataset.features[index, 0, 0]) * 0.1,
                )
            )
    # Missing reserve packets plus unavailable reserve features catch accidental
    # decision preparation there. Native financing/price processing stays valid.
    stream = replace(marked.stream, packets=tuple(packets))
    allocator = (
        replace(marked.allocator, lower_weight=-0.5, upper_weight=0.0)
        if side == -1
        else marked.allocator
    )
    cost = replace(marked.execution_cost, **(cost_changes or {}))
    clock = SettledAllocationClockContract(
        actor_steps=actor_steps,
        reserve_steps=reserve_steps,
        execution_interval_seconds=3600,
        rollout_steps=2,
        gamma=1.0,
        gae_lambda=0.95,
    )
    schema = AllocationObservationSchema(("signal",), 8, 1000.0, horizon)
    economics = MarketExecutor(
        dataset, cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    profile = AllocationRuntimeProfile(
        economics,
        content_digest(marked.risk_config),
        1000.0,
        "USD",
        3600,
        horizon * 3600,
        calendar_kind=MarketCalendarKind(dataset.calendar_kind).value,
        execution_bar_hours=dataset.bar_hours,
    )
    recipe = allocation_recipe_payload_v4(
        marked.action_contract,
        ("signal",),
        allocator=allocator,
        expected_horizon_seconds=3600,
        runtime_profile=profile,
        settled_clock=clock,
        observation_schema=schema,
    )
    objective = ObjectiveContract(
        CapitalContract("independent_symbol", "USD", (1000.0,)),
        datetime(2026, 1, 1, start, tzinfo=UTC),
        datetime(2026, 1, 1, start + horizon, tzinfo=UTC),
        "settled",
        economics,
        content_digest(marked.risk_config),
        content_digest(recipe),
        maximum_drawdown=marked.risk_config.drawdown_stop,
    )
    estimates = tuple(
        replace(estimate, buy_cost=fee, sell_cost=fee)
        for estimate in marked._estimates.values()
    )
    return {
        "dataset": dataset,
        "stream": stream,
        "estimates": estimates,
        "bound": BoundSettledAllocationClock(objective, clock),
        "action_contract": marked.action_contract,
        "allocator": allocator,
        "execution_cost": cost,
        "risk_config": marked.risk_config,
        "feature_indices": (0,),
        "symbol_index": 0,
        "start_index": start,
        "account_id": "literal-settled-S0",
        "expected_horizon_seconds": 3600,
        "observation_schema": schema,
    }, marked


def environment(**changes):
    args, _ = arguments(**changes)
    return capability(
        "trade_rl.evaluation.rl_allocation.settled_env"
    ).SettledAllocationTradingEnv(**args)


def detached(info):
    raw = info["settled_transition"]
    assert type(raw) is bytes, "the frontier must be detached before overwrites"
    payload = json.loads(raw)
    assert canonical_json_bytes(payload) == raw
    return payload


def assert_zero_terminal(observation, terminated, truncated):
    assert terminated is True
    assert truncated is False
    assert observation.dtype == np.float32
    assert np.array_equal(observation, np.zeros_like(observation))


def assert_row_chain(rows, starts):
    assert [row["start_index"] for row in rows] == starts
    assert [row["processing_index"] for row in rows] == [i + 1 for i in starts]
    for row in rows:
        assert row["clock_valid"] is True
        assert row["returned_clock_types"] == {
            "next_index": "int",
            "bars_advanced": "int",
        }
        assert row["native"]["execution"]["next_index"] == row["processing_index"]
        assert row["native"]["execution"]["bars_advanced"] == 1
        assert row["native"]["book"]["as_of_index"] == row["processing_index"]
    for before, after in zip(rows, rows[1:]):
        assert before["after_state_digest"] == after["before_state_digest"]


def test_p2_r2_fee_reward_h_fraction_and_native_chain():
    env = environment()
    assert not isinstance(env, AllocationTradingEnv), "distinct composition lane"
    initial, _ = env.reset(seed=101)
    horizon_field = env.observation_schema.fields.index("remaining_horizon_fraction")
    assert initial[horizon_field] == pytest.approx(1.0)
    executor, risk = env.executor, env.risk
    first, r0, t0, x0, info0 = env.step(np.int64(3))
    assert (r0, t0, x0) == (-0.001, False, False)
    assert first[horizon_field] == pytest.approx(3.0 / 4.0)
    assert (env.actor_step, env.index) == (1, 7)
    assert env.book.portfolio_value == pytest.approx(999.0)
    prefix = detached(info0)
    assert (prefix["actor_step"], prefix["actor_index"]) == (0, 6)
    assert_row_chain(prefix["rows"], [6])
    frozen_prefix = info0["settled_transition"]

    final, r1, t1, x1, info1 = env.step(np.int64(0))
    assert_zero_terminal(final, t1, x1)
    assert r1 == pytest.approx(-0.001)
    assert (env.actor_step, env.index) == (2, 10)
    assert env.book.portfolio_value == pytest.approx(998.0)
    assert env.book.cash == pytest.approx(998.0)
    assert np.array_equal(env.book.quantities, np.array([0.0]))
    assert not env.order_book.active_orders
    assert env.executor is executor and env.risk is risk
    transition = detached(info1)
    assert (transition["actor_step"], transition["actor_index"]) == (1, 7)
    rows = transition["rows"]
    assert [row["kind"] for row in rows] == ["actor", "reserve", "reserve"]
    assert_row_chain(rows, [7, 8, 9])
    assert prefix["rows"][-1]["after_state_digest"] == rows[0]["before_state_digest"]
    assert [row["native"]["book"]["equity"] for row in rows] == pytest.approx(
        [999.0, 998.0, 998.0]
    )
    witness = transition["settlement_witness"]
    assert (witness["actor_steps_processed"], witness["physical_steps_processed"]) == (
        2,
        4,
    )
    assert witness["reserve_steps_processed"] == 2
    assert witness["settled"] is True and witness["eligible"] is True
    assert info0["settled_transition"] == frozen_prefix, "no aliasing of prior rows"


def test_p1_r2_flat_account_processes_every_native_cash_carry_bar():
    env = environment(actor_steps=1, reserve_steps=2, fee=0.0, cash_rate=8.76)
    env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step(np.int64(0))
    assert_zero_terminal(observation, terminated, truncated)
    assert env.book.cash == pytest.approx(1003.003001)
    assert reward == pytest.approx(0.003003001)
    rows = detached(info)["rows"]
    assert_row_chain(rows, [6, 7, 8])
    assert [row["native"]["book"]["equity"] for row in rows] == pytest.approx(
        [1001.0, 1002.001, 1003.003001]
    )
    assert detached(info)["settlement_witness"]["reserve_steps_processed"] == 2


@pytest.mark.parametrize(
    "side, action, price, cash, reward",
    [(1, 3, 110.0, 1039.0, 0.039), (-1, 1, 90.0, 1041.0, 0.041)],
)
def test_p1_r1_literal_native_long_short_after_fee_and_funding(
    side, action, price, cash, reward
):
    env = environment(
        actor_steps=1,
        reserve_steps=1,
        fee=0.01,
        side=side,
        terminal_price=price,
        funding=True,
    )
    env.reset()
    observation, actual_reward, terminated, truncated, info = env.step(np.int64(action))
    assert_zero_terminal(observation, terminated, truncated)
    assert env.book.cash == pytest.approx(cash)
    assert env.book.portfolio_value == pytest.approx(cash)
    assert actual_reward == pytest.approx(reward)
    assert env.book.exact_quantities[0] == 0 and not env.order_book.active_orders
    transition = detached(info)
    assert_row_chain(transition["rows"], [6, 7])
    assert transition["settlement_witness"]["settled"] is True


def test_signed_native_debt_stops_early_and_never_claims_full_reserve():
    # Short proceeds1495.5 after fee5/funding+.5, liability2000 at gap400.
    env = environment(
        actor_steps=1,
        reserve_steps=2,
        fee=0.01,
        side=-1,
        terminal_price=400.0,
        funding=True,
    )
    env.reset()
    observation, reward, terminated, truncated, info = env.step(np.int64(1))
    assert_zero_terminal(observation, terminated, truncated)
    assert env.book.portfolio_value == pytest.approx(-504.5)
    assert reward == pytest.approx(-1.5045), "retain debt; no loss clipping"
    assert env.book.insolvent is True and env.book.termination_reason is not None
    assert (env.actor_step, env.index) == (1, 8), "no fabricated unexecuted bar"
    transition = detached(info)
    assert_row_chain(transition["rows"], [6, 7])
    witness = transition["settlement_witness"]
    assert witness["reserve_steps_processed"] == 1
    assert witness["settled"] is False and witness["eligible"] is False


def test_full_reserve_with_partial_fill_keeps_real_residual_and_active_order():
    env = environment(actor_steps=1, reserve_steps=2, partial=True)
    env.reset()
    observation, reward, terminated, truncated, info = env.step(np.int64(3))
    assert_zero_terminal(observation, terminated, truncated)
    assert (env.actor_step, env.index) == (1, 9)
    assert env.book.exact_quantities[0] == 3
    assert env.order_book.active_orders, "no synthetic cancellation/fill"
    assert env.book.portfolio_value == pytest.approx(998.6)
    assert reward == pytest.approx(-0.0014)
    transition = detached(info)
    assert_row_chain(transition["rows"], [6, 7, 8])
    witness = transition["settlement_witness"]
    assert witness["reserve_steps_processed"] == 2
    assert witness["settled"] is False and witness["eligible"] is False
    # Fit-receipt tests separately prove sticky publication rejection;
    # a later eligible/reset episode must not erase this mechanical failure.


def test_reserve_never_prepares_forecast_or_policy_feature_observation(monkeypatch):
    settled_env = capability("trade_rl.evaluation.rl_allocation.settled_env")
    env = environment(reserve_features=False)
    prepared = []
    original = allocation_decision.prepare_allocation_decision

    def actor_only(*args, **kwargs):
        index = kwargs["start_index"]
        assert index < env.actor_stop_index
        prepared.append(index)
        return original(*args, **kwargs)

    monkeypatch.setattr(allocation_decision, "prepare_allocation_decision", actor_only)
    # Cover the environment alias and the canonical actor rebind.
    monkeypatch.setattr(settled_env, "prepare_allocation_decision", actor_only)
    env.reset()
    env.step(np.int64(3))
    _, _, terminated, _, info = env.step(np.int64(0))
    assert terminated is True
    assert set(prepared) == {6, 7}
    assert_row_chain(detached(info)["rows"], [7, 8, 9])


@pytest.mark.parametrize("native_bars", [2, np.int64(1), True])
def test_final_generator_resume_rejects_bad_clock_retaining_returned_frontier(
    monkeypatch, native_bars
):
    env = environment()
    env.reset()
    env.step(np.int64(3))
    native_execute = reserved_closing._execute_allocation_target

    def invalid_last_clock(*args, **kwargs):
        target, execution = native_execute(*args, **kwargs)
        if kwargs["start_index"] == 9:
            execution = replace(execution, bars_advanced=native_bars)
        return target, execution

    monkeypatch.setattr(
        reserved_closing, "_execute_allocation_target", invalid_last_clock
    )
    with pytest.raises(ValueError, match="one-bar clock") as failure:
        env.step(np.int64(0))
    # The last yielded actual state is adopted before resume rejects its clock.
    assert env.index == 10 and env.book.as_of_index == 10
    assert env.book.portfolio_value == pytest.approx(998.0)
    raw = failure.value.settled_execution_receipt
    assert type(raw) is bytes
    receipt = json.loads(raw)
    assert canonical_json_bytes(receipt) == raw
    assert [row["start_index"] for row in receipt["rows"]] == [7, 8, 9]
    last = receipt["rows"][-1]
    assert last["native"]["execution"]["bars_advanced"] == native_bars
    assert last["returned_clock_types"]["bars_advanced"] == type(native_bars).__name__
    assert last["native"]["book"]["equity"] == pytest.approx(998.0)
    assert last["clock_valid"] is False
    assert receipt["failure_stage"] == "closing_validation"
    assert receipt["attempted_start_index"] == 9
    assert receipt["settlement_witness"]["settled"] is False


def test_settled_return_retains_exact_terminal_fill_order_states():
    env = environment()
    env.reset()
    _, _, _, _, first = env.step(np.int64(3))
    opened = detached(first)["rows"][0]
    orders = opened["accepted_orders"]
    assert {order["intent"]["order_id"] for order in orders} == {
        event["order_id"] for event in opened["native"]["order_events"]
    }
    assert len(orders) == 1 and orders[0]["status"] == "filled"
    assert orders[0]["exact_cumulative_filled_quantity"] == "5"
    _, _, _, _, last = env.step(np.int64(0))
    closed = [
        order for row in detached(last)["rows"] for order in row["accepted_orders"]
    ]
    assert len(closed) == 1 and closed[0]["status"] == "filled"
    assert closed[0]["exact_cumulative_filled_quantity"] == "-5"


@pytest.mark.parametrize("invalid", ["old_bound", "limit", "latency", "horizon"])
def test_constructor_rejects_unsupported_lane_or_clock_before_reset(
    monkeypatch, invalid
):
    changes = {}
    if invalid == "limit":
        changes["cost_changes"] = {"order_type": "limit"}
    if invalid == "latency":
        changes["cost_changes"] = {"order_latency_bars": 1}
    args, marked = arguments(**changes)
    SettledAllocationTradingEnv = capability(
        "trade_rl.evaluation.rl_allocation.settled_env"
    ).SettledAllocationTradingEnv
    if invalid == "old_bound":
        args["bound"] = marked.bound
    if invalid == "horizon":
        args["observation_schema"] = replace(
            args["observation_schema"], episode_steps=3
        )

    def forbidden_reset(*args, **kwargs):
        pytest.fail("constructor admission must precede reset")

    monkeypatch.setattr(SettledAllocationTradingEnv, "reset", forbidden_reset)
    with pytest.raises((TypeError, ValueError)):
        SettledAllocationTradingEnv(**args)


@pytest.mark.parametrize("entrypoint", ["build", "fit"])
def test_legacy_marked_build_fit_explicitly_rejects_settled_before_backend_or_reset(
    monkeypatch, entrypoint
):
    env = environment()
    called = []

    def forbidden(*args, **kwargs):
        called.append((args, kwargs))
        pytest.fail("legacy class admission must precede binding/reset/backend import")

    monkeypatch.setattr(env, "validate_binding", forbidden)
    monkeypatch.setattr(env, "reset", forbidden)
    monkeypatch.setattr(marked_training.importlib, "import_module", forbidden)
    # Avoid accepting AttributeError
    # from dereferencing a different clock as an accidental legacy lane guard.
    with pytest.raises((TypeError, ValueError), match="AllocationTradingEnv|marked"):
        if entrypoint == "build":
            marked_training.build_allocation_ppo(env)
        else:
            marked_training.fit_allocation_ppo(env, total_timesteps=2)
    assert called == []


def test_settled_dd_breach_is_retained_without_claiming_risk_eligibility():
    env = environment(actor_steps=1, reserve_steps=1, terminal_price=60.0)
    env.reset()
    _, reward, terminal, _, info = env.step(np.int64(3))
    assert terminal is True and reward == pytest.approx(-0.2016)
    assert env.book.cash == pytest.approx(798.4)
    witness = detached(info)["settlement_witness"]
    assert witness["settled"] is True and witness["eligible"] is False
    assert witness["maximum_drawdown"] > 0.20
    assert env.publication_invalid is False, "risk-ineligible settlement is retained"


def test_later_cash_episode_cannot_erase_incomplete_closing_publication_failure():
    env = environment(actor_steps=1, reserve_steps=2, partial=True)
    env.reset()
    env.step(np.int64(3))
    assert env.publication_invalid is True
    env.reset()
    _, _, _, _, info = env.step(np.int64(0))
    assert detached(info)["settlement_witness"]["eligible"] is True
    assert env.publication_invalid is True


@pytest.mark.parametrize("replacement", ["executor", "risk", "book", "order_book"])
def test_live_native_instance_replacement_is_rejected_even_with_same_profile(
    replacement,
):
    env = environment()
    env.reset()
    env.step(np.int64(3))
    if replacement == "executor":
        env.executor = MarketExecutor(
            env.dataset, env.execution_cost, insolvency_valuation="retain_debt"
        )
    elif replacement == "risk":
        env.risk = type(env.risk)(env.risk_config)
    else:
        setattr(env, replacement, deepcopy(getattr(env, replacement)))
    with pytest.raises(ValueError, match="instance"):
        env.step(np.int64(0))
    assert (env.index, env.actor_step, env.book.portfolio_value) == (7, 1, 999.0)


def test_live_reset_rejects_account_discard_and_preserves_unfinished_frontier():
    env = environment()
    env.reset()
    env.step(np.int64(3))
    book, orders = env.book, env.order_book
    with pytest.raises(ValueError, match="unfinished"):
        env.reset()
    assert env.book is book and env.order_book is orders
    assert (env.index, env.actor_step, env.book.portfolio_value) == (7, 1, 999.0)
    assert env.book.exact_quantities[0] == 5
    assert env.publication_invalid is True
