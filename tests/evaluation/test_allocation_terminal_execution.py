"""Literal native closing oracles; no fit, market data or research selection."""

from dataclasses import replace
from importlib import import_module, util

import numpy as np
import pytest

from tests.evaluation.test_allocation_global_execution import declare
from tests.evaluation.test_allocation_nonrl_walk_forward import fixture
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2


def capability():
    name = "trade_rl.evaluation.allocation_terminal_execution"
    assert util.find_spec(name) is not None, "native terminal closing owner is missing"
    return import_module(name)


def carrier(
    *, side=1, kind="nonrl", partial=False, cash_rate=0, borrow_rate=0, risk=None
):
    folds, (original,) = fixture(
        ranges=((6, 7),),
        signals=(side, 0, 0, 0),
        fee=0.01,
        mode="direct" if kind == "cash" else "residual",
        risk=risk,
    )
    prices = np.full((12, 1), 100.0)
    prices[8:] = 110 if side == 1 else 90
    volume = original.dataset.volume.copy()
    if partial:
        volume[8] = 2
    dataset = replace(
        original.dataset,
        open=prices,
        close=prices,
        high=prices,
        low=prices,
        mark_price=prices,
        funding_price_rate=prices * 0.001,
        funding_due=np.ones((12, 1), dtype=bool),
        volume=volume,
        cash_rate=np.full(12, cash_rate),
        borrow_rate=np.full((12, 1), borrow_rate),
    )
    allocator = (
        replace(original.allocator, lower_weight=-0.5, upper_weight=0.0)
        if side == -1
        else original.allocator
    )
    from trade_rl.simulation import MarketExecutor
    from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile

    cost = (
        replace(original.execution_cost, borrow_rate_multiplier=1.0)
        if borrow_rate
        else original.execution_cost
    )
    economics = MarketExecutor(
        dataset, cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest

    profile = AllocationRuntimeProfile(
        economics,
        content_digest(original.risk_config),
        1000.0,
        "USD",
        3600,
        3600,
    )
    recipe = allocation_recipe_payload_v2(
        original.action_contract,
        ("signal",),
        allocator=allocator,
        expected_horizon_seconds=3600,
        observation_schema=original.observation_schema,
        runtime_profile=profile,
    )
    bound = replace(
        original.bound,
        objective=replace(
            original.bound.objective,
            deployment_recipe_digest=content_digest(recipe),
            economics_digest=economics,
        ),
    )
    env = AllocationTradingEnv(
        dataset,
        stream=original.stream,
        estimates=tuple(original._estimates.values()),
        bound=bound,
        action_contract=original.action_contract,
        allocator=allocator,
        execution_cost=cost,
        risk_config=original.risk_config,
        feature_indices=(0,),
        symbol_index=0,
        start_index=6,
        stop_index=7,
        account_id=original.account_id,
        observation_schema=original.observation_schema,
    )
    return folds, env


def plans(api, env, folds, *, kind="nonrl", stop=8):
    native = import_module("trade_rl.evaluation.allocation_global_execution")
    prefix = declare(native, env, folds, kind=kind)
    return api.declare_terminal_allocation_execution(
        folds,
        env,
        prefix,
        settlement_stop_index=stop,
    )


@pytest.mark.parametrize(
    "side,final_cash,funding", [(1, 1039.0, -0.5), (-1, 1041.0, 0.5)]
)
def test_literal_round_trip_charges_real_close_and_telescopes(
    side, final_cash, funding, monkeypatch
):
    api = capability()
    folds, env = carrier(side=side)
    plan = plans(api, env, folds)
    old_bound, old_recipe = env.bound.digest, env.recipe_digest
    resets = []
    native_reset = env.reset

    def reset(**kwargs):
        resets.append(True)
        return native_reset(**kwargs)

    monkeypatch.setattr(env, "reset", reset)
    result = api.run_terminal_allocation_execution(folds, env, plan)
    raw = result.payload
    assert resets == [True]
    assert env.bound.digest == old_bound and env.recipe_digest == old_recipe
    assert env.index == env.stop_index == 7 and env._terminated
    assert raw["native_receipt"] == result.native_execution.receipt.payload
    assert raw["native_receipt"]["plan"] == plan.native_plan.payload
    assert (
        raw["closing_opening_state_digest"]
        == raw["native_receipt"]["closing_state_digest"]
    )
    assert plan.bound.objective.terminal_valuation == "settled"
    assert plan.bound.clock.economic_horizon_seconds == 7200
    assert plan.bound.objective.deployment_recipe_digest != env.recipe_digest
    assert raw["final_book"]["cash"] == pytest.approx(final_cash)
    assert raw["final_book"]["equity"] == pytest.approx(final_cash)
    assert raw["final_book"]["funding_pnl"] == pytest.approx(funding)
    assert raw["final_book"]["exact_quantities"] == ["0"]
    assert raw["final_active_orders"] == []
    assert raw["settlement_complete"] and raw["risk_eligible"]
    assert raw["settled_profit_rate"] == pytest.approx((final_cash - 1000) / 1000)
    assert sum(raw["fixed_capital_rewards"]) == pytest.approx(
        raw["settled_profit_rate"]
    )
    assert raw["closing_rows"][0]["native"]["execution"][
        "interval_cost"
    ] == pytest.approx(5.5 if side == 1 else 4.5)
    frozen = canonical_json_bytes(raw)
    raw["final_book"]["cash"] = 0
    assert canonical_json_bytes(result.payload) == frozen


def test_cash_really_processes_all_reserved_bars_and_interest():
    api = capability()
    folds, env = carrier(kind="cash", cash_rate=0.876)
    result = api.run_terminal_allocation_execution(
        folds, env, plans(api, env, folds, kind="cash", stop=9)
    )
    raw = result.payload
    assert len(raw["closing_rows"]) == 2
    assert raw["final_book"]["cash"] == pytest.approx(1000 * 1.0001**3)
    assert raw["final_book"]["fill_count"] == 0
    assert raw["settlement_complete"]


def test_short_borrow_is_charged_on_held_entry_and_not_flat_closing():
    api = capability()
    folds, env = carrier(side=-1, borrow_rate=8.76)
    raw = api.run_terminal_allocation_execution(
        folds, env, plans(api, env, folds)
    ).payload
    assert raw["final_book"]["cash"] == pytest.approx(1040.5)
    assert raw["final_book"]["borrow_cost"] == pytest.approx(0.5)
    assert raw["closing_rows"][0]["native"]["execution"]["interval_borrow_cost"] == 0


@pytest.mark.parametrize(
    "side,cash,equity,quantity",
    [(1, 711.97, 1041.97, "3"), (-1, 1313.97, 1043.97, "-3")],
)
def test_profitable_partial_close_keeps_actual_inventory_and_cannot_be_settled(
    side, cash, equity, quantity
):
    api = capability()
    folds, env = carrier(side=side, partial=True)
    raw = api.run_terminal_allocation_execution(
        folds, env, plans(api, env, folds)
    ).payload
    assert raw["status"] == "completed"
    assert raw["final_book"]["cash"] == pytest.approx(cash)
    assert raw["final_book"]["equity"] == pytest.approx(equity)
    assert raw["final_book"]["exact_quantities"] == [quantity]
    assert raw["final_active_orders"]
    assert not raw["settlement_complete"] and not raw["risk_eligible"]
    assert raw["settled_profit_rate"] is None
    assert raw["observed_profit_rate"] > 0


@pytest.mark.parametrize("mutation", ["volume", "mark", "clock", "cost", "bound"])
def test_changed_frozen_execution_source_rejects_before_first_reset(
    mutation, monkeypatch
):
    api = capability()
    folds, env = carrier()
    plan = plans(api, env, folds)
    if mutation in {"volume", "mark"}:
        name = "volume" if mutation == "volume" else "mark_price"
        array = env.dataset.resolved_array(name).copy()
        array[8] *= 2
        object.__setattr__(env.dataset, name, array)
    elif mutation == "clock":
        array = env.dataset.timestamps.copy()
        array[8] += np.timedelta64(1, "s")
        object.__setattr__(env.dataset, "timestamps", array)
    elif mutation == "cost":
        env.execution_cost = replace(env.execution_cost, fee_rate=0.02)
    else:
        env.bound = replace(
            env.bound, objective=replace(env.bound.objective, maximum_drawdown=0.19)
        )
    monkeypatch.setattr(env, "reset", lambda **_: pytest.fail("reset before refusal"))
    with pytest.raises(ValueError):
        api.run_terminal_allocation_execution(folds, env, plan)
    assert not hasattr(env, "book")


def test_closing_does_not_consume_forecasts_or_features(monkeypatch):
    api = capability()
    folds, env = carrier()
    plan = plans(api, env, folds)
    values = env.dataset.features.copy()
    values[8:] = 1e6
    object.__setattr__(env.dataset, "features", values)
    original = api._execute_allocation_target

    def close(*args, **kwargs):
        monkeypatch.setattr(
            env, "_observation", lambda: pytest.fail("closing observation")
        )
        monkeypatch.setattr(env, "_prepare", lambda: pytest.fail("closing forecast"))
        monkeypatch.setattr(env, "step", lambda *_: pytest.fail("closing policy step"))
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "_execute_allocation_target", close)
    raw = api.run_terminal_allocation_execution(folds, env, plan).payload
    assert raw["final_book"]["cash"] == pytest.approx(1039)


def test_close_exception_preserves_returned_bar_and_original_error(monkeypatch):
    api = capability()
    folds, env = carrier(partial=True)
    plan = plans(api, env, folds, stop=9)
    original = api._execute_allocation_target

    def close(*args, **kwargs):
        if kwargs["start_index"] == 8:
            raise RuntimeError("closing failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "_execute_allocation_target", close)
    with pytest.raises(RuntimeError, match="closing failure") as caught:
        api.run_terminal_allocation_execution(folds, env, plan)
    raw = caught.value.terminal_execution_receipt
    assert raw["status"] == "integrity_failure"
    assert len(raw["closing_rows"]) == 1
    assert raw["attempted_start_index"] == 8
    assert raw["final_book"]["exact_quantities"] == ["3"]
    assert raw["settled_profit_rate"] is None


@pytest.mark.parametrize("stop", [True, 7, 12])
def test_reserve_requires_a_nonempty_inside_dataset_window(stop):
    api = capability()
    folds, env = carrier()
    with pytest.raises(ValueError):
        plans(api, env, folds, stop=stop)


def test_existing_prefix_trace_bytes_remain_historical():
    from hashlib import sha256

    native = import_module("trade_rl.evaluation.allocation_global_execution")
    folds, env = carrier()
    result = native.run_declared_global_allocation_execution(
        folds, env, declare(native, env, folds)
    )
    assert (
        sha256(
            canonical_json_bytes(result.receipt.payload["rows"][0]["native"])
        ).hexdigest()
        == "91aeaa84378b12d145f4b99f2d9f065fd88c946c7d4b0d1bff831e876143e590"
    )


@pytest.mark.parametrize("case", ["no_volume", "untradable", "dust"])
def test_failed_close_cannot_erase_residual_or_call_it_cash(case):
    from fractions import Fraction

    api = capability()
    folds, env = carrier()
    name = "tradable" if case == "untradable" else "volume"
    values = env.dataset.resolved_array(name).copy()
    values[8] = (
        False if case == "untradable" else 0 if case == "no_volume" else 4.999999999999
    )
    object.__setattr__(env.dataset, name, values)
    raw = api.run_terminal_allocation_execution(
        folds, env, plans(api, env, folds)
    ).payload
    quantity = Fraction(raw["final_book"]["exact_quantities"][0])
    assert quantity > 0
    if case == "dust":
        assert quantity < Fraction(1, 10**9)
    else:
        assert quantity == 5
    assert not raw["settlement_complete"] and raw["settled_profit_rate"] is None
    assert raw["final_book"]["fill_count"] >= 1


def test_pending_entry_is_cancelled_before_real_close_without_reopening():
    api = capability()
    folds, env = carrier()
    volume = env.dataset.volume.copy()
    volume[7] = 2
    object.__setattr__(env.dataset, "volume", volume)
    raw = api.run_terminal_allocation_execution(
        folds, env, plans(api, env, folds)
    ).payload
    native = raw["native_receipt"]["rows"][0]["native"]
    opening_order = native["active_orders"][0]["intent"]["order_id"]
    events = raw["closing_rows"][0]["native"]["order_events"]
    assert any(
        event["order_id"] == opening_order and event["event_type"] == "cancelled"
        for event in events
    )
    assert raw["final_book"]["cash"] == pytest.approx(1015.6)
    assert raw["final_book"]["fill_count"] == 2
    assert raw["settlement_complete"]


@pytest.mark.parametrize("side,price,expected", [(1, 50, 742), (-1, 400, -504.5)])
def test_adverse_close_preserves_dd_or_signed_debt_and_never_qualifies(
    side, price, expected
):
    api = capability()
    folds, env = carrier(side=side)
    for name in ("open", "high", "low", "close", "mark_price"):
        values = env.dataset.resolved_array(name).copy()
        values[8:] = price
        object.__setattr__(env.dataset, name, values)
    raw = api.run_terminal_allocation_execution(
        folds, env, plans(api, env, folds)
    ).payload
    assert raw["final_book"]["equity"] == pytest.approx(expected)
    assert raw["final_book"]["max_drawdown"] > 0.20
    assert not raw["risk_eligible"]
    assert raw["observed_profit_rate"] == pytest.approx((expected - 1000) / 1000)
    assert raw["fixed_capital_reward_sum"] == pytest.approx(raw["observed_profit_rate"])
    if side == -1:
        assert raw["status"] == "economic_stop"
        assert raw["settled_profit_rate"] is None
        assert raw["observed_profit_rate"] < -1
    else:
        assert raw["settlement_complete"]
        assert raw["settled_profit_rate"] < 0


def test_prefix_economic_stop_keeps_its_partial_receipt_and_never_closes(monkeypatch):
    from tests.evaluation.test_allocation_nonrl_walk_forward import (
        global_control_fixture,
        global_control_folds,
    )

    api = capability()
    folds = global_control_folds(7)
    env = global_control_fixture(signals=(-1, -1, -1, -1), insolvent_short=True)
    plan = plans(api, env, folds, stop=11)
    monkeypatch.setattr(
        api,
        "_execute_allocation_target",
        lambda *_a, **_k: pytest.fail("close after prefix stop"),
    )
    with pytest.raises(ValueError) as caught:
        api.run_terminal_allocation_execution(folds, env, plan)
    raw = caught.value.terminal_execution_receipt
    assert raw["native_receipt"] == caught.value.global_execution_receipt.payload
    assert raw["status"] == "economic_stop" and raw["closing_rows"] == []
    assert raw["final_book"]["equity"] == pytest.approx(-501)
    assert not raw["settlement_complete"]


def test_unequal_segments_preserve_one_account_and_execution_rng(monkeypatch):
    from tests.evaluation.test_allocation_nonrl_walk_forward import (
        global_control_fixture,
        global_control_folds,
    )

    api = capability()
    results, resets = [], []
    for boundary in (7, 9, None):
        env = global_control_fixture(
            signals=(1, 1, 1, 1),
            cost_changes={"slippage_std": 0.001, "random_seed": 42},
        )
        folds = global_control_folds(boundary or 7)
        if boundary is None:
            folds = (replace(folds[0], test=replace(folds[0].test, stop=10)),)
        plan = plans(api, env, folds, stop=11)
        original_reset = env.reset

        def reset(_original=original_reset, **kwargs):
            resets.append(True)
            return _original(**kwargs)

        monkeypatch.setattr(env, "reset", reset)
        execute = api._execute_allocation_target

        def close(executor, book, orders, **kwargs):
            assert executor is env.executor and kwargs["pretrade_risk"] is env.risk
            assert book is env.book and orders is env.order_book
            return execute(executor, book, orders, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(api, "_execute_allocation_target", close)
            raw = api.run_terminal_allocation_execution(folds, env, plan).payload
        results.append(raw)
    assert resets == [True, True, True]
    assert (
        results[0]["final_book"] == results[1]["final_book"] == results[2]["final_book"]
    )
    assert all(raw["settlement_complete"] for raw in results)


def test_structurally_valid_ppo_plan_is_rejected_without_policy_loading(monkeypatch):
    api = capability()
    native = import_module("trade_rl.evaluation.allocation_global_execution")
    folds, env = carrier()
    raw = declare(native, env, folds).payload
    cell = raw["cell"]
    cell.update(kind="residual_ppo", seed=0, reset_seed=0, control_policy_digest=None)
    cell["artifacts"] = [
        {
            "fold_index": 0,
            "policy_digest": "1" * 64,
            "recipe_digest": cell["original_recipe_digest"],
            "policy_sha256": "2" * 64,
        }
    ]
    changed = native.GlobalAllocationExecutionPlan.from_payload(raw)
    monkeypatch.setattr(env, "reset", lambda **_: pytest.fail("PPO reset"))
    with pytest.raises(ValueError, match="NONRL/cash"):
        api.declare_terminal_allocation_execution(
            folds, env, changed, settlement_stop_index=8
        )


@pytest.mark.parametrize("first_volume,final_cash", [(0, 1038.45), (2, 1038.67)])
def test_deferred_close_reuses_actual_pending_order_book_and_held_carry(
    first_volume, final_cash, monkeypatch
):
    api = capability()
    folds, env = carrier()
    volume = env.dataset.volume.copy()
    volume[8] = first_volume
    object.__setattr__(env.dataset, "volume", volume)
    plan = plans(api, env, folds, stop=9)
    native = api._execute_allocation_target
    previous = []

    def close(executor, book, orders, **kwargs):
        assert executor is env.executor and kwargs["pretrade_risk"] is env.risk
        if previous:
            assert book is previous[-1].book and orders is previous[-1].order_book
        else:
            assert book is env.book and orders is env.order_book
        target, execution = native(executor, book, orders, **kwargs)
        previous.append(execution)
        return target, execution

    monkeypatch.setattr(api, "_execute_allocation_target", close)
    raw = api.run_terminal_allocation_execution(folds, env, plan).payload
    first, last = (row["native"] for row in raw["closing_rows"])
    pending = first["active_orders"][0]["intent"]["order_id"]
    assert any(
        event["order_id"] == pending and event["event_type"] == "filled"
        for event in last["order_events"]
    )
    assert not any(event["event_type"] == "submitted" for event in last["order_events"])
    assert raw["final_book"]["cash"] == pytest.approx(final_cash)
    assert raw["settlement_complete"] and raw["full_close_coverage"]


def test_hard_turnover_constraint_keeps_nonzero_target_unsettled():
    from fractions import Fraction

    from trade_rl.risk import PreTradeRiskConfig

    api = capability()
    folds, env = carrier(
        risk=PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0, max_turnover=0.25)
    )
    raw = api.run_terminal_allocation_execution(
        folds, env, plans(api, env, folds)
    ).payload
    assert "max_turnover" in raw["closing_rows"][0]["risk_reasons"]
    assert Fraction(raw["final_book"]["exact_quantities"][0]) > 0
    assert not raw["settlement_complete"]


def test_malformed_returned_clock_retains_detached_native_facts(monkeypatch):
    api = capability()
    folds, env = carrier()
    plan = plans(api, env, folds)
    native = api._execute_allocation_target

    def close(*args, **kwargs):
        target, execution = native(*args, **kwargs)
        return target, replace(execution, next_index=99)

    monkeypatch.setattr(api, "_execute_allocation_target", close)
    with pytest.raises(ValueError, match="one-bar clock") as caught:
        api.run_terminal_allocation_execution(folds, env, plan)
    raw = caught.value.terminal_execution_receipt
    assert raw["closing_rows"][0]["processing_time_ns"] is None
    assert raw["closing_rows"][0]["native"]["book"]["cash"] == pytest.approx(1039)
    assert not raw["settlement_complete"]
