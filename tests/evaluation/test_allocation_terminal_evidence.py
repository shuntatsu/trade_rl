"""Closed synthetic terminal evidence: literal endpoints and adversarial records."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from importlib import import_module, util

import pytest

from tests.evaluation.test_allocation_terminal_execution import carrier, plans
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation import allocation_reserved_closing as reserved_closing


def capability():
    name = "trade_rl.evaluation.allocation_terminal_evidence"
    assert util.find_spec(name) is not None, "closed terminal evidence reader missing"
    return import_module(name)


def observed(**kwargs):
    api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(**kwargs)
    plan = plans(api, env, folds, kind=kwargs.get("kind", "nonrl"))
    result = api.run_terminal_allocation_execution(folds, env, plan)
    return plan, result.payload


@pytest.mark.parametrize("side,wealth", [(1, 1039.0), (-1, 1041.0)])
def test_closed_round_trip_literal_and_detached_bytes(side, wealth):
    api = capability()
    plan, raw = observed(side=side)
    receipt = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    )
    assert receipt.payload["schema"] == "allocation_terminal_runtime_observation_v2"
    assert receipt.payload["settled_profit_rate"] == pytest.approx(
        (wealth - 1000) / 1000
    )
    assert receipt.payload["final_book"]["cash"] == pytest.approx(wealth)
    assert receipt.payload["fixed_capital_reward_sum"] == pytest.approx(
        (wealth - 1000) / 1000
    )
    frozen = canonical_json_bytes(raw)
    raw["closing_rows"][0]["native"]["book"]["cash"] = 1
    assert canonical_json_bytes(receipt.payload) == frozen
    assert receipt.digest == content_digest(receipt.payload)


@pytest.mark.parametrize(
    "side,cash,equity,q", [(1, 711.97, 1041.97, "3"), (-1, 1313.97, 1043.97, "-3")]
)
def test_partial_close_is_valid_observation_without_settled_profit(
    side, cash, equity, q
):
    api = capability()
    plan, raw = observed(side=side, partial=True)
    result = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    ).payload
    assert result["final_book"]["cash"] == pytest.approx(cash)
    assert result["final_book"]["equity"] == pytest.approx(equity)
    assert result["final_book"]["exact_quantities"] == [q]
    assert result["final_active_orders"]
    assert not result["settlement_complete"] and not result["risk_eligible"]
    assert result["settled_profit_rate"] is None


@pytest.mark.parametrize(
    "stage,partial,notional",
    [("prefix", False, 500.0), ("closing", False, 550.0), ("closing", True, 220.0)],
)
def test_retained_fill_event_notional_must_match_execution(stage, partial, notional):
    api = capability()
    plan, raw = observed(partial=partial)
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = (
        raw["native_receipt"]["rows"][0]["native"]
        if stage == "prefix"
        else raw["closing_rows"][0]["native"]
    )
    fill = next(
        event
        for event in native["order_events"]
        if event["event_type"] in ("filled", "partial_fill")
    )
    assert fill["filled_notional"] == pytest.approx(notional)
    assert native["execution"]["filled_notional"] == pytest.approx(notional)
    fill["filled_notional"] = 1.0
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_fill_event_symbol_totals_cannot_move_to_an_untraded_coordinate():
    api = capability()
    plan, raw = observed(n_symbols=2)
    receipt = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    )
    assert receipt.payload["settled_profit_rate"] == pytest.approx(0.039)
    native = raw["closing_rows"][0]["native"]
    fills = [
        event for event in native["order_events"] if event["event_type"] == "filled"
    ]
    assert [event["symbol_index"] for event in fills] == [0]
    assert native["execution"]["filled_notional"] == pytest.approx(550.0)
    assert native["execution"]["filled_notional_by_symbol"] == [550.0, 0.0]
    native["execution"]["filled_notional_by_symbol"] = [0.0, 550.0]
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_fill_cost_cannot_move_to_an_untraded_coordinate():
    api = capability()
    plan, raw = observed(n_symbols=2)
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    execution = raw["closing_rows"][0]["native"]["execution"]
    assert execution["cost_by_symbol"] == [5.5, 0.0]
    execution["cost_by_symbol"] = [0.0, 5.5]
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize("mutation", ["cash", "ulp_cash", "inventory"])
def test_flat_cash_cannot_gain_wealth_without_a_returned_cash_flow(mutation):
    import math

    from trade_rl.evaluation.rl_allocation.transition_facts import (
        terminal_allocation_state_digest,
    )

    api = capability()
    plan, raw = observed(kind="cash")
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    row = raw["closing_rows"][0]
    native = row["native"]
    execution, book = native["execution"], native["book"]
    assert not native["order_events"] and book["cash"] == 1000.0
    for field in (
        "interval_cost",
        "interval_funding",
        "interval_borrow_cost",
        "interval_dividend",
        "interval_cash_interest",
    ):
        assert execution[field] == 0.0
    wealth = 2000.0 if mutation == "cash" else 1110.0
    if mutation == "ulp_cash":
        wealth = math.nextafter(1000.0, math.inf)
    if mutation != "inventory":
        book["cash"] = wealth
    else:
        book.update(quantities=[1.0], exact_quantities=["1"])
    book.update(equity=wealth, peak_value=wealth)
    rate = (wealth - 1000.0) / 1000.0
    execution.update(
        interval_net_return=rate,
        interval_log_return=math.log1p(rate),
        interval_gross_return=rate,
    )
    for boundary in execution["funding_evidence"]:
        boundary.update(equity_before_funding=wealth, equity_after_funding=wealth)
        if mutation == "inventory":
            boundary.update(signed_quantities=[1.0], funding_price_rates=[0.0])
    row["after_state_digest"] = terminal_allocation_state_digest(
        plan.native_plan.payload["common"],
        plan.closing_source_digest,
        native,
        index=row["processing_index"],
        time_ns=row["processing_time_ns"],
    )
    row["fixed_capital_reward"] = rate
    raw.update(
        final_book=book,
        settlement_complete=mutation != "inventory",
        risk_eligible=mutation != "inventory",
        observed_profit_rate=rate,
        settled_profit_rate=rate if mutation != "inventory" else None,
        fixed_capital_rewards=[0.0, rate],
        fixed_capital_reward_sum=rate,
    )
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_carried_active_order_cannot_be_replaced_by_a_fresh_event_id():
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(partial=True)
    plan = plans(runner, env, folds, stop=9)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    previous, native = [row["native"] for row in raw["closing_rows"]]
    order_id = previous["active_orders"][0]["intent"]["order_id"]
    assert not native["active_orders"]
    replacement = "b" * 64 if order_id != "b" * 64 else "a" * 64
    for event in native["order_events"]:
        if event["order_id"] == order_id:
            event["order_id"] = replacement
    for reason in native["terminal_order_reasons"]:
        if reason["order_id"] == order_id:
            reason["order_id"] = replacement
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize("mutation", ["initial_only", "initial_and_remaining"])
def test_retained_capacity_conserves_its_consumed_and_remaining_notional(mutation):
    api = capability()
    plan, raw = observed()
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    capacity = raw["closing_rows"][0]["native"]["execution"]["capacity_evidence"][0]
    assert capacity["initial_capacity_notional"] == pytest.approx(
        capacity["consumed_capacity_notional"] + capacity["remaining_capacity_notional"]
    )
    capacity["initial_capacity_notional"] += 1.0
    if mutation == "initial_and_remaining":
        capacity["remaining_capacity_notional"] += 1.0
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def repin_last_close(plan, raw):
    """Keep forged redundant summaries coherent to exercise native fact links."""
    import math

    from trade_rl.evaluation.rl_allocation.transition_facts import (
        terminal_allocation_state_digest,
    )

    row = raw["closing_rows"][-1]
    native = row["native"]
    book, ex = native["book"], native["execution"]
    opening = row["opening_equity"]
    net = max(max(book["equity"], 0.0) / opening - 1, -1 + 1e-12)
    ex["interval_net_return"], ex["interval_log_return"] = net, math.log1p(net)
    ex["interval_gross_return"] = (
        max(book["equity"], 0.0)
        + ex["interval_cost"]
        - ex["interval_funding"]
        + ex["interval_borrow_cost"]
        - ex["interval_dividend"]
        - ex["interval_cash_interest"]
    ) / opening - 1
    for boundary in ex["funding_evidence"]:
        boundary.update(
            equity_before_funding=book["equity"] - boundary["funding_amount"],
            equity_after_funding=book["equity"],
            signed_quantities=book["quantities"],
            mark_prices=book["mark_prices"],
            contract_multipliers=book["contract_multipliers"],
        )
    row["after_state_digest"] = terminal_allocation_state_digest(
        plan.native_plan.payload["common"],
        plan.closing_source_digest,
        native,
        index=row["processing_index"],
        time_ns=row["processing_time_ns"],
    )
    row["fixed_capital_reward"] = (book["equity"] - opening) / 1000.0
    raw["fixed_capital_rewards"][-1] = row["fixed_capital_reward"]
    raw["fixed_capital_reward_sum"] = sum(raw["fixed_capital_rewards"])
    raw["final_book"] = book
    raw["observed_profit_rate"] = (book["equity"] - 1000.0) / 1000.0
    if raw["settlement_complete"]:
        raw["settled_profit_rate"] = raw["observed_profit_rate"]


@pytest.mark.parametrize("flow", ["dividend", "borrow_cost"])
def test_flat_cash_cannot_use_inventory_cash_flows(flow):
    api = capability()
    plan, raw = observed(kind="cash")
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = raw["closing_rows"][0]["native"]
    native["execution"]["interval_" + flow] = 1.0
    if flow == "dividend":
        native["book"].update(cash=1001.0, equity=1001.0, peak_value=1001.0)
    else:
        native["book"].update(
            cash=999.0, equity=999.0, borrow_cost=1.0, max_drawdown=0.001
        )
    repin_last_close(plan, raw)
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize(
    "field",
    ["requested_quantity", "reduce_only", "replaced_order_id", "previous_status"],
)
def test_carried_order_event_preserves_known_intent_and_status(field):
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(partial=True)
    plan = plans(runner, env, folds, stop=9)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    event = raw["closing_rows"][1]["native"]["order_events"][0]
    assert event["event_type"] == "filled"
    event[field] = {
        "requested_quantity": event["requested_quantity"] * 10,
        "reduce_only": not event["reduce_only"],
        "replaced_order_id": "a" * 64,
        "previous_status": "accepted",
    }[field]
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_carried_no_fill_cannot_change_cumulative_notional():
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(partial=True)
    volume = env.dataset.volume.copy()
    volume[9] = 0.0
    object.__setattr__(env.dataset, "volume", volume)
    plan = plans(runner, env, folds, stop=9)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = raw["closing_rows"][1]["native"]
    assert [e["event_type"] for e in native["order_events"]] == ["no_fill"]
    native["active_orders"][0]["cumulative_filled_notional"] += 1.0
    repin_last_close(plan, raw)
    raw["final_active_orders"] = native["active_orders"]
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize("field", ["cumulative_filled_notional", "evidence_version"])
def test_carried_partial_fill_cannot_invent_progress(field):
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(partial=True)
    volume = env.dataset.volume.copy()
    volume[9] = 2.0
    object.__setattr__(env.dataset, "volume", volume)
    plan = plans(runner, env, folds, stop=9)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    previous = raw["closing_rows"][0]["native"]["active_orders"][0]
    native = raw["closing_rows"][1]["native"]
    assert [e["event_type"] for e in native["order_events"]] == ["partial_fill"]
    active = native["active_orders"][0]
    assert previous["cumulative_filled_notional"] == 220.0
    assert active["cumulative_filled_notional"] == 440.0
    active[field] += 1
    repin_last_close(plan, raw)
    raw["final_active_orders"] = native["active_orders"]
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_new_order_requires_its_native_submission_event():
    api = capability()
    plan, raw = observed()
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = raw["native_receipt"]["rows"][0]["native"]
    assert native["order_events"][0]["event_type"] == "submitted"
    native["order_events"].pop(0)
    for sequence, event in enumerate(native["order_events"]):
        event["sequence"] = sequence
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_live_margin_deficit_cannot_be_invented():
    api = capability()
    plan, raw = observed()
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = raw["closing_rows"][0]["native"]
    assert native["book"]["margin_deficit"] == 0.0
    native["book"]["margin_deficit"] = 1.0
    repin_last_close(plan, raw)
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize("mutation", ["duplicate", "multiplier", "rate"])
def test_capacity_pool_cannot_be_disconnected_from_selected_fills(mutation):
    api = capability()
    plan, raw = observed(n_symbols=2)
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = raw["closing_rows"][0]["native"]
    ex = native["execution"]
    capacity = ex["capacity_evidence"][0]
    if mutation == "duplicate":
        extra = deepcopy(capacity)
        extra.update(
            consumed_capacity_notional=0.0,
            remaining_capacity_notional=extra["initial_capacity_notional"],
        )
        ex["capacity_evidence"].append(extra)
    elif mutation == "multiplier":
        capacity["contract_multiplier"] = 2.0
    else:
        next(e for e in native["order_events"] if e["event_type"] == "filled")[
            "participation_rate"
        ] = 0.0
        ex.update(max_participation=0.0, participation_by_symbol=[0.0, 0.0])
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_fill_capacity_sequence_cannot_be_shifted():
    api = capability()
    plan, raw = observed()
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    event = next(
        e
        for e in raw["closing_rows"][0]["native"]["order_events"]
        if e["event_type"] == "filled"
    )
    event["capacity_before"] += 1.0
    event["capacity_after"] += 1.0
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_new_active_order_requires_a_retained_submission():
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier()
    volume = env.dataset.volume.copy()
    volume[8] = 0.0
    object.__setattr__(env.dataset, "volume", volume)
    plan = plans(runner, env, folds)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = raw["closing_rows"][0]["native"]
    assert native["active_orders"][0]["cumulative_filled_notional"] == 0.0
    native["order_events"] = []
    native["active_orders"][0].update(
        status="submitted",
        evidence_version=0,
        last_processed_index=None,
    )
    repin_last_close(plan, raw)
    raw["final_active_orders"] = native["active_orders"]
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize("field", ["requested_quantity", "reduce_only"])
def test_new_active_order_events_match_its_returned_intent(field):
    api = capability()
    plan, raw = observed(partial=True)
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = raw["closing_rows"][0]["native"]
    assert native["active_orders"][0]["intent"]["reduce_only"] is True
    for event in native["order_events"]:
        if field == "reduce_only":
            event.pop("reduce_only")
        else:
            event[field] *= 10.0
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize(
    "mutation", ["policy_limit", "consumed", "participation", "symbol_participation"]
)
def test_capacity_and_participation_join_native_policy_and_fills(mutation):
    api = capability()
    plan, raw = observed(max_participation=0.5)
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    ex = raw["closing_rows"][0]["native"]["execution"]
    capacity = ex["capacity_evidence"][0]
    if mutation == "policy_limit":
        capacity["participation_limit"] = 0.75
        capacity["initial_capacity_notional"] = capacity["market_notional"] * 0.75
        capacity["remaining_capacity_notional"] = (
            capacity["initial_capacity_notional"]
            - capacity["consumed_capacity_notional"]
        )
    elif mutation == "consumed":
        capacity["consumed_capacity_notional"] = 0.0
        capacity["remaining_capacity_notional"] = capacity["initial_capacity_notional"]
    elif mutation == "participation":
        ex["max_participation"] = 2.0
    else:
        ex["participation_by_symbol"][0] = 2.0
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize(
    "mutation", ["multiplier", "margin", "rebalance", "foreign_quantity"]
)
def test_known_returned_book_invariants_cannot_be_rewritten(mutation):
    api = capability()
    plan, raw = observed(n_symbols=2, partial=mutation == "foreign_quantity")
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = raw["closing_rows"][0]["native"]
    book, ex = native["book"], native["execution"]
    if mutation == "multiplier":
        book["contract_multipliers"][0] = 2.0
        next(e for e in native["order_events"] if e["event_type"] == "filled")[
            "execution_price"
        ] /= 2.0
    elif mutation == "margin":
        book["margin_used"] = 1.0
    elif mutation == "rebalance":
        ex["rebalance_events"] = 0
        book["rebalance_events"] -= 1
    else:
        book["quantities"][1], book["exact_quantities"][1] = 1.0, "1"
        book["equity"] += 110.0
        book["peak_value"] = book["equity"]
        ex["funding_evidence"][0]["funding_price_rates"][1] = 0.0
    repin_last_close(plan, raw)
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize("mutation", ["diagnostic", "closing_stage"])
def test_original_prefix_failure_cannot_be_replaced(monkeypatch, mutation):
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(kind="cash")
    plan = plans(runner, env, folds, kind="cash")

    def fail(_action):
        raise RuntimeError("original prefix failure")

    monkeypatch.setattr(env, "step", fail)
    with pytest.raises(RuntimeError, match="original prefix failure") as caught:
        runner.run_terminal_allocation_execution(folds, env, plan)
    raw = caught.value.terminal_execution_receipt
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    assert (
        raw["failure_stage"] == "prefix"
        and raw["native_receipt"]["failure"] == raw["failure"]
    )
    raw["failure"]["message"] = "invented failure"
    if mutation == "closing_stage":
        raw["failure_stage"] = "closing_execution"
        raw["attempted_start_index"] = plan.native_plan.payload["common"]["range"][1]
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize("mutation", ["remaining", "overfill"])
def test_filled_event_preserves_native_quantity_completion(mutation):
    api = capability()
    plan, raw = observed()
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    event = next(
        e
        for e in raw["closing_rows"][0]["native"]["order_events"]
        if e["event_type"] == "filled"
    )
    if mutation == "remaining":
        event["remaining_quantity"] = -1.0
    else:
        event["filled_quantity"] *= 2.0
        event["execution_price"] /= 2.0
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_cumulative_metrics_accept_native_large_operand_rounding():
    api = capability()
    plan, raw = observed(partial=True)
    previous = raw["native_receipt"]["rows"][0]["native"]["book"]
    native = raw["closing_rows"][0]["native"]
    previous["funding_pnl"] = 1_000_000.0
    amount = native["execution"]["interval_funding"]
    assert amount == pytest.approx(-0.33)
    native["book"]["funding_pnl"] = previous["funding_pnl"] + amount
    assert abs(native["book"]["funding_pnl"] - previous["funding_pnl"] - amount) > 1e-12
    api._metrics(
        previous,
        native,
        previous_orders=raw["native_receipt"]["rows"][0]["native"]["active_orders"],
        cost=plan.native_plan.payload["cell"]["runtime"]["cost_payload"],
        symbol_index=0,
    )


def test_flat_cash_closure_preserves_native_sequential_interest_rounding():
    import numpy as np

    from trade_rl.simulation.accounting import BookState

    api = capability()
    capital = float(2**53)
    book = BookState(
        quantities=np.zeros(1), cash=capital, mark_prices=np.ones(1), peak_value=capital
    )
    rate = -1.0 + 2.0**-52
    gap = book.apply_cash_interest(rate, year_fraction=0.75)
    basis = book.clone()
    basis.cash -= gap
    processing = basis.apply_cash_interest(rate, year_fraction=0.25)
    book.cash += processing
    assert book.cash == 2.5 and gap + processing == -capital + 2.0
    native = {
        "book": {"exact_quantities": ["0"], "cash": book.cash},
        "active_orders": [],
        "order_events": [],
        "execution": {
            "interval_cost": 0.0,
            "interval_funding": 0.0,
            "interval_borrow_cost": 0.0,
            "interval_dividend": 0.0,
            "interval_cash_interest": gap + processing,
        },
    }
    api._interval_continuity({"exact_quantities": ["0"], "cash": capital}, [], native)


def test_flat_cash_closure_rejects_doubled_sub_epsilon_capital():
    api = capability()
    native = {
        "book": {"exact_quantities": ["0"], "cash": 2e-14},
        "active_orders": [],
        "order_events": [],
        "execution": {
            "interval_cost": 0.0,
            "interval_funding": 0.0,
            "interval_borrow_cost": 0.0,
            "interval_dividend": 0.0,
            "interval_cash_interest": 0.0,
        },
    }
    with pytest.raises(ValueError):
        api._interval_continuity({"exact_quantities": ["0"], "cash": 1e-14}, [], native)


@pytest.mark.parametrize("mutation", ["direction", "zero", "price", "no_price"])
def test_retained_fill_quantity_and_price_cannot_contradict_the_order(mutation):
    api = capability()
    plan, raw = observed()
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    fill = next(
        event
        for event in raw["closing_rows"][0]["native"]["order_events"]
        if event["event_type"] == "filled"
    )
    assert fill["requested_quantity"] == fill["filled_quantity"] == -5.0
    assert fill["execution_price"] == 110.0 and fill["filled_notional"] == 550.0
    if mutation == "direction":
        fill["filled_quantity"] = 5.0
    elif mutation == "zero":
        fill["filled_quantity"] = 0.0
    else:
        fill["execution_price"] = None if mutation == "no_price" else 1.0
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize(
    "mutation", ["equity", "duplicate", "quantity", "mark", "multiplier"]
)
def test_normal_funding_boundary_must_match_its_returned_account(mutation):
    api = capability()
    plan, raw = observed()
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    native = raw["closing_rows"][0]["native"]
    boundaries = native["execution"]["funding_evidence"]
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary["funding_amount"] == 0.0
    assert boundary["equity_after_funding"] == native["book"]["equity"] == 1039.0
    if mutation == "equity":
        boundary["equity_before_funding"] += 1
        boundary["equity_after_funding"] += 1
    elif mutation == "duplicate":
        boundaries.append(deepcopy(boundary))
    elif mutation == "quantity":
        boundary["signed_quantities"][0] = 1.0
        boundary["funding_price_rates"][0] = 0.0
    elif mutation == "mark":
        boundary["mark_prices"][0] += 1.0
    else:
        boundary["contract_multipliers"][0] += 1.0
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_last_nonterminal_event_cannot_lose_its_known_active_order():
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        terminal_allocation_state_digest,
    )

    api = capability()
    plan, raw = observed(partial=True)
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    row = raw["closing_rows"][0]
    native = row["native"]
    assert native["active_orders"]
    assert native["order_events"][-1]["new_status"] == "partially_filled"
    native["active_orders"] = []
    row["after_state_digest"] = terminal_allocation_state_digest(
        plan.native_plan.payload["common"],
        plan.closing_source_digest,
        native,
        index=row["processing_index"],
        time_ns=row["processing_time_ns"],
    )
    raw["final_active_orders"] = []
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize(
    "mutation", ["ratio", "unfilled", "participation", "symbol_participation"]
)
def test_zero_request_cash_preserves_native_diagnostic_bounds(mutation):
    api = capability()
    plan, raw = observed(kind="cash")
    api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    execution = raw["closing_rows"][0]["native"]["execution"]
    assert execution["requested_notional"] == 0.0 and execution["fill_ratio"] == 1.0
    if mutation == "ratio":
        execution["fill_ratio"] = 0.0
    elif mutation == "unfilled":
        execution["unfilled_turnover"] = -1.0
    elif mutation == "participation":
        execution["max_participation"] = -1.0
    else:
        execution["participation_by_symbol"][0] = -1.0
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("bounds", "transition unsigned notionals/fill ratio exceed native bounds"),
        ("clock", "transition interval-local order clock differs"),
        ("empty", None),
    ],
)
def test_legacy_execution_error_precedes_unused_recipe_economics_lookup(
    mutation, message
):
    from tests.evaluation.test_allocation_static_cap_receipts import historical
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        validate_allocation_execution_facts,
    )

    _, raw = historical("uncapped_success")
    native = raw["rows"][0]["native"]
    recipe = raw["plan"]["cell"]["runtime"]["recipe"]
    del recipe["runtime_profile"]["economics_digest"]
    if mutation == "bounds":
        native["execution"]["fill_ratio"] = -1.0
    elif mutation == "clock":
        native["order_events"][0]["timestamp_ns"] += 1
    else:
        for field in ("order_events", "active_orders", "terminal_order_reasons"):
            native[field] = []
        context = native["proposal"]["decision"]["baseline"]["context"]
        expected = (native["book"]["equity"] - context["equity"]) / 1000.0
        assert validate_allocation_execution_facts(
            native,
            recipe,
            dataset_id=raw["plan"]["common"]["dataset_id"],
            action_code=2,
        ) == pytest.approx(expected)
        return
    with pytest.raises(ValueError, match=message) as caught:
        validate_allocation_execution_facts(
            native,
            recipe,
            dataset_id=raw["plan"]["common"]["dataset_id"],
            action_code=2,
        )
    assert caught.value.__cause__ is None


def test_partial_fill_followed_by_economic_cancellation_is_not_pending():
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(side=-1, partial=True)
    price_rates = env.dataset.resolved_array("funding_price_rate").copy()
    price_rates[8] = -1000.0
    object.__setattr__(env.dataset, "funding_price_rate", price_rates)
    plan = plans(runner, env, folds)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    native = raw["closing_rows"][0]["native"]
    partial = next(
        event
        for event in native["order_events"]
        if event["event_type"] == "partial_fill"
    )
    same_order = [
        event
        for event in native["order_events"]
        if event["order_id"] == partial["order_id"]
    ]
    assert same_order[-1]["event_type"] == "cancelled"
    assert native["active_orders"] == [] and native["book"]["insolvent"]
    assert native["execution"]["funding_evidence"][0]["signed_quantities"] == [-3.0]
    assert native["book"]["quantities"] == [0.0]
    receipt = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    ).payload
    assert receipt["status"] == "economic_stop"
    assert receipt["observed_profit_rate"] < -1.0
    assert receipt["settled_profit_rate"] is None


@pytest.mark.parametrize(
    "field",
    [
        "settled_profit_rate",
        "observed_profit_rate",
        "fixed_capital_reward_sum",
        "marked_prefix_profit_rate",
        "full_close_coverage",
        "risk_eligible",
        "settlement_complete",
        "final_active_orders",
        "final_book",
    ],
)
def test_forged_summary_rejected(field):
    api = capability()
    plan, raw = observed(partial=True)
    raw[field] = {
        "settled_profit_rate": 0.1,
        "observed_profit_rate": 0.1,
        "fixed_capital_reward_sum": 0.1,
        "marked_prefix_profit_rate": 0.1,
        "full_close_coverage": False,
        "risk_eligible": True,
        "settlement_complete": True,
        "final_active_orders": [],
        "final_book": None,
    }[field]
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "clock",
        "clock_type",
        "before",
        "after",
        "equity",
        "reward",
        "risk",
        "cost",
        "funding",
        "count",
        "dd",
        "symbol_total",
        "missing",
        "extra",
        "legacy",
    ],
)
def test_closed_row_and_version_tampering_rejected(mutation):
    api = capability()
    plan, raw = observed()
    row = raw["closing_rows"][0]
    if mutation == "clock":
        row["processing_time_ns"] += 1
    elif mutation == "clock_type":
        row["returned_clock_types"]["next_index"] = "int64"
    elif mutation in ("before", "after"):
        row[mutation + "_state_digest"] = "a" * 64
    elif mutation == "equity":
        row["opening_equity"] += 1
    elif mutation == "reward":
        row["fixed_capital_reward"] += 0.01
    elif mutation == "risk":
        row["risk"]["weights"][0] = 0.1
    elif mutation == "cost":
        row["native"]["execution"]["interval_cost"] += 1
    elif mutation == "funding":
        row["native"]["execution"]["interval_funding"] += 1
    elif mutation == "count":
        row["native"]["book"]["fill_count"] += 1
    elif mutation == "dd":
        row["native"]["book"]["max_drawdown"] = 0
    elif mutation == "symbol_total":
        row["native"]["execution"]["cost_by_symbol"][0] += 1
    elif mutation == "missing":
        del row["native"]["execution"]["interval_cost"]
    elif mutation == "extra":
        row["trusted"] = True
    else:
        raw["schema"] = "allocation_terminal_runtime_observation_v1"
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_external_plan_and_physical_endpoints_cannot_be_relabelled():
    api = capability()
    plan, raw = observed()
    changed = replace(
        plan,
        bound=replace(
            plan.bound, objective=replace(plan.bound.objective, maximum_drawdown=0.19)
        ),
    )
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=changed)
    forged = deepcopy(raw)
    forged["plan"]["objective"]["evaluation_start"] = "2026-01-01T05:00:00+00:00"
    forged["plan_digest"] = content_digest(forged["plan"])
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(forged, expected_plan=plan)


def test_cash_interest_covers_flat_reserve_and_rejects_missing_bar():
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(kind="cash", cash_rate=0.876)
    plan = plans(runner, env, folds, kind="cash", stop=9)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    receipt = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    )
    assert receipt.payload["final_book"]["cash"] == pytest.approx(1000 * 1.0001**3)
    raw["closing_rows"].pop()
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_flat_cash_interest_preserves_signed_debt_at_economic_stop():
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(kind="cash")
    rates = env.dataset.cash_rate.copy()
    rates[8] = -17520.0
    object.__setattr__(env.dataset, "cash_rate", rates)
    plan = plans(runner, env, folds, kind="cash")
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    result = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    ).payload
    assert result["status"] == "economic_stop"
    assert result["final_book"]["exact_quantities"] == ["0"]
    assert result["final_book"]["cash"] == pytest.approx(-1000.0)
    assert result["observed_profit_rate"] == pytest.approx(-2.0)
    assert result["fixed_capital_reward_sum"] == pytest.approx(-2.0)
    assert result["settled_profit_rate"] is None


def test_known_close_failure_frontier_keeps_original_error(monkeypatch):
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(partial=True)
    plan = plans(runner, env, folds, stop=9)
    native = reserved_closing._execute_allocation_target

    def close(*args, **kwargs):
        if kwargs["start_index"] == 8:
            raise RuntimeError("known failure")
        return native(*args, **kwargs)

    monkeypatch.setattr(reserved_closing, "_execute_allocation_target", close)
    with pytest.raises(RuntimeError, match="known failure") as caught:
        runner.run_terminal_allocation_execution(folds, env, plan)
    raw = caught.value.terminal_execution_receipt
    assert (
        type(
            raw["closing_rows"][0]["native"]["active_orders"][0]["intent"]["order_type"]
        )
        is str
    )
    receipt = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    )
    assert receipt.payload["failure"] == {
        "error_type": "RuntimeError",
        "message": "known failure",
    }
    assert receipt.payload["attempted_start_index"] == 8
    assert receipt.payload["settled_profit_rate"] is None
    raw["attempted_start_index"] = 7
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


@pytest.mark.parametrize(
    "side,price,equity,dd", [(1, 50, 742.0, 0.258), (-1, 400, -504.5, 1.0)]
)
def test_adverse_close_distinguishes_settlement_from_dd_and_signed_debt(
    side, price, equity, dd
):
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(side=side)
    for name in ("open", "high", "low", "close", "mark_price"):
        values = env.dataset.resolved_array(name).copy()
        values[8:] = price
        object.__setattr__(env.dataset, name, values)
    plan = plans(runner, env, folds)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    receipt = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    )
    result = receipt.payload
    assert result["final_book"]["equity"] == pytest.approx(equity)
    assert result["final_book"]["max_drawdown"] == pytest.approx(dd)
    assert not result["risk_eligible"]
    assert result["fixed_capital_reward_sum"] == pytest.approx((equity - 1000) / 1000)
    if side == -1:
        assert (
            result["settled_profit_rate"] is None
            and result["status"] == "economic_stop"
        )
        assert result["observed_profit_rate"] < -1
    else:
        assert result["settlement_complete"] and result["settled_profit_rate"] < 0


@pytest.mark.parametrize("block", ["tradability", "turnover"])
def test_zero_target_can_be_blocked_by_native_execution_or_hard_risk(block):
    from trade_rl.risk import PreTradeRiskConfig

    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    risk = (
        PreTradeRiskConfig(max_gross=1, max_abs_weight=1, max_turnover=0.1)
        if block == "turnover"
        else None
    )
    folds, env = carrier(risk=risk)
    if block == "tradability":
        values = env.dataset.tradable.copy()
        values[8:] = False
        object.__setattr__(env.dataset, "tradable", values)
    plan = plans(runner, env, folds)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    result = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    ).payload
    assert result["final_book"]["exact_quantities"] != ["0"]
    assert not result["settlement_complete"] and result["settled_profit_rate"] is None


def test_close_to_close_state_chain_and_late_funding_preserve_returned_book():
    api = capability()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    folds, env = carrier(partial=True)
    plan = plans(runner, env, folds, stop=9)
    raw = runner.run_terminal_allocation_execution(folds, env, plan).payload
    result = api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    ).payload
    assert (
        result["closing_rows"][0]["after_state_digest"]
        == result["closing_rows"][1]["before_state_digest"]
    )
    assert result["final_book"]["funding_pnl"] == pytest.approx(-0.83)
    assert result["final_book"]["exact_quantities"] == ["0"]
    first = deepcopy(raw["closing_rows"][0])
    raw["closing_rows"][1]["before_state_digest"] = first["before_state_digest"]
    with pytest.raises(ValueError):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)


def test_shifted_external_objective_with_identical_duration_is_not_the_source_window():
    api = capability()
    plan, raw = observed()
    objective = replace(
        plan.bound.objective,
        evaluation_start=plan.bound.objective.evaluation_start + timedelta(hours=1),
        evaluation_stop_exclusive=plan.bound.objective.evaluation_stop_exclusive
        + timedelta(hours=1),
    )
    shifted = replace(plan, bound=replace(plan.bound, objective=objective))
    raw["plan"], raw["plan_digest"] = shifted.payload, shifted.digest
    with pytest.raises(ValueError, match="physical source clocks"):
        api.TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=shifted)


def test_closed_reader_is_pure_and_uses_no_dataset_or_executor(monkeypatch):
    api = capability()
    plan, raw = observed()
    runner = import_module("trade_rl.evaluation.allocation_terminal_execution")
    monkeypatch.setattr(
        reserved_closing,
        "_execute_allocation_target",
        lambda *_a, **_k: pytest.fail("reader executed"),
    )
    monkeypatch.setattr(
        runner,
        "declare_terminal_allocation_execution",
        lambda *_a, **_k: pytest.fail("reader redeclared a Dataset"),
    )
    assert api.TerminalAllocationExecutionReceipt.from_payload(
        raw, expected_plan=plan
    ).payload["settlement_complete"]


def test_native_turnover_denominator_keeps_floor_for_tiny_equity():
    api = capability()
    # Native turnover uses a 1e-12 denominator floor; economic reward does not.
    previous = {
        "equity": 1e-14,
        "cash": 1e-14,
        "exact_quantities": ["0"],
        "peak_value": 1e-14,
        "max_drawdown": 0.0,
        "turnover_total": 0.0,
        "total_cost": 0.0,
        "funding_pnl": 0.0,
        "borrow_cost": 0.0,
        "fill_count": 0,
        "rebalance_events": 0,
    }
    native = {
        "book": dict(
            previous,
            insolvent=False,
            termination_reason=None,
            quantities=[0.0],
            mark_prices=[1.0],
            contract_multipliers=[1.0],
            margin_used=0.0,
            maintenance_margin=0.0,
            maintenance_requirement=0.0,
            margin_deficit=0.0,
        ),
        "order_events": [],
        "active_orders": [],
        "execution": {
            "interval_net_return": 0.0,
            "interval_log_return": 0.0,
            "interval_gross_return": 0.0,
            "interval_cost": 0.0,
            "interval_funding": 0.0,
            "interval_borrow_cost": 0.0,
            "interval_dividend": 0.0,
            "interval_cash_interest": 0.0,
            "fill_count": 0,
            "rebalance_events": 0,
            "completed_fill_count": 0,
            "rejected_count": 0,
            "expired_count": 0,
            "requested_notional": 1e-15,
            "requested_notional_by_symbol": [1e-15],
            "requested_turnover": 0.001,
            "filled_notional": 0.0,
            "filled_notional_by_symbol": [0.0],
            "filled_turnover": 0.0,
            "cost_by_symbol": [0.0],
            "funding_evidence": [],
            "capacity_evidence": [],
            "unfilled_turnover": 0.001,
            "max_participation": 0.0,
            "participation_by_symbol": [0.0],
            "fill_ratio": 1.0,
        },
    }
    api._metrics(
        previous,
        native,
        previous_orders=[],
        cost={
            "max_participation_rate": 1.0,
            "max_leverage": 1.0,
            "maintenance_margin_rate": 0.0,
            "collateral_haircut": 1.0,
            "margin_mode": "cross",
        },
        symbol_index=0,
    )
