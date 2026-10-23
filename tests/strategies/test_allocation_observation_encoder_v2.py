import json
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from importlib import import_module

import numpy as np
import pytest

from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
)
from trade_rl.strategies.allocation_action import (
    AllocationActionContract,
    AllocationDecision,
)
from trade_rl.strategies.allocation_snapshot import AllocationAccountSnapshot
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema


@pytest.fixture(scope="module")
def api():
    return import_module("trade_rl.strategies.rl.allocation_observation_encoder_v2")


def declared_order(sign=1):
    order = json.loads("""{
      "intent": {"target_identity":"target", "symbol_index":4, "order_type":"market",
        "time_in_force":"gtc", "limit_price":null, "stop_price":null, "submit_index":1,
        "eligible_index":2, "expiry_index":null, "submission_reference_price":110.0,
        "decision_equity":300.0, "replaced_order_id":null, "reduce_only":false},
      "cumulative_filled_notional":33.333333333333336, "status":"partially_filled",
      "trigger_index":2, "last_processed_index":3, "terminal_reason":null, "evidence_version":1
    }""")
    order["intent"].update(
        order_id="b" * 64,
        dataset_id="a" * 64,
        execution_policy_digest="c" * 64,
        requested_quantity=float(sign),
    )
    order.update(
        remaining_quantity=float(sign * Fraction(2, 3)),
        cumulative_filled_quantity=float(sign * Fraction(1, 3)),
        exact_requested_quantity=str(sign),
        exact_cumulative_filled_quantity=str(sign * Fraction(1, 3)),
        exact_remaining_quantity=str(sign * Fraction(2, 3)),
    )
    return order


def declared_snapshot(orders=None, **changes):
    book = json.loads("""{
      "quantities":[5.0], "cash":-200.0, "mark_prices":[100.0], "peak_value":400.0,
      "contract_multipliers":[1.0], "max_drawdown":0.5, "turnover_total":1.0,
      "total_cost":1.0, "funding_pnl":0.0, "fill_count":1, "rebalance_events":1,
      "returns_history":[0.01], "borrow_cost":0.0, "margin_used":100.0,
      "maintenance_margin":0.05, "maintenance_requirement":25.0, "margin_deficit":0.0,
      "insolvent":false, "termination_reason":null, "as_of_index":6,
      "exact_quantities":["5"], "equity":300.0, "current_drawdown":0.25
    }""")
    book["as_of_dataset_id"] = "a" * 64
    values = dict(
        account_id="independent-4",
        dataset_id="a" * 64,
        symbol="B",
        symbol_index=4,
        decision_index=6,
        decision_time="2026-01-01T06:00:00.000000000",
        available_at="2026-01-01T06:00:00.000000000",
        execution_policy_digest="c" * 64,
        risk_digest="d" * 64,
        source_state_digest="e" * 64,
        observable_tradable=True,
        book_facts=book,
        active_orders=(declared_order(),) if orders is None else orders,
    )
    return AllocationAccountSnapshot(**(values | changes))


def declared_decision(snapshot=None, **changes):
    snapshot = snapshot or declared_snapshot()
    time = np.datetime64(snapshot.decision_time)
    returns = (0.02, 0.04, 0.001, 0.002, 0.003, -0.004, 0.005, 0.006)
    inputs = AllocationInputs(
        "B",
        time,
        time - np.timedelta64(1, "h"),
        time + np.timedelta64(1, "h"),
        "forecast",
        *returns,
    )
    facts = (
        "independent-4",
        "B",
        time,
        "e" * 64,
        5 / 3,
        "5",
        -200.0,
        300.0,
        "reporting-not-an-exact-remainder",
    )
    context = AllocationContext(*facts)
    baseline = AfterCostTargetAllocator().propose(inputs, context)
    values = dict(
        baseline=baseline,
        action_contract=AllocationActionContract(),
        feature_names=("alpha", "beta"),
        feature_values=(2.0, -3.0),
        max_drawdown=0.5,
        initial_capital=100.0,
        remaining_steps=8,
        pending_gross=999.0,
        pending_count=len(snapshot.active_orders),
    )
    return AllocationDecision(**(values | changes))


def schema(**changes):
    values = dict(
        feature_names=("alpha", "beta"),
        max_active_orders=2,
        initial_capital=100.0,
        episode_steps=16,
    )
    return AllocationObservationSchema(**(values | changes))


def encode(api, snapshot=None, decision=None, **changes):
    snapshot = snapshot or declared_snapshot()
    return api.encode_allocation_observation_v2(
        snapshot,
        decision or declared_decision(snapshot),
        schema=schema(),
        episode_steps=16,
        **changes,
    )


def oracle(value):
    """Independent binary-value comparison, never a production projection helper."""
    exact = value if isinstance(value, Fraction) else Fraction(str(value))
    result = np.float32(float(exact))
    if abs(Fraction.from_float(float(result))) > abs(exact):
        result = np.nextafter(result, np.float32(0))
    assert abs(Fraction.from_float(float(result))) <= abs(exact)
    return result


def test_full_declared_vector_and_inputs_are_detached(api):
    snapshot, decision, contract = declared_snapshot(), declared_decision(), schema()
    before = snapshot.canonical_bytes(), decision.decision_digest
    result = api.encode_allocation_observation_v2(
        snapshot, decision, schema=contract, episode_steps=16
    )
    tokens = (
        "2 -3 .02 .04 .001 .002 .003 -.004 .005 .006 1 "
        "5 5/3 -2 3 4 .25 .5 1 .05 .25 0 1 .5 "
        "1 1 1/3 2/3 .33333333333333336 11/10 3 -5/16 -4/16 "
        "0 0 1 -4/16 1 -3/16 0 0 0 1 0 0 0 0 1"
    )
    expected = [Fraction(value) for value in tokens.split()] + [0] * 24
    np.testing.assert_array_equal(result, [oracle(value) for value in expected])
    assert result.dtype == np.float32 and result.shape == (2 + 22 + 24 * 2,)
    result[:] = 0
    assert np.any(encode(api))
    assert before == (snapshot.canonical_bytes(), decision.decision_digest)


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_signed_thirds_are_never_overstated(api, sign):
    result = encode(api, declared_snapshot((declared_order(sign),)))
    for index, exact in (
        (25, Fraction(sign)),
        (26, sign * Fraction(1, 3)),
        (27, sign * Fraction(2, 3)),
    ):
        represented = Fraction.from_float(float(result[index]))
        assert represented * exact > 0 and abs(represented) <= abs(exact)
        assert result[index] == oracle(exact)


def test_economic_sort_is_id_and_input_order_independent(api):
    buy, sell = declared_order(), declared_order(-1)
    original = encode(api, declared_snapshot((buy, sell)))
    buy["intent"]["order_id"], sell["intent"]["order_id"] = "f" * 64, "a" * 64
    np.testing.assert_array_equal(original, encode(api, declared_snapshot((sell, buy))))
    assert original[25] == -1 and original[49] == 1
    for key, value in (
        ("submission_reference_price", 120),
        ("decision_equity", 310),
        ("reduce_only", True),
        ("time_in_force", "ioc"),
    ):
        changed = deepcopy(buy)
        changed["intent"][key] = value
        assert not np.array_equal(
            encode(api), encode(api, declared_snapshot((changed,)))
        )
    changed = deepcopy(buy)
    changed["cumulative_filled_notional"] = 34
    assert not np.array_equal(encode(api), encode(api, declared_snapshot((changed,))))
    earlier, later = declared_order(), declared_order()
    earlier["intent"]["submission_reference_price"] = 120
    later.update(
        exact_cumulative_filled_quantity=str(Fraction(1, 3) + Fraction(1, 10**20)),
        exact_remaining_quantity=str(Fraction(2, 3) - Fraction(1, 10**20)),
    )
    tied_projection = encode(api, declared_snapshot((later, earlier)))
    assert tied_projection[26] == tied_projection[50]
    assert tied_projection[29] == oracle(Fraction(6, 5))


def test_order_slots_reject_overflow_and_distinguish_zero_clock_from_absence(api):
    with pytest.raises(ValueError, match="order"):
        encode(api, declared_snapshot(tuple(declared_order() for _ in range(3))))
    empty = encode(api, declared_snapshot(()))
    assert not empty[24:].any()
    order = declared_order()
    order.update(
        status="submitted",
        trigger_index=None,
        last_processed_index=None,
        exact_cumulative_filled_quantity="0",
        cumulative_filled_quantity=0.0,
        cumulative_filled_notional=0.0,
        exact_remaining_quantity="1",
        remaining_quantity=1.0,
    )
    order["intent"].update(submit_index=0, eligible_index=1)
    snap = declared_snapshot((order,))
    book = dict(snap.book_facts) | {"as_of_index": 0}
    snap = replace(
        snap,
        decision_index=0,
        book_facts=book,
        decision_time="2026-01-01T00:00:00.000000000",
        available_at="2026-01-01T00:00:00.000000000",
    )
    absent = encode(api, snap)
    order.update(status="latency_wait", last_processed_index=0)
    present = encode(api, replace(snap, active_orders=(order,)))
    assert absent[37] == 0 and present[37] == 1
    assert absent[38] == present[38] == 0
    order["intent"].update(time_in_force="day", expiry_index=1)
    day = encode(api, replace(snap, active_orders=(order,)))
    assert day[33] == 1 and day[34] == oracle(Fraction(1, 16)) and day[41] == 1
    partial = declared_order()
    partial["intent"].update(submit_index=0, eligible_index=0)
    partial.update(trigger_index=None, last_processed_index=0)
    no_trigger = encode(api, replace(snap, active_orders=(partial,)))
    partial["trigger_index"] = 0
    triggered = encode(api, replace(snap, active_orders=(partial,)))
    assert no_trigger[35] == 0 and triggered[35] == 1
    assert no_trigger[36] == triggered[36] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"account_id": "other"},
        {"symbol": "C"},
        {"source_state_digest": "f" * 64},
        {"decision_time": "2026-01-01T07:00:00.000000000"},
    ],
)
def test_snapshot_decision_identity_mismatch_rejected(api, change):
    with pytest.raises(ValueError, match="match"):
        encode(api, replace(declared_snapshot(), **change), declared_decision())


@pytest.mark.parametrize(
    "key,value",
    [
        ("cash", -201),
        ("equity", 301),
        ("max_drawdown", 0.6),
        ("exact_quantities", ["4"]),
        ("current_drawdown", True),
        ("current_drawdown", -0.1),
        ("max_drawdown", 1.1),
        ("margin_used", -1),
        ("margin_used", np.inf),
        ("maintenance_margin", 1.1),
        ("maintenance_requirement", True),
        ("margin_deficit", np.nan),
    ],
)
def test_consumed_book_facts_are_bound_and_validated(api, key, value):
    snap = declared_snapshot()
    book = dict(snap.book_facts) | {key: value}
    if key == "exact_quantities":
        book["quantities"] = [4.0]
    with pytest.raises(ValueError):
        encode(api, replace(snap, book_facts=book), declared_decision())


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_capital", 101),
        ("remaining_steps", 17),
        ("feature_names", ("other", "beta")),
        ("pending_count", 0),
        ("feature_values", (1e39, 1)),
        ("feature_values", (1e-60, 1)),
    ],
)
def test_decision_shape_horizon_and_float32_extremes_rejected(api, field, value):
    with pytest.raises(ValueError):
        encode(api, decision=replace(declared_decision(), **{field: value}))


def test_reporting_summaries_are_not_exact_slots_and_v1_remains_available(api):
    from trade_rl.strategies.rl.allocation_policy import (
        AllocationRuntimeProfile,
        allocation_recipe_payload,
        encode_allocation_observation,
    )

    decision = declared_decision()
    v1 = encode_allocation_observation(decision, episode_steps=16)
    expected = "2 -3 .02 .04 .001 .002 .003 -.004 .005 .006 1 1.6666666666666667 -2 3 .5 999 1 .5"
    np.testing.assert_array_equal(v1, np.array(expected.split(), dtype=np.float32))
    recipe = allocation_recipe_payload(
        decision.action_contract,
        decision.feature_names,
        allocator=decision.baseline.allocator,
        expected_horizon_seconds=3600,
        runtime_profile=AllocationRuntimeProfile(
            "a" * 64, "d" * 64, 100, "USD", 3600, 3600
        ),
    )
    assert recipe["schema"] == "allocation_ppo_recipe_v1"
    assert recipe["observation"]["schema"] == "allocation_account_observation_v1"
    names = "expected_simple_return return_variance buy_cost sell_cost exit_cost funding_return borrow_return cash_return baseline_target_weight current_weight cash_over_initial_capital equity_over_initial_capital maximum_drawdown pending_gross_over_initial_capital pending_order_count remaining_horizon_fraction"
    assert recipe["observation"]["fields"] == names.split()
    altered = replace(decision, pending_gross=123)
    np.testing.assert_array_equal(
        encode(api, decision=decision), encode(api, decision=altered)
    )
    for horizon in (True, 0, 15, 32):
        with pytest.raises(ValueError):
            api.encode_allocation_observation_v2(
                declared_snapshot(), decision, schema=schema(), episode_steps=horizon
            )


@pytest.mark.parametrize(
    "value",
    [
        np.float32(np.finfo(np.float32).max),
        np.nextafter(np.float32(0), np.float32(1)),
        0.1,
        -0.1,
    ],
)
def test_raw_feature_guard_retains_representable_float32_extremes(api, value):
    result = encode(
        api, decision=replace(declared_decision(), feature_values=(float(value), 1))
    )
    assert result[0] == np.float32(value)


def test_clock_offsets_ignore_large_common_absolute_index_and_evidence_ids(api):
    original = declared_snapshot()
    shifted_order = declared_order()
    shift = 2**54
    for field in ("submit_index", "eligible_index"):
        shifted_order["intent"][field] += shift
    for field in ("trigger_index", "last_processed_index"):
        shifted_order[field] += shift
    shifted_order["intent"].update(
        order_id="f" * 64, target_identity="unrelated", replaced_order_id="g" * 64
    )
    shifted_order["evidence_version"] = 99
    shifted = replace(
        original,
        decision_index=6 + shift,
        book_facts=dict(original.book_facts) | {"as_of_index": 6 + shift},
        active_orders=(shifted_order,),
    )
    np.testing.assert_array_equal(encode(api, original), encode(api, shifted))


def test_exact_normalization_precedes_potentially_overflowing_products(api):
    original = declared_snapshot()
    order = declared_order()
    order.update(
        status="submitted",
        trigger_index=None,
        last_processed_index=None,
        exact_requested_quantity=str(Fraction("1e120")),
        exact_cumulative_filled_quantity="0",
        cumulative_filled_quantity=0,
        exact_remaining_quantity=str(Fraction("1e120")),
        remaining_quantity=1e120,
        cumulative_filled_notional=0,
    )
    order["intent"].update(
        requested_quantity=1e120,
        submission_reference_price=1e200,
        decision_equity=1e308,
    )
    book = dict(original.book_facts) | dict(
        quantities=[1e100],
        exact_quantities=[str(Fraction("1e100"))],
        mark_prices=[1e200],
        cash=1e308,
        equity=1e308,
        peak_value=1e308,
        margin_used=1e308,
        maintenance_requirement=1e308,
        current_drawdown=0,
    )
    snap = replace(original, book_facts=book, active_orders=(order,))
    decision = declared_decision()
    context = replace(
        decision.baseline.context,
        quantity=str(Fraction("1e100")),
        current_weight=1e100 * 1e200 / 1e308,
        cash=1e308,
        equity=1e308,
    )
    baseline = decision.baseline.allocator.propose(decision.baseline.inputs, context)
    decision = replace(decision, baseline=baseline, initial_capital=1e308)
    result = api.encode_allocation_observation_v2(
        snap, decision, schema=schema(initial_capital=1e308), episode_steps=16
    )
    assert result[25] == oracle(Fraction(10**12))


def test_native_capital_normalization_matches_the_existing_decision(api):
    capital = np.float32(1 / 3)
    contract = schema(initial_capital=capital)
    decision = declared_decision(initial_capital=capital)
    assert contract.initial_capital == decision.initial_capital == float(capital)
    api.encode_allocation_observation_v2(
        declared_snapshot(), decision, schema=contract, episode_steps=16
    )


@pytest.mark.parametrize(
    "remaining", [Fraction(1, 10), Fraction(1, 2**149), Fraction(1, 2**150)]
)
def test_exact_economic_projection_is_closest_conservative_or_rejects_underflow(
    api, remaining
):
    order = declared_order()
    cumulative = 1 - remaining
    reported = float(cumulative)
    if Fraction(str(reported)) > cumulative:
        reported = np.nextafter(reported, 0.0)
    order.update(
        exact_remaining_quantity=str(remaining),
        remaining_quantity=float(remaining),
        exact_cumulative_filled_quantity=str(cumulative),
        cumulative_filled_quantity=reported,
    )
    snap = declared_snapshot((order,))
    if remaining < Fraction(1, 2**149):
        with pytest.raises(ValueError, match="underflow"):
            encode(api, snap)
    else:
        result = encode(api, snap)[27]
        assert 0 < Fraction.from_float(float(result)) <= remaining
        away = np.nextafter(result, np.float32(np.inf))
        assert Fraction.from_float(float(away)) > remaining
        if remaining == Fraction(1, 10):
            assert result == np.float32(Fraction(3355443, 33554432))
