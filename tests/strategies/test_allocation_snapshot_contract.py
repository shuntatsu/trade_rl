import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.strategies.allocation_snapshot import AllocationAccountSnapshot


def declared_book():
    return dict(
        quantities=[5.0],
        cash=-200.0,
        mark_prices=[100.0],
        peak_value=400.0,
        contract_multipliers=[1.0],
        max_drawdown=0.25,
        turnover_total=1.0,
        total_cost=1.0,
        funding_pnl=0.0,
        fill_count=1,
        rebalance_events=1,
        returns_history=[0.01],
        borrow_cost=0.0,
        margin_used=100.0,
        maintenance_margin=0.05,
        maintenance_requirement=25.0,
        margin_deficit=0.0,
        insolvent=False,
        termination_reason=None,
        as_of_index=6,
        as_of_dataset_id="a" * 64,
        exact_quantities=["5"],
        equity=300.0,
        current_drawdown=0.25,
    )


def declared_order():
    return dict(
        intent=dict(
            order_id="b" * 64,
            dataset_id="a" * 64,
            target_identity="target",
            execution_policy_digest="c" * 64,
            symbol_index=4,
            requested_quantity=1.0,
            order_type="market",
            time_in_force="gtc",
            limit_price=None,
            stop_price=None,
            submit_index=1,
            eligible_index=2,
            expiry_index=None,
            submission_reference_price=100.0,
            decision_equity=300.0,
            replaced_order_id=None,
            reduce_only=False,
        ),
        remaining_quantity=float(Fraction(2, 3)),
        cumulative_filled_quantity=float(Fraction(1, 3)),
        cumulative_filled_notional=100.0 / 3,
        status="partially_filled",
        trigger_index=2,
        last_processed_index=3,
        terminal_reason=None,
        evidence_version=1,
        exact_cumulative_filled_quantity="1/3",
        exact_requested_quantity="1",
        exact_remaining_quantity="2/3",
    )


def snapshot(book=None, orders=None, **changes):
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
        book_facts=declared_book() if book is None else book,
        active_orders=(declared_order(),) if orders is None else orders,
    )
    return AllocationAccountSnapshot(**(values | changes))


@pytest.mark.parametrize("status", ["eligible", "triggered", "partially_filled"])
@pytest.mark.parametrize("transition", [0, 1, None])
def test_active_progress_cannot_precede_submission_or_eligibility(status, transition):
    order = declared_order()
    order.update(status=status, last_processed_index=transition, trigger_index=None)
    if status != "partially_filled":
        order.update(
            cumulative_filled_quantity=0.0, exact_cumulative_filled_quantity="0"
        )
        order.update(remaining_quantity=1.0, exact_remaining_quantity="1")
    with pytest.raises(ValueError, match="transition|eligib"):
        snapshot(orders=(order,))


def test_any_recorded_transition_cannot_precede_submission():
    order = declared_order()
    order.update(status="latency_wait", last_processed_index=0, trigger_index=None)
    with pytest.raises(ValueError, match="transition|submission"):
        snapshot(orders=(order,))


@pytest.mark.parametrize("status, last", [("submitted", None), ("latency_wait", 1)])
def test_waiting_before_eligibility_and_older_valid_transition_remain_allowed(
    status, last
):
    order = declared_order()
    order.update(status=status, last_processed_index=last, trigger_index=None)
    order.update(cumulative_filled_quantity=0.0, exact_cumulative_filled_quantity="0")
    order.update(remaining_quantity=1.0, exact_remaining_quantity="1")
    order["cumulative_filled_notional"] = 0.0
    assert snapshot(orders=(order,)).active_orders[0]["status"] == status
    assert snapshot().active_orders[0]["last_processed_index"] == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "submitted", "trigger_index": None, "last_processed_index": 1},
        {"status": "latency_wait", "trigger_index": None},
        {"status": "eligible", "trigger_index": None},
        {"status": "triggered", "trigger_index": 1},
        {"status": "triggered", "trigger_index": 4, "last_processed_index": 3},
        {"status": "triggered", "trigger_index": None},
        {
            "exact_cumulative_filled_quantity": "1",
            "cumulative_filled_quantity": 1.0,
            "exact_remaining_quantity": "0",
            "remaining_quantity": 0.0,
        },
        {
            "exact_cumulative_filled_quantity": "0",
            "cumulative_filled_quantity": 0.0,
            "exact_remaining_quantity": "1",
            "remaining_quantity": 1.0,
        },
    ],
)
def test_status_and_exact_progress_describe_a_possible_native_active_state(changes):
    order = declared_order() | changes
    with pytest.raises(ValueError):
        snapshot(orders=(order,))


def test_partial_fill_without_trigger_and_retriggered_partial_order_remain_allowed():
    assert snapshot(orders=(declared_order() | {"trigger_index": None},)).digest
    assert snapshot(orders=(declared_order() | {"status": "triggered"},)).digest


@pytest.mark.parametrize("status, last", [("submitted", 1), ("latency_wait", None)])
def test_waiting_status_matches_whether_a_native_transition_was_processed(status, last):
    order = declared_order() | {
        "status": status,
        "last_processed_index": last,
        "trigger_index": None,
        "cumulative_filled_quantity": 0.0,
        "exact_cumulative_filled_quantity": "0",
        "remaining_quantity": 1.0,
        "exact_remaining_quantity": "1",
        "cumulative_filled_notional": 0.0,
    }
    with pytest.raises(ValueError, match="waiting status"):
        snapshot(orders=(order,))


@pytest.mark.parametrize(
    "status", ("submitted", "latency_wait", "eligible", "triggered")
)
def test_zero_filled_quantity_cannot_have_positive_filled_notional(status):
    order = declared_order() | {
        "status": status,
        "last_processed_index": None if status == "submitted" else 2,
        "trigger_index": 2 if status == "triggered" else None,
        "cumulative_filled_quantity": 0.0,
        "exact_cumulative_filled_quantity": "0",
        "remaining_quantity": 1.0,
        "exact_remaining_quantity": "1",
        "cumulative_filled_notional": 1.0,
    }
    with pytest.raises(ValueError, match="notional"):
        snapshot(orders=(order,))


@pytest.mark.parametrize("status", ["submitted", "latency_wait", "eligible"])
def test_prefill_state_cannot_retain_a_recorded_trigger(status):
    order = declared_order() | dict(
        status=status,
        cumulative_filled_quantity=0.0,
        exact_cumulative_filled_quantity="0",
        remaining_quantity=1.0,
        exact_remaining_quantity="1",
    )
    with pytest.raises(ValueError):
        snapshot(orders=(order,))


def test_completed_order_cannot_reenter_triggered_active_state():
    order = declared_order() | dict(
        status="triggered",
        cumulative_filled_quantity=1.0,
        exact_cumulative_filled_quantity="1",
        remaining_quantity=0.0,
        exact_remaining_quantity="0",
    )
    with pytest.raises(ValueError):
        snapshot(orders=(order,))


@pytest.mark.parametrize(
    "field, value",
    [
        ("eligible_index", 0),
        ("time_in_force", "day"),
        ("limit_price", 100.0),
        ("stop_price", 100.0),
        ("submission_reference_price", 0.0),
        ("decision_equity", True),
        ("cumulative_filled_notional", -1.0),
        ("evidence_version", True),
    ],
)
def test_native_intent_and_order_value_bounds_reject(field, value):
    order = declared_order()
    destination = order["intent"] if field in order["intent"] else order
    destination[field] = value
    with pytest.raises(ValueError):
        snapshot(orders=(order,))


def test_snapshot_is_detached_deeply_immutable_and_canonical():
    book, order = declared_book(), declared_order()
    item = snapshot(book=book, orders=(order,))
    before = item.canonical_bytes()
    book["quantities"][0] = 99.0
    order["intent"]["reduce_only"] = True
    assert item.canonical_bytes() == before
    assert item.digest == content_digest(json.loads(before))
    assert before == canonical_json_bytes(json.loads(before))
    with pytest.raises(TypeError):
        item.book_facts["cash"] = 0.0
    with pytest.raises(TypeError):
        item.book_facts["quantities"][0] = 99.0
    with pytest.raises(TypeError):
        item.active_orders[0]["intent"]["reduce_only"] = True
    with pytest.raises(FrozenInstanceError):
        item.symbol_index = 0
    payload = item.payload()
    payload["account_id"] = "changed"
    assert item.canonical_bytes() == before


def test_global_slot_selects_one_vector_and_signed_orders_are_digest_bound():
    item = snapshot()
    assert item.symbol_index == 4 and len(item.book_facts["quantities"]) == 1
    assert item.book_facts["cash"] == -200.0 and item.book_facts["equity"] == 300.0
    negative = deepcopy(declared_order())
    for name in ("remaining_quantity", "cumulative_filled_quantity"):
        negative[name] *= -1
    for name in (
        "exact_remaining_quantity",
        "exact_cumulative_filled_quantity",
        "exact_requested_quantity",
    ):
        negative[name] = str(-Fraction(negative[name]))
    negative["intent"]["requested_quantity"] = -1.0
    opposed = snapshot(orders=(negative,))
    assert abs(item.active_orders[0]["remaining_quantity"]) == abs(
        opposed.active_orders[0]["remaining_quantity"]
    )
    assert item.digest != opposed.digest
    assert replace(item, source_state_digest="f" * 64).digest != item.digest
    assert replace(item, observable_tradable=False).digest != item.digest
    assert replace(item, account_id="other").digest != item.digest


@pytest.mark.parametrize("exact", [Fraction(1, 3), Fraction(-1, 3)])
def test_exact_thirds_accept_only_conservative_one_ulp_reporting(exact):
    book = declared_book()
    book["exact_quantities"] = [str(exact)]
    book["quantities"] = [float(np.nextafter(float(exact), 0.0))]
    assert snapshot(book=book).book_facts["exact_quantities"] == (str(exact),)
    book["quantities"] = [float(np.nextafter(float(exact), float(exact) * 2))]
    with pytest.raises(ValueError, match="quantity"):
        snapshot(book=book)


@pytest.mark.parametrize(
    "field, value",
    [
        ("symbol_index", True),
        ("decision_index", -1),
        ("observable_tradable", 1),
        ("available_at", "2026-01-01T06:00:00.000000001"),
        ("decision_time", "2500-01-01T00:00:00.000000000"),
        ("decision_time", "NaT"),
        ("decision_time", "2026-01-01T06:00:00Z"),
        ("risk_digest", "x" * 64),
        ("account_id", " "),
        ("active_orders", []),
    ],
)
@pytest.mark.filterwarnings("error")
def test_outer_scope_and_causal_nanosecond_clock_reject(field, value):
    with pytest.raises(ValueError):
        snapshot(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("cash", True),
        ("cash", float("nan")),
        ("equity", 0.0),
        ("peak_value", -1.0),
        ("insolvent", True),
        ("termination_reason", "margin_call"),
        ("as_of_index", 5),
        ("as_of_dataset_id", "f" * 64),
        ("quantities", [5.0, 0.0]),
        ("exact_quantities", ["5/1"]),
        ("mark_prices", [0.0]),
        ("contract_multipliers", [True]),
        ("extra", None),
    ],
)
def test_live_book_schema_shape_clock_and_financial_values_reject(field, value):
    with pytest.raises(ValueError):
        snapshot(book=declared_book() | {field: value})


@pytest.mark.parametrize("time_in_force", ["ioc", "day", "gtc"])
def test_native_tif_expiry_reduce_only_and_exact_identity_are_retained(time_in_force):
    order = declared_order()
    order["intent"].update(
        time_in_force=time_in_force, expiry_index=7, reduce_only=True
    )
    assert snapshot(orders=(order,)).active_orders[0]["intent"]["reduce_only"] is True


@pytest.mark.parametrize(
    "section, field, value",
    [
        ("intent", "symbol_index", 0),
        ("intent", "dataset_id", "f" * 64),
        ("intent", "execution_policy_digest", "f" * 64),
        ("intent", "order_type", "limit"),
        ("intent", "time_in_force", "fok"),
        ("intent", "expiry_index", 6),
        ("intent", "reduce_only", 1),
        ("intent", "submit_index", True),
        ("intent", "extra", None),
        ("order", "status", "filled"),
        ("order", "terminal_reason", "cancelled"),
        ("order", "last_processed_index", 7),
        ("order", "exact_remaining_quantity", "1/2"),
        ("order", "remaining_quantity", True),
        ("order", "extra", None),
    ],
)
def test_nested_native_order_schema_scope_and_progress_reject(section, field, value):
    order = declared_order()
    destination = order["intent"] if section == "intent" else order
    destination[field] = value
    with pytest.raises(ValueError):
        snapshot(orders=(order,))
