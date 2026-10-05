"""No-fit observation oracles over declared canonical account/order facts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.simulation.quantities import project_quantity


def market(*, future_price=100.0, future_volume=1000.0, available=True, n_symbols=1):
    price = np.full((4, n_symbols), 100.0)
    price[2:] = future_price
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(4) * np.timedelta64(
        1, "h"
    )
    known = np.repeat(timestamps[:, None], n_symbols, axis=1)
    if not available:
        known[1, 0] += np.timedelta64(1, "m")
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=tuple(f"S{i}" for i in range(n_symbols)),
        timestamps=timestamps,
        features=np.zeros((4, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=np.repeat(
            np.array([[1000.0], [1000.0], [future_volume], [future_volume]]),
            n_symbols,
            axis=1,
        ),
        volume_units=(VolumeUnit.BASE_ASSET,) * n_symbols,
        funding_rate=np.repeat(
            np.array([[0.0], [0.0], [future_price / 10000], [0.0]]), n_symbols, axis=1
        ),
        tradable=np.ones((4, n_symbols), dtype=np.bool_),
        feature_available=np.ones((4, n_symbols, 1), dtype=np.bool_),
        available_at=known,
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8760,
    )


def runtime(dataset=None, *, quantity=2.5, exact_quantity=None, symbol_index=0):
    dataset = market() if dataset is None else dataset
    executor = MarketExecutor(
        dataset, replace(ExecutionCostConfig.zero(), maintenance_margin_rate=0.01)
    )
    quantities = np.zeros(dataset.n_symbols)
    quantities[symbol_index] = quantity
    book = BookState(
        quantities,
        1000.0 - quantity * 100.0,
        dataset.close[1],
        1100.0,
        max_drawdown=0.12,
        _exact_quantities=None
        if exact_quantity is None
        else tuple(
            exact_quantity if i == symbol_index else "0"
            for i in range(dataset.n_symbols)
        ),
        as_of_index=1,
        as_of_dataset_id=dataset.dataset_id,
    )
    executor._update_margin(book)
    risk = PreTradeRisk(PreTradeRiskConfig(drawdown_start=0.10, drawdown_stop=0.20))
    return executor, book, risk


def order(
    executor,
    *,
    quantity=1.0,
    tif=TimeInForce.GTC,
    reduce_only=False,
    submit_index=0,
    eligible_index=1,
):
    intent = OrderIntent.create(
        dataset_id=executor.dataset.dataset_id,
        target_identity="previous-target",
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=OrderType.MARKET,
        time_in_force=tif,
        limit_price=None,
        stop_price=None,
        submit_index=submit_index,
        eligible_index=eligible_index,
        expiry_index=3 if tif is TimeInForce.DAY else None,
        submission_reference_price=100.0,
        decision_equity=1000.0,
        reduce_only=reduce_only,
    )
    return PendingOrder.from_intent(intent)


def capture(executor, book, risk, orders=None, *, index=1, symbol_index=0):
    module_name = "trade_rl.evaluation.allocation_snapshot"
    assert find_spec(module_name) is not None, "allocation snapshot producer is missing"
    return import_module(module_name).snapshot_allocation_account(
        executor,
        book,
        OrderBookState.empty() if orders is None else orders,
        account_id=f"independent-S{symbol_index}",
        pretrade_risk=risk,
        symbol_index=symbol_index,
        start_index=index,
    )


def test_snapshot_retains_current_financial_facts_and_source_identity():
    executor, book, risk = runtime()
    snapshot = capture(executor, book, risk)
    assert snapshot.dataset_id == executor.dataset.dataset_id
    assert snapshot.execution_policy_digest == executor.execution_policy_digest
    assert snapshot.account_id == "independent-S0"
    assert snapshot.symbol == "S0"
    assert snapshot.symbol_index == 0
    assert snapshot.decision_index == 1
    assert snapshot.observable_tradable is True
    facts = snapshot.book_facts
    assert facts["exact_quantities"] == ("5/2",)
    assert facts["cash"] == 750.0
    assert facts["equity"] == 1000.0
    assert facts["mark_prices"] == (100.0,)
    assert facts["contract_multipliers"] == (1.0,)
    assert facts["peak_value"] == 1100.0
    assert facts["current_drawdown"] == pytest.approx(1 - 1000 / 1100)
    assert facts["max_drawdown"] == 0.12
    assert facts["margin_used"] == 250.0
    assert facts["maintenance_margin"] == 0.01
    assert facts["maintenance_requirement"] == 2.5
    assert facts["margin_deficit"] == 0.0
    assert facts["insolvent"] is False
    assert facts["termination_reason"] is None
    assert not any(
        "reserved" in name or "free_cash" in name or "age" in name for name in facts
    )


def test_snapshot_is_detached_and_deeply_immutable_without_source_mutation():
    executor, book, risk = runtime()
    source_orders = OrderBookState.empty().add(order(executor))
    book.returns_history[:] = [0.01]
    before = deepcopy(asdict(book))
    orders_before = asdict(source_orders)
    rng_before = deepcopy(executor._rng.bit_generator.state)
    snapshot = capture(executor, book, risk, source_orders)
    for name, expected in before.items():
        actual = getattr(book, name)
        if isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected
    assert asdict(source_orders) == orders_before
    assert executor._rng.bit_generator.state == rng_before
    digest = snapshot.digest
    book.cash += 25.0
    book.quantities[0] = 3.0
    book.returns_history.append(0.02)
    assert snapshot.book_facts["cash"] == 750.0
    assert snapshot.book_facts["returns_history"] == (0.01,)
    assert snapshot.digest == digest
    with pytest.raises(TypeError):
        snapshot.book_facts["cash"] = 0.0
    with pytest.raises(TypeError):
        snapshot.active_orders[0]["intent"]["reduce_only"] = True
    with pytest.raises(FrozenInstanceError):
        snapshot.decision_index = 2


def test_read_does_not_refresh_original_exact_quantity_cache():
    executor, book, risk = runtime()
    book.quantities[0] = 3.0
    book.cash = 700.0
    book.margin_used, book.maintenance_requirement = 300.0, 3.0
    exact_before = book._exact_quantities
    snapshot = capture(executor, book, risk)
    assert snapshot.book_facts["exact_quantities"] == ("3",)
    assert book._exact_quantities == exact_before


def test_same_unsigned_gross_and_count_preserves_opposite_order_directions():
    executor, book, risk = runtime()
    buy = capture(executor, book, risk, OrderBookState.empty().add(order(executor)))
    sell = capture(
        executor, book, risk, OrderBookState.empty().add(order(executor, quantity=-1.0))
    )
    assert len(buy.active_orders) == len(sell.active_orders) == 1
    assert abs(float(buy.active_orders[0]["remaining_quantity"])) == abs(
        float(sell.active_orders[0]["remaining_quantity"])
    )
    assert buy.active_orders[0]["exact_remaining_quantity"] == "1"
    assert sell.active_orders[0]["exact_remaining_quantity"] == "-1"
    assert buy.digest != sell.digest


def test_exact_book_and_pending_fraction_survive_without_decimal_reconstruction():
    executor, book, risk = runtime(
        quantity=project_quantity(Fraction(1, 3)), exact_quantity="1/3"
    )
    pending = order(executor)
    pending = replace(
        pending,
        cumulative_filled_quantity=project_quantity(Fraction(1, 3)),
        remaining_quantity=project_quantity(Fraction(2, 3)),
        exact_cumulative_filled_quantity="1/3",
        status=OrderStatus.PARTIALLY_FILLED,
        last_processed_index=1,
        evidence_version=1,
    )
    snapshot = capture(executor, book, risk, OrderBookState.empty().add(pending))
    assert snapshot.book_facts["exact_quantities"] == ("1/3",)
    facts = snapshot.active_orders[0]
    assert facts["exact_requested_quantity"] == "1"
    assert facts["exact_cumulative_filled_quantity"] == "1/3"
    assert facts["exact_remaining_quantity"] == "2/3"
    assert facts["status"] == "partially_filled"
    assert facts["last_processed_index"] == 1
    assert facts["evidence_version"] == 1


@pytest.mark.parametrize("tif", tuple(TimeInForce))
def test_order_clock_tif_reduce_only_and_evidence_identity_are_preserved(tif):
    executor, book, risk = runtime()
    pending = order(executor, quantity=-1.0, tif=tif, reduce_only=True)
    snapshot = capture(executor, book, risk, OrderBookState.empty().add(pending))
    facts = snapshot.active_orders[0]
    assert facts["intent"]["order_id"] == pending.order_id
    assert facts["intent"]["time_in_force"] == tif.value
    assert facts["intent"]["reduce_only"] is True
    assert facts["intent"]["submit_index"] == 0
    assert facts["intent"]["eligible_index"] == 1
    assert facts["intent"]["expiry_index"] == pending.intent.expiry_index
    assert facts["status"] == "submitted"
    assert facts["last_processed_index"] is None


@pytest.mark.parametrize("change", ("unknown", "index", "dataset"))
def test_account_clock_mismatch_is_rejected_without_mutating_book(change):
    executor, book, risk = runtime()
    if change == "unknown":
        book.as_of_index = book.as_of_dataset_id = None
    elif change == "index":
        book.as_of_index = 0
    else:
        book.as_of_dataset_id = "a" * 64
    with pytest.raises(ValueError, match="processing clock"):
        capture(executor, book, risk)


def test_stale_margin_facts_are_rejected_instead_of_refreshed_on_original():
    executor, book, risk = runtime()
    book.margin_used = 0.0
    with pytest.raises(ValueError, match="margin"):
        capture(executor, book, risk)
    assert book.margin_used == 0.0


def test_unavailable_current_market_source_is_rejected():
    executor, book, risk = runtime(market(available=False))
    with pytest.raises(ValueError, match="available"):
        capture(executor, book, risk)


def test_future_market_suffix_does_not_change_current_snapshot():
    first = runtime(market(future_price=100.0, future_volume=1000.0))
    second = runtime(market(future_price=900.0, future_volume=0.0))
    left, right = capture(*first), capture(*second)
    assert left == right
    assert left.digest == right.digest


def test_snapshot_rejects_another_execution_policy_order():
    executor, book, risk = runtime()
    different = MarketExecutor(executor.dataset, ExecutionCostConfig.zero())
    orders = OrderBookState.empty().add(order(different))
    with pytest.raises(ValueError, match="execution policy"):
        capture(executor, book, risk, orders)


@pytest.mark.parametrize(
    "change", ({"time_in_force": "fok"}, {"submit_index": True}, {"order_id": "f" * 64})
)
def test_producer_rejects_orders_that_bypass_canonical_intent_validation(change):
    executor, book, risk = runtime()
    pending = order(executor)
    forged = PendingOrder.from_intent(replace(pending.intent, **change))
    with pytest.raises(ValueError):
        capture(executor, book, risk, OrderBookState.empty().add(forged))


@pytest.mark.parametrize(
    "changes",
    (
        {"as_of_index": True},
        {"mark_prices": ()},
        {"exact_quantities": ("1/3", "0")},
        {"cash": "not a financial value"},
        {"equity": 0.0},
        {"peak_value": 0.0},
    ),
)
def test_direct_dto_rejects_malformed_structural_account_facts(changes):
    snapshot = capture(*runtime())
    facts = dict(snapshot.book_facts)
    facts.update(changes)
    with pytest.raises(ValueError):
        replace(snapshot, book_facts=facts)


@pytest.mark.parametrize("location", ("order", "intent"))
def test_direct_dto_rejects_missing_nested_order_schema(location):
    executor, book, risk = runtime()
    snapshot = capture(
        executor, book, risk, OrderBookState.empty().add(order(executor))
    )
    facts = dict(snapshot.active_orders[0])
    if location == "order":
        del facts["status"]
    else:
        facts["intent"] = dict(facts["intent"])
        del facts["intent"]["time_in_force"]
    with pytest.raises(ValueError):
        replace(snapshot, active_orders=(facts,))


@pytest.mark.parametrize("change", ("terminal", "future", "exact", "missing_clock"))
def test_direct_dto_rejects_incompatible_active_order_facts(change):
    executor, book, risk = runtime()
    snapshot = capture(
        executor, book, risk, OrderBookState.empty().add(order(executor))
    )
    facts = dict(snapshot.active_orders[0])
    facts["intent"] = dict(facts["intent"])
    if change == "terminal":
        facts["status"] = "filled"
    elif change == "future":
        facts["intent"]["submit_index"] = 2
    elif change == "missing_clock":
        facts["intent"]["submit_index"] = None
    else:
        facts["exact_requested_quantity"] = facts["exact_remaining_quantity"] = "2"
    with pytest.raises(ValueError):
        replace(snapshot, active_orders=(facts,))


def test_snapshot_projects_only_selected_market_facts_and_preserves_global_slot():
    dataset = market(n_symbols=2)
    known = dataset.resolved_array("available_at").copy()
    known[1, 0] += np.timedelta64(1, "m")
    price = dataset.close.copy()
    price[1, 0] = 900.0
    dataset = replace(
        dataset,
        open=price,
        high=price,
        low=price,
        close=price,
        mark_price=price,
        available_at=known,
        information_available=None,
    )
    executor, book, risk = runtime(dataset, symbol_index=1)
    snapshot = capture(executor, book, risk, symbol_index=1)
    assert snapshot.symbol_index == 1
    assert snapshot.symbol == "S1"
    assert snapshot.book_facts["mark_prices"] == (100.0,)
    assert snapshot.book_facts["quantities"] == (2.5,)
    assert snapshot.book_facts["exact_quantities"] == ("5/2",)
    assert snapshot.book_facts["contract_multipliers"] == (1.0,)


def test_source_mutated_zero_peak_fails_through_canonical_account_validation():
    executor, book, risk = runtime()
    book.peak_value = 0.0
    with pytest.raises(ValueError, match="peak_value"):
        capture(executor, book, risk)
    assert book.peak_value == 0.0


def test_live_snapshot_preserves_negative_cash_without_inventing_free_balance():
    executor, book, risk = runtime(quantity=15.0)
    snapshot = capture(executor, book, risk)
    assert snapshot.book_facts["cash"] == -500.0
    assert snapshot.book_facts["equity"] == 1000.0


def test_available_nontradable_market_is_observed_as_nontradable():
    dataset = replace(market(), tradable=np.zeros((4, 1), dtype=np.bool_))
    snapshot = capture(*runtime(dataset))
    assert snapshot.observable_tradable is False
    assert snapshot.book_facts["mark_prices"] == (100.0,)


def test_order_last_transition_can_precede_current_account_processing_clock():
    executor, book, risk = runtime()
    book.as_of_index = 2
    pending = order(executor).mark_eligible(processing_index=1)
    snapshot = capture(
        executor, book, risk, OrderBookState.empty().add(pending), index=2
    )
    assert snapshot.active_orders[0]["last_processed_index"] == 1


def test_direct_dto_rejects_overreported_exact_book_quantity():
    snapshot = capture(*runtime())
    facts = dict(snapshot.book_facts)
    facts["quantities"] = (float(np.nextafter(2.5, np.inf)),)
    with pytest.raises(ValueError, match="quantity"):
        replace(snapshot, book_facts=facts)


def test_direct_dto_rejects_wrong_direction_remainders_despite_exact_sum():
    executor, book, risk = runtime()
    snapshot = capture(
        executor, book, risk, OrderBookState.empty().add(order(executor))
    )
    facts = dict(snapshot.active_orders[0])
    facts.update(
        cumulative_filled_quantity=2.0,
        remaining_quantity=-1.0,
        exact_cumulative_filled_quantity="2",
        exact_remaining_quantity="-1",
    )
    with pytest.raises(ValueError, match="quantity|quantities"):
        replace(snapshot, active_orders=(facts,))


@pytest.mark.parametrize("last", (0, 1))
def test_producer_rejects_native_reader_accepted_fill_before_eligibility(last):
    executor, book, risk = runtime()
    pending = replace(
        order(executor, submit_index=1, eligible_index=2),
        status=OrderStatus.PARTIALLY_FILLED,
        cumulative_filled_quantity=0.5,
        remaining_quantity=0.5,
        exact_cumulative_filled_quantity="1/2",
        last_processed_index=last,
        evidence_version=1,
    )
    assert PendingOrder.from_mapping(asdict(pending)).order_id == pending.order_id
    with pytest.raises(ValueError, match="submission|eligibility"):
        capture(executor, book, risk, OrderBookState.empty().add(pending))


@pytest.mark.parametrize(
    "status", (OrderStatus.SUBMITTED, OrderStatus.LATENCY_WAIT, OrderStatus.ELIGIBLE)
)
def test_producer_rejects_prefill_state_with_recorded_trigger(status):
    executor, book, risk = runtime()
    pending = replace(
        order(executor),
        status=status,
        trigger_index=1,
        last_processed_index=1,
        evidence_version=1,
    )
    assert PendingOrder.from_mapping(asdict(pending)).order_id == pending.order_id
    with pytest.raises(ValueError, match="progress"):
        capture(executor, book, risk, OrderBookState.empty().add(pending))


def test_producer_rejects_completed_triggered_active_order():
    executor, book, risk = runtime()
    pending = replace(
        order(executor),
        status=OrderStatus.TRIGGERED,
        cumulative_filled_quantity=1.0,
        remaining_quantity=0.0,
        exact_cumulative_filled_quantity="1",
        trigger_index=1,
        last_processed_index=1,
        evidence_version=1,
    )
    assert PendingOrder.from_mapping(asdict(pending)).order_id == pending.order_id
    with pytest.raises(ValueError, match="completion"):
        capture(executor, book, risk, OrderBookState.empty().add(pending))


@pytest.mark.parametrize(
    "status, last", ((OrderStatus.SUBMITTED, 1), (OrderStatus.LATENCY_WAIT, None))
)
def test_producer_rejects_native_reader_accepted_impossible_waiting_state(status, last):
    executor, book, risk = runtime()
    pending = replace(
        order(executor), status=status, last_processed_index=last, evidence_version=1
    )
    assert PendingOrder.from_mapping(asdict(pending)).order_id == pending.order_id
    with pytest.raises(ValueError, match="waiting status"):
        capture(executor, book, risk, OrderBookState.empty().add(pending))


def native_phase(executor, status):
    pending = order(executor)
    if status == OrderStatus.LATENCY_WAIT:
        return pending.mark_latency_wait(processing_index=0)
    if status in (OrderStatus.ELIGIBLE, OrderStatus.TRIGGERED):
        pending = pending.mark_eligible(processing_index=1)
    if status == OrderStatus.TRIGGERED:
        pending = pending.mark_triggered(processing_index=1)
    return pending


@pytest.mark.parametrize(
    "status",
    (
        OrderStatus.SUBMITTED,
        OrderStatus.LATENCY_WAIT,
        OrderStatus.ELIGIBLE,
        OrderStatus.TRIGGERED,
    ),
)
def test_producer_rejects_positive_notional_without_any_filled_quantity(status):
    executor, book, risk = runtime()
    pending = replace(native_phase(executor, status), cumulative_filled_notional=1.0)
    assert PendingOrder.from_mapping(asdict(pending)).order_id == pending.order_id
    with pytest.raises(ValueError, match="notional"):
        capture(executor, book, risk, OrderBookState.empty().add(pending))


@pytest.mark.parametrize(
    "status", (OrderStatus.PARTIALLY_FILLED, OrderStatus.TRIGGERED)
)
def test_producer_rejects_active_remainder_that_native_fill_has_already_completed(
    status,
):
    executor, book, risk = runtime()
    cumulative = Fraction(999999999999999, 1000000000000000)
    original = native_phase(executor, OrderStatus.TRIGGERED)
    completed = original.apply_fill(
        quantity=float(cumulative), notional=1.0, processing_index=1
    )
    assert completed.status == OrderStatus.FILLED
    assert completed.remaining_quantity == 0.0
    pending = replace(
        original,
        status=status,
        cumulative_filled_quantity=project_quantity(cumulative),
        remaining_quantity=project_quantity(1 - cumulative),
        exact_cumulative_filled_quantity=str(cumulative),
        cumulative_filled_notional=1.0,
    )
    assert PendingOrder.from_mapping(asdict(pending)).order_id == pending.order_id
    with pytest.raises(ValueError, match="completion"):
        capture(executor, book, risk, OrderBookState.empty().add(pending))


def test_public_partial_fill_with_zero_notional_remains_observable():
    executor, book, risk = runtime()
    pending = native_phase(executor, OrderStatus.TRIGGERED).apply_fill(
        quantity=0.5, notional=0.0, processing_index=1
    )
    assert pending.status == OrderStatus.PARTIALLY_FILLED
    item = capture(executor, book, risk, OrderBookState.empty().add(pending))
    assert item.active_orders[0]["exact_cumulative_filled_quantity"] == "1/2"
    assert item.active_orders[0]["cumulative_filled_notional"] == 0.0
