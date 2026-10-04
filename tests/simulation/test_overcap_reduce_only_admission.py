from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.admission import AdmissionDecision, OrderAdmissionPolicy
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)


def _executor(
    symbols: int = 1,
    *,
    volume: float = 1_000.0,
    lot_size: float = 0.0,
    fee_rate: float = 0.001,
) -> MarketExecutor:
    bars = 6
    shape = (bars, symbols)
    prices = np.full(shape, 100.0)
    market = MarketDataset(
        dataset_id="d" * 64,
        symbols=tuple(f"S{i}" for i in range(symbols)),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(bars) * np.timedelta64(1, "h"),
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((bars, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices + 10.0,
        low=prices - 10.0,
        close=prices.copy(),
        volume=np.full(shape, volume),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((*shape, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )
    return MarketExecutor(
        market,
        replace(
            ExecutionCostConfig.zero(),
            max_participation_rate=1.0,
            lot_size=lot_size,
            fee_rate=fee_rate,
            max_leverage=1.0,
            maintenance_margin_rate=0.01,
        ),
    )


def _book(positions: tuple[float, ...], equity: float) -> BookState:
    quantities = np.array(positions)
    return BookState(
        quantities=quantities,
        cash=equity - float(quantities.sum()) * 100.0,
        mark_prices=np.full(len(positions), 100.0),
        peak_value=equity,
    )


def _intent(
    executor: MarketExecutor,
    quantity: float,
    *,
    symbol: int = 0,
    eligible_index: int = 1,
    reduce_only: bool = True,
) -> OrderIntent:
    return OrderIntent.create(
        dataset_id=executor.dataset.dataset_id,
        target_identity=f"close-{symbol}-{eligible_index}-{quantity}",
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=symbol,
        requested_quantity=quantity,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=eligible_index,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=300.0,
        reduce_only=reduce_only,
    )


def _decision(
    executor: MarketExecutor,
    book: BookState,
    intent: OrderIntent,
    **overrides: object,
) -> AdmissionDecision:
    arguments: dict[str, object] = {
        "book": book,
        "processing_index": 1,
        "asset_active": True,
        "tradable": True,
        "buy_allowed": True,
        "sell_allowed": True,
        "borrow_available": True,
        "tick_size": 0.0,
        "lot_size": 0.0,
        "minimum_notional": 0.0,
        "reference_prices": np.full(book.quantities.size, 100.0),
    }
    arguments.update(overrides)
    policy = OrderAdmissionPolicy(
        expected_dataset_id=executor.dataset.dataset_id,
        expected_execution_policy_digest=executor.execution_policy_digest,
        allow_short=True,
        max_leverage=1.0,
    )
    return policy.evaluate(intent, **arguments)


@pytest.mark.parametrize("direction", [-1, 1])
def test_reduce_only_admits_close_while_other_leg_remains_over_cap(
    direction: int,
) -> None:
    executor = _executor(2)
    book = _book((direction * 4.0, direction * 4.0), 300.0)

    decision = _decision(executor, book, _intent(executor, -direction * 4.0))

    assert 400.0 > book.portfolio_value * executor.cost.max_leverage
    assert decision.accepted
    assert decision.reason is None
    assert decision.admitted_quantity == -direction * 4.0
    assert decision.admitted_notional == 400.0
    assert book.exact_quantities == (Fraction(direction * 4),) * 2


@pytest.mark.parametrize("direction", [-1, 1])
def test_two_over_cap_reduce_only_closes_fill_to_exact_zero(direction: int) -> None:
    executor = _executor(2)
    book = _book((direction * 4.0, direction * 4.0), 300.0)
    closes = tuple(
        _intent(executor, -direction * 4.0, symbol=i, eligible_index=i)
        for i in range(2)
    )

    result = executor.execute_orders(
        book, OrderBookState.empty(), closes, start_index=0, bars=1
    )

    assert result.book.exact_quantities == (Fraction(0), Fraction(0))
    assert result.fill_count == 2
    assert result.rejected_count == 0
    assert not result.order_book.active_orders
    assert all(
        o.status is OrderStatus.FILLED for o in result.order_book.terminal_orders
    )
    assert result.interval_cost == pytest.approx(0.8)
    assert book.exact_quantities == (Fraction(direction * 4),) * 2
    assert book.portfolio_value == 300.0


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("second_size", [4.0, 6.0])
def test_over_cap_closes_reserve_inventory_before_filling(
    direction: int, second_size: float
) -> None:
    executor = _executor()
    first = _intent(executor, -direction * 6.0, eligible_index=0)
    second = _intent(executor, -direction * second_size, eligible_index=1)

    result = executor.execute_orders(
        _book((direction * 10.0,), 500.0),
        OrderBookState.empty(),
        (first, second),
        start_index=0,
        bars=1,
    )

    terminal = {order.order_id: order for order in result.order_book.terminal_orders}
    assert terminal[first.order_id].status is OrderStatus.FILLED
    if second_size == 4.0:
        assert terminal[second.order_id].status is OrderStatus.FILLED
        assert result.book.exact_quantities == (Fraction(0),)
        assert result.interval_cost == pytest.approx(1.0)
        assert result.fill_count == 2
    else:
        assert terminal[second.order_id].status is OrderStatus.REJECTED
        assert result.book.exact_quantities == (Fraction(direction * 4),)
        assert result.interval_cost == pytest.approx(0.6)
        assert result.fill_count == 1
        assert not any(
            e.filled_quantity
            for e in result.order_events
            if e.order_id == second.order_id
        )
    assert sum(e.consumed_capacity_notional for e in result.capacity_evidence) == (
        1_000.0 if second_size == 4.0 else 600.0
    )


@pytest.mark.parametrize("second_size", [0.3, 0.30000000000000004])
def test_over_cap_decimal_reservations_do_not_allow_an_exact_overdraw(
    second_size: float,
) -> None:
    executor = _executor(fee_rate=0.0)
    first = _intent(executor, -0.1, eligible_index=0)
    second = _intent(executor, -second_size, eligible_index=1)

    result = executor.execute_orders(
        _book((0.4,), 35.0),
        OrderBookState.empty(),
        (first, second),
        start_index=0,
        bars=1,
    )

    terminal = {order.order_id: order for order in result.order_book.terminal_orders}
    assert terminal[first.order_id].status is OrderStatus.FILLED
    if second_size == 0.3:
        assert terminal[second.order_id].status is OrderStatus.FILLED
        assert result.book.exact_quantities == (Fraction(0),)
    else:
        assert terminal[second.order_id].status is OrderStatus.REJECTED
        assert result.book.exact_quantities == (Fraction("0.3"),)
        assert result.fill_count == 1


def test_over_cap_no_lot_closes_reserve_the_full_exact_rational_inventory() -> None:
    executor = _executor()
    # This reporting value lies inside 7/3; the canonical rational remains
    # authoritative across the clone made by the public executor call.
    reporting_quantity = 2.333333333333333
    book = BookState(
        quantities=np.array([reporting_quantity]),
        cash=100.0 - reporting_quantity * 100.0,
        mark_prices=np.array([100.0]),
        peak_value=100.0,
        _exact_quantities=("7/3",),
    )
    first = _intent(executor, -1.0, eligible_index=0)
    # 4/3 cannot be expressed as a finite decimal; its conservative float
    # request still denotes a full close of the remaining exact inventory.
    second = _intent(executor, -1.3333333333333333, eligible_index=1)

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (first, second),
        start_index=0,
        bars=1,
    )

    assert result.book.exact_quantities == (Fraction(0),)
    assert result.fill_count == 2
    assert result.rejected_count == 0
    assert not result.order_book.active_orders
    assert all(
        o.status is OrderStatus.FILLED for o in result.order_book.terminal_orders
    )
    assert result.interval_cost == pytest.approx(float(Fraction(7, 3)) * 0.1)
    assert sum(e.consumed_capacity_notional for e in result.capacity_evidence) == (
        pytest.approx(float(Fraction(7, 3)) * 100.0)
    )
    assert book.exact_quantities == (Fraction(7, 3),)
    assert book.portfolio_value == 100.0


def test_unfilled_closes_do_not_authorize_an_opening_from_an_over_cap_book() -> None:
    executor = _executor(2, volume=0.0)
    closes = (
        _intent(executor, 4.0, symbol=0, eligible_index=0),
        _intent(executor, 4.0, symbol=1, eligible_index=1),
    )
    opening = _intent(executor, 1.0, eligible_index=2, reduce_only=False)

    result = executor.execute_orders(
        _book((-4.0, -4.0), 300.0),
        OrderBookState.empty(),
        (*closes, opening),
        start_index=1,
        bars=1,
    )

    assert {o.order_id for o in result.order_book.active_orders} == {
        close.order_id for close in closes
    }
    rejected = next(
        o for o in result.order_book.terminal_orders if o.order_id == opening.order_id
    )
    assert rejected.status is OrderStatus.REJECTED
    assert rejected.terminal_reason == "pretrade_leverage_exceeded"
    assert result.book.exact_quantities == (Fraction(-4), Fraction(-4))
    assert result.fill_count == 0
    assert result.interval_cost == 0.0
    assert sum(e.consumed_capacity_notional for e in result.capacity_evidence) == 0.0


def test_under_cap_competing_closes_retain_fill_time_clipping() -> None:
    executor = _executor()
    closes = tuple(_intent(executor, -1.0, eligible_index=i) for i in range(2))

    result = executor.execute_orders(
        _book((1.0,), 1_000.0),
        OrderBookState.empty(),
        closes,
        start_index=0,
        bars=1,
    )

    terminal = {order.order_id: order for order in result.order_book.terminal_orders}
    assert terminal[closes[0].order_id].status is OrderStatus.FILLED
    assert terminal[closes[1].order_id].status is OrderStatus.EXPIRED
    assert terminal[closes[1].order_id].terminal_reason == "reduce_only_exhausted"
    assert result.rejected_count == 0
    assert result.fill_count == 1
    assert result.book.exact_quantities == (Fraction(0),)
    assert result.interval_cost == pytest.approx(0.1)
    assert result.capacity_evidence[0].consumed_capacity_notional == 100.0


def test_over_cap_partial_close_survives_serialized_restart() -> None:
    executor = _executor(2, volume=1.0, lot_size=0.1)
    close = _intent(executor, -4.0)
    first = executor.execute_orders(
        _book((4.0, 4.0), 300.0),
        OrderBookState.empty(),
        (close,),
        start_index=0,
        bars=1,
    )

    assert first.book.exact_quantities == (Fraction(3), Fraction(4))
    assert first.interval_cost == pytest.approx(0.1)
    assert first.capacity_evidence[0].consumed_capacity_notional == 100.0
    pending = PendingOrder.from_mapping(
        json.loads(canonical_json_bytes(first.order_book.active_orders[0]))
    )
    assert pending.remaining_quantity == -3.0
    assert pending.intent.reduce_only

    resumed = executor.execute_orders(
        first.book.clone(),
        OrderBookState((pending,), ()),
        (),
        start_index=1,
        bars=3,
    )

    assert resumed.book.exact_quantities == (Fraction(0), Fraction(4))
    assert 400.0 > resumed.book.portfolio_value * executor.cost.max_leverage
    assert resumed.interval_cost == pytest.approx(0.3)
    assert resumed.fill_count == 3
    assert not resumed.order_book.active_orders
    assert resumed.order_book.terminal_orders[-1].status is OrderStatus.FILLED
    assert sum(e.consumed_capacity_notional for e in resumed.capacity_evidence) == 300.0


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"minimum_notional": 500.0}, "below_minimum_notional"),
        ({"sell_allowed": False}, "sell_disabled"),
        ({"lot_size": 5.0}, "zero_quantity_after_rounding"),
        ({"asset_active": False}, "inactive_asset"),
        ({"tradable": False}, "non_tradable_market"),
        ({"processing_index": 0}, "not_eligible"),
    ],
)
def test_over_cap_reduce_only_retains_venue_and_latency_constraints(
    overrides: dict[str, object], reason: str
) -> None:
    executor = _executor(2)
    decision = _decision(
        executor,
        _book((4.0, 4.0), 300.0),
        _intent(executor, -4.0),
        **overrides,
    )

    assert not decision.accepted
    assert decision.reason == reason
    assert decision.admitted_quantity == decision.admitted_notional == 0.0


def test_over_cap_reduce_only_retains_policy_identity_validation() -> None:
    executor = _executor(2)
    intent = replace(_intent(executor, -4.0), execution_policy_digest="e" * 64)

    decision = _decision(executor, _book((4.0, 4.0), 300.0), intent)

    assert not decision.accepted
    assert decision.reason == "identity_mismatch"


@pytest.mark.parametrize(
    "equity,declared_insolvent",
    [(0.0, False), (-100.0, False), (100.0, True)],
)
def test_reduce_only_cannot_bypass_bad_equity_or_insolvency(
    equity: float, declared_insolvent: bool
) -> None:
    executor = _executor()
    book = BookState(
        quantities=np.array([4.0]),
        cash=equity - 400.0,
        mark_prices=np.array([100.0]),
        peak_value=100.0,
        insolvent=declared_insolvent,
    )

    decision = _decision(executor, book, _intent(executor, -4.0))

    assert book.portfolio_value == equity
    assert book.insolvent
    assert not decision.accepted
    assert decision.reason == "pretrade_leverage_exceeded"
    assert decision.admitted_quantity == decision.admitted_notional == 0.0
    assert book.exact_quantities == (Fraction(4),)
    assert book.cash == equity - 400.0


@pytest.mark.parametrize("quantity", [-2.0, 1.0])
def test_ordinary_orders_cannot_leave_an_over_cap_book_over_cap(
    quantity: float,
) -> None:
    executor = _executor(2)
    decision = _decision(
        executor,
        _book((4.0, 4.0), 300.0),
        _intent(executor, quantity, reduce_only=False),
    )

    assert not decision.accepted
    assert decision.reason == "pretrade_leverage_exceeded"
