from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tests.simulation.test_overcap_reduce_only_admission import (
    _book,
    _decision,
    _executor,
    _intent,
)
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.simulation.orders.reconciliation import (
    ReconciliationResult,
    reconcile_target,
)


def _reconcile(
    executor: MarketExecutor,
    book: BookState,
    target_weight: float,
    *,
    reference_price: float = 100.0,
    decision_equity: float | None = None,
    reduce_only_symbols: tuple[int, ...] | None = None,
    order_book: OrderBookState | None = None,
    order_type: OrderType = OrderType.MARKET,
) -> ReconciliationResult:
    return reconcile_target(
        dataset_id=executor.dataset.dataset_id,
        target_identity="risk-bounded-target",
        execution_policy_digest=executor.execution_policy_digest,
        target_weights=np.array([target_weight]),
        book=book,
        order_book=OrderBookState.empty() if order_book is None else order_book,
        reference_prices=np.array([reference_price]),
        decision_equity=book.portfolio_value
        if decision_equity is None
        else decision_equity,
        submit_index=0,
        latency_bars=1,
        order_type=order_type,
        time_in_force=TimeInForce.GTC,
        expiry_index=None,
        limit_offset_rate=0.01,
        maximum_gross=executor.cost.max_leverage,
        reduce_only_symbols=reduce_only_symbols,
    )


@pytest.mark.parametrize("direction", [-1, 1])
def test_legacy_over_cap_nonzero_trim_creates_an_executable_reduce_only_order(
    direction: int,
) -> None:
    executor = _executor()
    book = _book((direction * 10.0,), 500.0)

    reconciled = _reconcile(executor, book, direction * 0.5)

    intent = reconciled.new_intents[0]
    assert intent.reduce_only
    assert intent.requested_quantity == -direction * 7.5
    result = executor.execute_orders(
        book,
        reconciled.order_book,
        reconciled.new_intents,
        start_index=0,
        bars=1,
    )
    assert result.book.exact_quantities == (Fraction(direction * 5, 2),)
    assert result.fill_count == 1
    assert result.rejected_count == 0


@pytest.mark.parametrize(
    "quantity,equity,weight", [(1.0, 1_000.0, 0.05), (10.0, 1_000.0, 0.5)]
)
def test_legacy_under_or_at_cap_nonzero_trim_remains_ordinary(
    quantity: float, equity: float, weight: float
) -> None:
    result = _reconcile(_executor(), _book((quantity,), equity), weight)

    assert not result.new_intents[0].reduce_only
    assert abs(result.new_intents[0].requested_quantity) < quantity


def test_legacy_under_cap_full_flat_remains_reduce_only() -> None:
    result = _reconcile(_executor(), _book((1.0,), 1_000.0), 0.0)

    assert result.new_intents[0].reduce_only
    assert result.new_intents[0].requested_quantity == -1.0


@pytest.mark.parametrize(
    "quantity,expected_reduce_only", [(-10.0, True), (10.0, False)]
)
def test_legacy_trim_classifies_actual_book_at_reference_prices(
    quantity: float, expected_reduce_only: bool
) -> None:
    direction = -1 if quantity < 0.0 else 1
    result = _reconcile(
        _executor(),
        _book((quantity,), 1_000.0),
        direction * 0.5,
        reference_price=120.0,
    )

    # The same stale mark equity (1000) becomes 800 for the short and 1200
    # for the long at this reference price. Only the short is above its cap.
    assert result.new_intents[0].reduce_only is expected_reduce_only


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize(
    "quantity,actual_equity,sizing_equity,expected_reduce_only",
    [(4.0, 1_000.0, 200.0, False), (10.0, 500.0, 2_000.0, True)],
)
def test_legacy_trim_classifies_actual_cap_independently_of_the_sizing_budget(
    direction: int,
    quantity: float,
    actual_equity: float,
    sizing_equity: float,
    expected_reduce_only: bool,
) -> None:
    executor = _executor()
    book = _book((direction * quantity,), actual_equity)

    result = _reconcile(
        executor,
        book,
        direction * 0.2,
        decision_equity=sizing_equity,
    )

    intent = result.new_intents[0]
    assert intent.reduce_only is expected_reduce_only
    assert intent.decision_equity == sizing_equity
    assert result.desired_quantities[0] == direction * 0.2 * sizing_equity / 100.0
    assert book.portfolio_value == actual_equity
    # A legacy reduce-only flag does not waive a venue notional floor.
    decision = _decision(executor, book, intent, minimum_notional=1_000.0)
    assert not decision.accepted
    assert decision.reason == "below_minimum_notional"


@pytest.mark.parametrize("direction", [-1, 1])
def test_over_cap_reversal_is_ordinary_and_cannot_bypass_admission(
    direction: int,
) -> None:
    executor = _executor()
    book = _book((direction * 10.0,), 500.0)
    reconciled = _reconcile(executor, book, -direction * 0.5)

    assert not reconciled.new_intents[0].reduce_only
    result = executor.execute_orders(
        book,
        reconciled.order_book,
        reconciled.new_intents,
        start_index=0,
        bars=1,
    )

    assert result.fill_count == 0
    assert result.interval_cost == 0.0
    assert result.book.exact_quantities == (Fraction(direction * 10),)
    assert result.order_book.terminal_orders[0].status is OrderStatus.REJECTED
    assert (
        result.order_book.terminal_orders[0].terminal_reason
        == "pretrade_leverage_exceeded"
    )


@pytest.mark.parametrize("symbols,expected", [((), False), ((0,), True)])
@pytest.mark.parametrize("equity", [500.0, 1_000.0])
def test_explicit_profile_symbols_own_trim_flag_even_when_actual_book_is_over_cap(
    symbols: tuple[int, ...], expected: bool, equity: float
) -> None:
    result = _reconcile(
        _executor(),
        _book((10.0,), equity),
        0.5,
        reduce_only_symbols=symbols,
    )

    assert result.new_intents[0].reduce_only is expected


@pytest.mark.parametrize("order_type", [OrderType.LIMIT, OrderType.STOP_MARKET])
def test_legacy_over_cap_nonmarket_trim_does_not_create_unsupported_reduce_only(
    order_type: OrderType,
) -> None:
    result = _reconcile(_executor(), _book((10.0,), 500.0), 0.5, order_type=order_type)

    assert not result.new_intents[0].reduce_only
    assert result.new_intents[0].order_type is order_type


def test_over_cap_trim_delta_uses_exact_rational_inventory() -> None:
    reporting_quantity = 2.333333333333333
    book = BookState(
        quantities=np.array([reporting_quantity]),
        cash=100.0 - reporting_quantity * 100.0,
        mark_prices=np.array([100.0]),
        peak_value=100.0,
        _exact_quantities=("7/3",),
    )

    result = _reconcile(_executor(), book, 0.5)

    intent = result.new_intents[0]
    assert intent.reduce_only
    assert intent.requested_quantity == -1.8333333333333333
    exact_request = abs(Fraction(str(intent.requested_quantity)))
    assert exact_request < Fraction(11, 6)
    assert Fraction(
        str(np.nextafter(abs(intent.requested_quantity), np.inf))
    ) > Fraction(11, 6)
    assert book.exact_quantities == (Fraction(7, 3),)


@pytest.mark.parametrize("existing_reduce_only", [False, True])
def test_over_cap_trim_reuses_only_a_matching_reduce_only_residual(
    existing_reduce_only: bool,
) -> None:
    executor = _executor()
    active = PendingOrder.from_intent(
        _intent(executor, -7.5, reduce_only=existing_reduce_only)
    )
    result = _reconcile(
        executor,
        _book((10.0,), 500.0),
        0.5,
        order_book=OrderBookState((active,), ()),
    )

    if existing_reduce_only:
        assert result.new_intents == result.cancelled_orders == ()
        assert result.order_book.active_orders == (active,)
    else:
        assert len(result.cancelled_orders) == 1
        assert result.cancelled_orders[0].order_id == active.order_id
        assert result.cancelled_orders[0].status is OrderStatus.CANCELLED
        assert result.cancelled_orders[0].terminal_reason == "superseded"
        assert result.new_intents[0].reduce_only
        assert result.new_intents[0].requested_quantity == -7.5
        assert result.new_intents[0].replaced_order_id == active.order_id
