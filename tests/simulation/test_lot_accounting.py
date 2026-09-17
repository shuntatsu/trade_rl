import json
import math
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from tests.simulation.test_stateful_execution import _intent
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.simulation import BookState, ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState, OrderStatus, PendingOrder
from trade_rl.simulation.quantities import (
    exact_quantity,
    project_quantity,
    quantize_quantity,
)
from trade_rl.simulation.targets.execution import execute_target_statefully


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize(
    ("lot", "lot_counts"),
    [
        (0.001, [1] * 100 + [-1] * 99),
        (0.001, [11] + [-1] * 10),
        (0.1, [3, -1, -1]),
        (1e-6, [5, -1, -1, -1, -1]),
    ],
)
def test_valid_lot_fills_remain_on_grid_and_close_without_residual(
    lot: float, lot_counts: list[int], direction: int
) -> None:
    n_bars = len(lot_counts) + 2
    shape = (n_bars, 1)
    prices = np.full(shape, 100.0 / lot)
    dataset = MarketDataset(
        dataset_id="d" * 64,
        symbols=("SYNTHETIC",),
        timestamps=np.datetime64("2000-01-01T00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
        feature_available=np.ones((n_bars, 1, 1), dtype=bool),
        feature_names=("unused",),
        global_feature_names=("unused",),
        periods_per_year=8760,
        lot_size=np.full(shape, lot),
        minimum_notional=np.full(shape, 50.0),
    )
    executor = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), processing_bar_volume_capacity=False),
    )
    book = BookState.zero(1, 10_000.0, prices[0])
    orders = OrderBookState.empty()
    for index, count in enumerate(lot_counts):
        intent = _intent(
            executor,
            float(direction * count * Decimal(str(lot))),
            submit_index=index,
            eligible_index=index + 1,
            target_identity=f"lot-fill-{index}",
        )
        result = executor.execute_orders(
            book, orders, (intent,), start_index=index, bars=1
        )
        book, orders = result.book, result.order_book
        assert result.fill_count == 1
    expected = float(direction * Decimal(str(lot)))
    assert book.quantities[0] == expected
    result = execute_target_statefully(
        executor,
        book,
        orders,
        np.array([0.0]),
        start_index=len(lot_counts),
        bars=1,
        target_identity="terminal-flat",
    )
    assert result.book.quantities[0] == 0.0
    assert result.book.cash == pytest.approx(10_000.0, abs=1e-8)
    assert result.book.total_cost == 0.0
    assert result.fill_count == 1


def test_genuine_sub_lot_remainder_survives_fills_clone_and_split() -> None:
    original = math.nextafter(1e-6, 0.0)
    book = BookState(np.array([original]), 1000.0, np.array([100.0]), 1001.0)

    def fill(current, count):
        current.execute_fill(
            symbol_index=0,
            quantity=count * 1e-6,
            lot_count=count,
            lot_size=1e-6,
            fill_prices=np.array([100.0]),
            cost_amount=0.0,
            turnover=0.0,
        )

    fill(book, 1)
    assert book.quantities[0] < 2e-6
    book = book.clone()
    book.apply_split(np.array([2.0]))
    book.apply_split(np.array([0.5]))
    fill(book, -1)
    assert book.quantities[0] == original
    assert quantize_quantity(book.quantities[0], 1e-6) == (0.0, 0)
    book.settle_positions(
        mask=np.array([True]), prices=np.array([100.0]), recovery=np.array([1.0])
    )
    fill(book, 1)
    fill(book, -1)
    assert book.quantities[0] == 0.0


def test_absolute_target_rebase_pays_for_exact_prior_remainder() -> None:
    original = math.nextafter(1e-6, 0.0)
    book = BookState(np.array([original]), 1e6, np.array([1e12]), 2e6)
    book.execute_fill(
        symbol_index=0,
        quantity=1e-6,
        lot_count=1,
        lot_size=1e-6,
        fill_prices=np.array([1e12]),
        cost_amount=0.0,
        turnover=0.0,
    )
    target = book.quantities.copy()
    exact_before = exact_quantity(original) + Fraction(1, 1_000_000)
    cash_before = book.cash
    book.execute(
        fill_prices=np.array([1e12]),
        target_quantities=target,
        cost_amount=0.0,
        turnover=0.0,
    )
    expected_proceeds = float(
        (exact_before - exact_quantity(float(target[0]))) * 10**12
    )
    assert expected_proceeds > 0.0
    assert book.cash - cash_before == pytest.approx(expected_proceeds, rel=1e-12, abs=0)


def test_partial_order_exact_state_survives_json_and_finishes_last_lot() -> None:
    from tests.simulation.test_orders import _intent as pending_intent

    order = PendingOrder.from_intent(pending_intent(requested_quantity=0.1))
    for index in range(99):
        order = order.apply_fill(
            quantity=0.001,
            lot_size=0.001,
            lot_count=1,
            notional=100.0,
            processing_index=5 + index,
        )
        order = PendingOrder.from_mapping(json.loads(canonical_json_bytes(order)))
    assert order.remaining_quantity == 0.001
    assert order.cumulative_filled_quantity == 0.099
    order = order.apply_fill(
        quantity=0.001,
        lot_size=0.001,
        lot_count=1,
        notional=100.0,
        processing_index=104,
    )
    assert order.status is OrderStatus.FILLED
    assert order.remaining_quantity == 0.0


def test_exact_quantity_payload_rejects_inconsistent_or_noncanonical_state() -> None:
    from tests.simulation.test_orders import _intent as pending_intent

    order = PendingOrder.from_intent(pending_intent(requested_quantity=0.1))
    payload = json.loads(canonical_json_bytes(order))
    for corrupt in ("1/10", "00", None, True):
        with pytest.raises(ValueError):
            PendingOrder.from_mapping(
                {**payload, "exact_cumulative_filled_quantity": corrupt}
            )
    del payload["exact_cumulative_filled_quantity"]
    assert PendingOrder.from_mapping(payload).remaining_quantity == 0.1


def test_exact_projection_never_promotes_an_unquantized_balance() -> None:
    below_two_lots = exact_quantity(math.nextafter(1e-6, 0.0)) + Fraction(1, 1_000_000)
    assert exact_quantity(project_quantity(below_two_lots)) <= below_two_lots
    assert quantize_quantity(project_quantity(below_two_lots), 1e-6)[1] == 1


def test_settlement_pays_for_the_exact_remainder_before_clearing_it() -> None:
    original = math.nextafter(1e-6, 0.0)
    book = BookState(np.array([original]), 1e6, np.array([1e12]), 2e6)
    book.execute_fill(
        symbol_index=0,
        quantity=1e-6,
        lot_count=1,
        lot_size=1e-6,
        fill_prices=np.array([1e12]),
        cost_amount=0.0,
        turnover=0.0,
    )
    expected = float((exact_quantity(original) + Fraction(1, 1_000_000)) * 10**12)
    proceeds = book.settle_positions(
        mask=np.array([True]), prices=np.array([1e12]), recovery=np.array([1.0])
    )
    assert proceeds == expected
    assert book.quantities[0] == 0


@pytest.mark.parametrize("direction", [1, -1])
def test_signed_fill_cash_matches_independent_decimal_ledger(direction: int) -> None:
    multiplier = Decimal("0.5")
    book = BookState.zero(1, 10_000.0, np.array([12345.67]), np.array([0.5]))
    cash = Decimal("10000")
    for count, price, fee in [
        (11, "12345.67", "0.1"),
        (-2, "11999.99", "0.23"),
        (-9, "12999.123", "0.12"),
    ]:
        signed = direction * count
        exact = Decimal(signed) * Decimal("0.001")
        cash -= exact * Decimal(price) * multiplier + Decimal(fee)
        book.execute_fill(
            symbol_index=0,
            quantity=float(exact),
            lot_count=signed,
            lot_size=0.001,
            fill_prices=np.array([float(price)]),
            cost_amount=float(fee),
            turnover=0.0,
        )
        book = book.clone()
        assert book.cash == pytest.approx(float(cash), abs=1e-9, rel=0)
    assert book.quantities[0] == 0.0
    assert book.total_cost == pytest.approx(0.45)


@pytest.mark.parametrize(
    ("price", "multiplier"),
    [(553.0793741519649, 1.0), (999.99, 1.0), (999.99, 0.1), (553.0793741519649, 0.5)],
)
def test_capacity_boundary_uses_the_same_signed_notional_as_book_cash(
    price: float, multiplier: float
) -> None:
    from tests.simulation.test_liquidity import _request
    from trade_rl.simulation.liquidity import allocate_symbol_capacity

    quantity = 53.571
    capacity = quantity * price * multiplier
    allocations, evidence = allocate_symbol_capacity(
        (_request("a", quantity, price=price),),
        processing_volume=1000.0,
        processing_market_notional=capacity,
        price=price,
        contract_multiplier=multiplier,
        participation_limit=1.0,
        lot_size=0.001,
        minimum_notional=0.0,
    )
    fill = allocations[0]
    assert fill.filled_quantity == quantity
    assert fill.filled_notional <= capacity
    assert evidence.remaining_capacity_notional >= 0.0
    book = BookState.zero(1, 100_000.0, np.array([price]), np.array([multiplier]))
    book.execute_fill(
        symbol_index=0,
        quantity=fill.filled_quantity,
        lot_size=fill.lot_size,
        lot_count=fill.filled_lot_count,
        fill_prices=np.array([price]),
        cost_amount=0.0,
        turnover=0.0,
    )
    assert book.cash == 100_000.0 - fill.filled_notional


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("one_ulp_below", [False, True])
@pytest.mark.parametrize(
    ("quantity", "price", "multiplier", "lot"),
    [
        (57.671, 999.99, 0.1, 0.001),
        (0.3, 3.3, 0.1, 0.1),
        (0.001, 73118.64049422563, 10.0, 0.001),
    ],
)
def test_partial_capacity_keeps_every_affordable_lot(
    direction: int,
    one_ulp_below: bool,
    quantity: float,
    price: float,
    multiplier: float,
    lot: float,
) -> None:
    from tests.simulation.test_liquidity import _request
    from trade_rl.simulation.liquidity import allocate_symbol_capacity

    capacity = quantity * price * multiplier
    expected = quantity
    if one_ulp_below:
        capacity = math.nextafter(capacity, 0.0)
        expected = float(Decimal(str(quantity)) - Decimal(str(lot)))
    allocations, evidence = allocate_symbol_capacity(
        (_request("a", direction * quantity * 2, price=price),),
        processing_volume=1000.0,
        processing_market_notional=capacity,
        price=price,
        contract_multiplier=multiplier,
        participation_limit=1.0,
        lot_size=lot,
        minimum_notional=0.0,
    )
    assert allocations[0].filled_quantity == direction * expected
    assert allocations[0].filled_notional <= capacity
    assert evidence.remaining_capacity_notional >= 0.0
