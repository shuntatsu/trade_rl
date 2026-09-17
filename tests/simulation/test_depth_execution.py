from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.simulation import BookState
from trade_rl.simulation.depth import DepthOrderRules, execute_depth_order

NOW = datetime(2026, 9, 18, tzinfo=UTC)


def execute(book, **changes):
    arguments = dict(
        book=book,
        symbol_index=0,
        lot_count=2,
        rules=DepthOrderRules(
            lot_size=1,
            minimum_quantity=1,
            maximum_quantity=100,
            minimum_notional=0,
            maximum_notional=None,
        ),
        bids=((99.0, 10.0), (98.0, 20.0)),
        asks=((101.0, 10.0), (102.0, 20.0)),
        decision_at=NOW,
        quote_at=NOW + timedelta(seconds=1),
        execution_at=NOW + timedelta(seconds=2),
        fee_bps=10.0,
        adverse_bps=5.0,
        depth_fraction=0.1,
    )
    arguments.update(changes)
    return execute_depth_order(**arguments)


def test_buy_uses_two_ask_levels_and_charges_cost_once():
    book = BookState.zero(1, 1000, np.array([100.0]))
    result = execute(book)
    price = 101.5 * 1.0005
    fee = 2 * price * 0.001
    assert result.filled_lots == 2
    assert result.unfilled_lots == 0
    assert result.price == pytest.approx(price)
    assert result.fee == pytest.approx(fee)
    assert book.quantities.tolist() == [2.0]
    assert book.mark_prices.tolist() == [100.0]
    assert book.cash == pytest.approx(1000 - 2 * price - fee)
    assert book.portfolio_value == pytest.approx(book.cash + 200)
    assert book.total_cost == pytest.approx(fee)
    assert book.peak_value == 1000


def test_sell_consumes_bids_and_preserves_partial_residual():
    book = BookState.zero(1, 1000, np.array([100.0]))
    result = execute(book, lot_count=-5)
    price = (99 + 2 * 98) / 3 * 0.9995
    assert result.filled_lots == -3
    assert result.unfilled_lots == -2
    assert result.price == pytest.approx(price)
    assert book.quantities.tolist() == [-3.0]
    assert book.cash == pytest.approx(1000 + 3 * price * 0.999)


def test_sub_lot_depth_aggregates_before_lot_rounding():
    book = BookState.zero(1, 1000, np.array([100.0]))
    result = execute(book, asks=((101.0, 5.0), (102.0, 5.0), (103.0, 5.0)))
    assert result.filled_lots == 1
    assert result.unfilled_lots == 1
    assert result.price == pytest.approx(101.5 * 1.0005)


def test_accepted_decimal_lots_can_close_exactly():
    book = BookState.zero(1, 1000, np.array([100.0]))
    rules = DepthOrderRules(0.001, 0.001, 100, 0, None)
    execute(book, lot_count=17, rules=rules)
    execute(book, lot_count=-17, rules=rules)
    assert book.quantities.tolist() == [0.0]


def test_empty_capacity_does_not_fill_or_discard_residual():
    book = BookState.zero(1, 1000, np.array([100.0]))
    result = execute(book, asks=((101.0, 5.0),))
    assert result.filled_lots == 0
    assert result.unfilled_lots == 2
    assert result.reason == "insufficient_depth"
    assert book.cash == 1000 and book.fill_count == 0


@pytest.mark.parametrize(
    "rules,reason",
    [
        (DepthOrderRules(1, 3, 100, 0, None), "quantity_below_minimum"),
        (DepthOrderRules(1, 1, 1, 0, None), "quantity_above_maximum"),
        (DepthOrderRules(1, 1, 100, 300, None), "notional_below_minimum"),
        (DepthOrderRules(1, 1, 100, 0, 150), "notional_above_maximum"),
    ],
)
def test_rule_rejection_preserves_account(rules, reason):
    book = BookState.zero(1, 1000, np.array([100.0]))
    result = execute(book, rules=rules)
    assert result.reason == reason
    assert result.unfilled_lots == 2
    assert book.cash == 1000 and book.fill_count == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"quote_at": NOW},
        {"quote_at": NOW - timedelta(seconds=1)},
        {"execution_at": NOW + timedelta(milliseconds=500)},
        {"execution_at": NOW + timedelta(seconds=7)},
        {
            "quote_at": NOW + timedelta(seconds=8),
            "execution_at": NOW + timedelta(seconds=11),
        },
        {"decision_at": NOW.replace(tzinfo=None)},
        {"lot_count": True},
        {"lot_count": 0},
        {"lot_count": 1.5},
        {"symbol_index": -1},
        {"symbol_index": True},
        {"fee_bps": -1},
        {"adverse_bps": float("nan")},
        {"adverse_bps": 10000},
        {"depth_fraction": 0},
        {"depth_fraction": 1.1},
        {"asks": ((98.0, 1.0),)},
        {"asks": ((101.0, -1.0),)},
        {"asks": ((102.0, 1.0), (101.0, 1.0))},
        {"bids": ()},
    ],
)
def test_invalid_input_is_rejected_before_book_mutation(changes):
    book = BookState.zero(1, 1000, np.array([100.0]))
    with pytest.raises(ValueError):
        execute(book, **changes)
    assert book.cash == 1000 and book.fill_count == 0


def test_fill_price_never_temporarily_revalues_existing_holding():
    book = BookState.from_weights(
        weights=np.array([0.5]), capital=1000, prices=np.array([100.0])
    )
    result = execute(book, asks=((200.0, 100.0),))
    assert result.filled_lots == 2
    assert book.peak_value == 1000
    assert book.portfolio_value == pytest.approx(1000 - 2 * (200.1 - 100) - 0.4002)
    assert book.max_drawdown == pytest.approx(1 - book.portfolio_value / 1000)


def test_bad_valuation_prices_fail_before_accounting_mutation():
    book = BookState.zero(1, 1000, np.array([100.0]))
    with pytest.raises(ValueError):
        book.execute_fill(
            symbol_index=0,
            quantity=1,
            fill_prices=np.array([101.0]),
            cost_amount=0,
            turnover=0.101,
            valuation_prices=np.array([-1.0]),
        )
    assert book.cash == 1000 and book.fill_count == 0


def test_contract_multiplier_is_part_of_notional_admission_and_cost():
    book = BookState.zero(1, 1000, np.array([100.0]), np.array([10.0]))
    rules = DepthOrderRules(1, 1, 100, 1000, None)
    result = execute(book, rules=rules, lot_count=1)
    assert result.filled_lots == 1
    assert result.fee == pytest.approx(101 * 1.0005 * 10 * 0.001)
    assert book.total_cost == pytest.approx(result.fee)
    book = BookState.zero(1, 1000, np.array([100.0]), np.array([10.0]))
    result = execute(book, rules=DepthOrderRules(1, 1, 100, 0, 1500))
    assert result.reason == "notional_above_maximum"
    assert book.cash == 1000


def test_worse_deeper_prices_cannot_overrun_maximum_notional():
    book = BookState.zero(1, 1000, np.array([100.0]))
    result = execute(
        book,
        asks=((101.0, 10.0), (200.0, 10.0)),
        rules=DepthOrderRules(1, 1, 100, 0, 250),
    )
    assert result.reason == "notional_above_maximum"
    assert book.cash == 1000


@pytest.mark.parametrize(
    "rules",
    [
        (0, 0, 100, 0, None),
        (1, -1, 100, 0, None),
        (1, 10, 1, 0, None),
        (1, 1, float("inf"), 0, None),
        (1, 1, 100, -1, None),
        (1, 1, 100, 10, 5),
        (1, 1, 100, 0, 0),
        (True, 1, 100, 0, None),
    ],
)
def test_invalid_rules_are_rejected(rules):
    with pytest.raises(ValueError):
        DepthOrderRules(*rules)


def test_overflow_rejection_preserves_canonical_account():
    book = BookState.zero(1, 1e308, np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        execute(
            book,
            lot_count=-1,
            rules=DepthOrderRules(1e308, 0, 1e308, 0, None),
            bids=((1.0, 1e308),),
            asks=((2.0, 1e308),),
            depth_fraction=1.0,
            fee_bps=0.0,
            adverse_bps=0.0,
        )
    assert book.cash == 1e308
    assert book.quantities.tolist() == [0.0]
    assert book.mark_prices.tolist() == [1.0]
    assert book.fill_count == 0 and book.turnover_total == 0


def test_accounting_counter_overflow_rejects_before_fill():
    book = BookState.zero(1, 1000, np.array([1.0]))
    book.turnover_total = 1e308
    with pytest.raises(ValueError, match="finite"):
        book.execute_fill(
            symbol_index=0,
            quantity=1.0,
            fill_prices=np.array([1.0]),
            cost_amount=0.0,
            turnover=1e308,
        )
    assert book.cash == 1000 and book.quantities.tolist() == [0.0]
    assert book.fill_count == 0 and book.turnover_total == 1e308
