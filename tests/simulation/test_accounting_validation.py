from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason


def _book() -> BookState:
    return BookState.zero(2, 1_000.0, np.array([100.0, 50.0]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quantities": np.array([])}, "non-empty finite"),
        ({"quantities": np.array([0.0])}, "identical shapes"),
        ({"mark_prices": np.array([100.0, 0.0])}, "strictly positive"),
        ({"cash": math.inf}, "cash must be finite"),
        ({"peak_value": 0.0}, "peak_value must be positive"),
        ({"fill_count": -1}, "execution counters"),
        ({"borrow_cost": -1.0}, "must be non-negative"),
        ({"maintenance_margin": 1.1}, r"within \[0, 1\]"),
        ({"insolvent": 1}, "insolvent must be a boolean"),
        ({"termination_reason": "unknown"}, "not supported"),
    ],
)
def test_book_state_rejects_invalid_constructor_state(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "quantities": np.zeros(2),
        "cash": 1_000.0,
        "mark_prices": np.array([100.0, 50.0]),
        "peak_value": 1_000.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        BookState(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BookState.zero(0, 1.0),
        lambda: BookState.zero(1, 0.0),
        lambda: BookState.zero(2, 1.0, np.array([1.0])),
        lambda: BookState.from_weights(
            weights=np.array([0.0]), capital=1.0, prices=np.array([1.0, 2.0])
        ),
        lambda: BookState.from_weights(
            weights=np.array([0.0]), capital=0.0, prices=np.array([1.0])
        ),
        lambda: BookState.from_weights(
            weights=np.array([0.0]), capital=1.0, prices=np.array([1.0]), max_gross=0.0
        ),
        lambda: BookState.from_weights(
            weights=np.array([1.1]), capital=1.0, prices=np.array([1.0])
        ),
        lambda: BookState.from_weights(
            weights=np.array([0.0]),
            capital=1.0,
            prices=np.array([1.0]),
            peak_value=0.5,
        ),
    ],
)
def test_book_factories_reject_invalid_inputs(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_accounting_operation_validation_and_edge_properties() -> None:
    book = _book()
    with pytest.raises(ValueError):
        book.apply_split(np.array([1.0]))
    with pytest.raises(ValueError):
        book.apply_dividend(np.array([0.0]))
    with pytest.raises(ValueError):
        book.apply_cash_interest(math.inf, periods_per_year=1)
    with pytest.raises(ValueError):
        book.apply_cash_interest(0.01, periods_per_year=0)
    with pytest.raises(ValueError):
        book.settle_positions(
            mask=np.array([True]),
            prices=np.array([100.0, 50.0]),
            recovery=np.ones(2),
        )
    with pytest.raises(ValueError):
        book.settle_positions(
            mask=np.array([True, False]),
            prices=np.array([100.0, 50.0]),
            recovery=np.array([1.1, 1.0]),
        )
    with pytest.raises(ValueError):
        book.set_margin(margin_used=-1.0, maintenance_margin=0.2)
    with pytest.raises(ValueError):
        book.set_margin(margin_used=0.0, maintenance_margin=1.1)
    with pytest.raises(ValueError):
        book.set_margin(
            margin_used=0.0,
            maintenance_margin=0.2,
            maintenance_requirement=-1.0,
        )
    with pytest.raises(ValueError):
        book.revalue(np.array([100.0]))
    with pytest.raises(ValueError):
        book.execute(
            fill_prices=np.array([100.0]),
            target_quantities=np.zeros(2),
            cost_amount=0.0,
            turnover=0.0,
        )
    with pytest.raises(ValueError):
        book.execute(
            fill_prices=np.array([100.0, 50.0]),
            target_quantities=np.zeros(2),
            cost_amount=-1.0,
            turnover=0.0,
        )
    with pytest.raises(ValueError):
        book.execute(
            fill_prices=np.array([100.0, 50.0]),
            target_quantities=np.zeros(2),
            cost_amount=0.0,
            turnover=-1.0,
        )
    with pytest.raises(ValueError):
        book.charge_borrow(-1.0)
    with pytest.raises(ValueError):
        book.mark_to_market(
            mark_prices=np.array([100.0, 50.0]), funding_amount=math.inf
        )
    with pytest.raises(ValueError):
        book.mark_to_market(
            mark_prices=np.array([100.0, 50.0]),
            funding_amount=0.0,
            period_start_value=0.0,
        )


def test_negative_value_properties_and_termination_are_fail_closed() -> None:
    book = BookState(
        quantities=np.array([1.0]),
        cash=-200.0,
        mark_prices=np.array([100.0]),
        peak_value=100.0,
        insolvent=True,
        termination_reason=EconomicTerminationReason.INSOLVENCY,
    )
    np.testing.assert_array_equal(book.weights, np.zeros(1))
    assert book.cash_weight == 0.0
    assert book.margin_utilization == 1.0
    clone = book.clone()
    assert clone.termination_reason is EconomicTerminationReason.INSOLVENCY


def test_book_state_clone_uses_constructor_copy_without_array_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    book = BookState(
        quantities=np.array([1.25, -0.5]),
        cash=1_000.0,
        mark_prices=np.array([100.0, 50.0]),
        peak_value=1_200.0,
        contract_multipliers=np.array([1.0, 2.0]),
        max_drawdown=0.125,
        turnover_total=42.0,
        total_cost=3.5,
        funding_pnl=-1.25,
        fill_count=4,
        rebalance_events=5,
        returns_history=[0.1, -0.2],
        borrow_cost=0.75,
        margin_used=80.0,
        maintenance_margin=0.2,
        maintenance_requirement=15.0,
    )
    constructor_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    original_post_init = BookState.__post_init__

    def capture_constructor_inputs(clone: BookState) -> None:
        assert clone.contract_multipliers is not None
        constructor_inputs.append(
            (clone.quantities, clone.mark_prices, clone.contract_multipliers)
        )
        original_post_init(clone)

    monkeypatch.setattr(BookState, "__post_init__", capture_constructor_inputs)
    clone = book.clone()

    assert len(constructor_inputs) == 1
    assert constructor_inputs[0][0] is book.quantities
    assert constructor_inputs[0][1] is book.mark_prices
    assert constructor_inputs[0][2] is book.contract_multipliers
    assert book.contract_multipliers is not None
    assert clone.contract_multipliers is not None
    np.testing.assert_array_equal(clone.quantities, book.quantities)
    np.testing.assert_array_equal(clone.mark_prices, book.mark_prices)
    np.testing.assert_array_equal(clone.contract_multipliers, book.contract_multipliers)
    assert clone.exact_quantities == book.exact_quantities
    for field_name in (
        "cash",
        "peak_value",
        "max_drawdown",
        "turnover_total",
        "total_cost",
        "funding_pnl",
        "fill_count",
        "rebalance_events",
        "borrow_cost",
        "margin_used",
        "maintenance_margin",
        "maintenance_requirement",
        "margin_deficit",
        "insolvent",
        "termination_reason",
    ):
        assert getattr(clone, field_name) == getattr(book, field_name)
    assert clone.portfolio_value == pytest.approx(book.portfolio_value)
    original_quantities = book.quantities.copy()
    original_mark_prices = book.mark_prices.copy()
    assert book.contract_multipliers is not None
    original_multipliers = book.contract_multipliers.copy()
    original_history = book.returns_history.copy()

    clone.quantities[0] = 9.0
    clone.mark_prices[0] = 110.0
    assert clone.contract_multipliers is not None
    clone.contract_multipliers[0] = 3.0
    clone.returns_history.append(0.3)

    assert not np.shares_memory(book.quantities, clone.quantities)
    assert not np.shares_memory(book.mark_prices, clone.mark_prices)
    assert not np.shares_memory(book.contract_multipliers, clone.contract_multipliers)
    np.testing.assert_array_equal(book.quantities, original_quantities)
    np.testing.assert_array_equal(book.mark_prices, original_mark_prices)
    np.testing.assert_array_equal(book.contract_multipliers, original_multipliers)
    assert book.returns_history == original_history
    assert clone.returns_history == [*original_history, 0.3]
    assert clone.returns_history is not book.returns_history


def test_mark_to_market_invalid_price_does_not_apply_funding_or_change_book() -> None:
    book = _book()
    before = book.clone()

    with pytest.raises(ValueError, match="mark_prices"):
        book.mark_to_market(
            mark_prices=np.array([-1.0, 50.0]),
            funding_amount=25.0,
            period_start_value=1_000.0,
        )

    assert book.cash == before.cash
    assert book.funding_pnl == before.funding_pnl
    assert book.returns_history == before.returns_history
    assert book.max_drawdown == before.max_drawdown
    assert book.termination_reason == before.termination_reason
    np.testing.assert_array_equal(book.mark_prices, before.mark_prices)
    np.testing.assert_array_equal(book.quantities, before.quantities)


def test_mark_to_market_nonfinite_projected_equity_does_not_mutate_book() -> None:
    book = BookState.zero(1, 1e308, np.array([100.0]))
    before = book.clone()

    with pytest.raises(ValueError, match="non-finite"):
        book.mark_to_market(
            mark_prices=np.array([100.0]),
            funding_amount=1e308,
            period_start_value=1e308,
        )

    assert book.cash == before.cash
    assert book.funding_pnl == before.funding_pnl
    assert book.returns_history == before.returns_history
    assert book.termination_reason == before.termination_reason
    np.testing.assert_array_equal(book.mark_prices, before.mark_prices)


def _assert_cash_and_inventory_unchanged(book: BookState, before: BookState) -> None:
    for name in (
        "cash",
        "funding_pnl",
        "borrow_cost",
        "peak_value",
        "max_drawdown",
        "turnover_total",
        "total_cost",
        "insolvent",
        "termination_reason",
    ):
        assert getattr(book, name) == getattr(before, name), name
    assert book.exact_quantities == before.exact_quantities
    assert book.returns_history == before.returns_history
    np.testing.assert_array_equal(book.mark_prices, before.mark_prices)
    np.testing.assert_array_equal(book.quantities, before.quantities)


def test_interest_overflow_does_not_corrupt_cash() -> None:
    book = BookState.zero(1, 1e308, np.array([1.0]))
    before = book.clone()
    with pytest.raises(ValueError, match="non-finite"):
        book.apply_cash_interest(2.0, year_fraction=1.0)
    _assert_cash_and_inventory_unchanged(book, before)


def test_dividend_overflow_does_not_corrupt_cash() -> None:
    book = BookState.from_weights(
        weights=np.array([1.0]), capital=1e308, prices=np.array([1.0])
    )
    before = book.clone()
    with pytest.raises(ValueError, match="non-finite"):
        book.apply_dividend(np.array([2.0]))
    _assert_cash_and_inventory_unchanged(book, before)


def test_borrow_overflow_does_not_corrupt_cash_or_borrow_cost() -> None:
    book = BookState.from_weights(
        weights=np.array([1.5]),
        capital=1e308,
        prices=np.array([1.0]),
        max_gross=2.0,
    )
    before = book.clone()
    with pytest.raises(ValueError, match="non-finite"):
        book.charge_borrow(1.7e308)
    _assert_cash_and_inventory_unchanged(book, before)


def test_revalue_overflow_does_not_corrupt_mark_prices() -> None:
    book = BookState.from_weights(
        weights=np.array([1.0]), capital=1e308, prices=np.array([1.0])
    )
    before = book.clone()
    with pytest.raises(ValueError, match="non-finite"):
        book.revalue(np.array([2.0]))
    _assert_cash_and_inventory_unchanged(book, before)


def test_split_price_overflow_does_not_corrupt_quantities_or_marks() -> None:
    book = BookState.from_weights(
        weights=np.array([1.0]), capital=100.0, prices=np.array([1.0])
    )
    before = book.clone()
    with pytest.raises(ValueError, match="non-finite"):
        book.apply_split(np.array([1e-309]))
    _assert_cash_and_inventory_unchanged(book, before)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowed_intermediate_cash_notional_rejects_atomically(
    sign: float,
) -> None:
    """Algebraically finite three-factor fills remain unsafe in the float book."""
    book = BookState.zero(
        1,
        1e200,
        np.array([1e200]),
        contract_multipliers=np.array([1e-200]),
    )
    assert math.isfinite(abs(sign * 1e200) * (1e200 * 1e-200))
    with pytest.raises(ValueError, match="proposed execution must preserve finite"):
        book.execute_fill(
            symbol_index=0,
            quantity=sign * 1e200,
            fill_prices=np.array([1e200]),
            cost_amount=0.0,
            turnover=0.0,
        )

    np.testing.assert_array_equal(book.quantities, np.array([0.0]))
    np.testing.assert_array_equal(book.mark_prices, np.array([1e200]))
    assert book.cash == 1e200
    assert book.portfolio_value == 1e200
    assert book.fill_count == 0
    assert book.rebalance_events == 0
    assert book.turnover_total == 0.0
    assert book.total_cost == 0.0


@pytest.mark.parametrize(
    ("capital", "price", "multiplier"),
    (
        (1_000.0, 1e308, 1e308),
        (1e-200, 1e308, 1.0),
        (1e-100, 1.0, 1.0),
    ),
)
def test_from_weights_rejects_unrepresentable_nonzero_target(
    capital: float, price: float, multiplier: float
) -> None:
    """A nonzero requested weight must never silently become a flat portfolio."""
    weights = np.array([1e-300 if capital == 1e-100 else 1.0])
    with pytest.raises(ValueError, match="weighted position"):
        BookState.from_weights(
            weights=weights,
            capital=capital,
            prices=np.array([price]),
            contract_multipliers=np.array([multiplier]),
        )


def test_from_weights_zero_weight_does_not_require_finite_product() -> None:
    """No active exposure needs pricing even when price * multiplier overflows."""
    book = BookState.from_weights(
        weights=np.array([0.0]),
        capital=1_000.0,
        prices=np.array([1e308]),
        contract_multipliers=np.array([1e308]),
    )
    assert book.quantities[0] == 0.0
    assert book.cash == 1_000.0
    assert book.portfolio_value == 1_000.0

@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_execute_fill_rejects_underflowed_nonzero_leg_atomically(
    sign: float,
) -> None:
    """A nonzero fill must not change inventory while debiting zero cash."""
    book = BookState.zero(
        1,
        1_000.0,
        np.array([1e-308]),
        contract_multipliers=np.array([1e-308]),
    )
    with pytest.raises(ValueError, match="nonzero.*notional"):
        book.execute_fill(
            symbol_index=0,
            quantity=sign,
            fill_prices=np.array([1e-308]),
            cost_amount=0.0,
            turnover=0.0,
        )
    np.testing.assert_array_equal(book.quantities, np.array([0.0]))
    assert book.exact_quantities == (0,)
    assert book.cash == 1_000.0
    assert book.portfolio_value == 1_000.0
    assert book.fill_count == 0
    assert book.rebalance_events == 0
    assert book.turnover_total == 0.0


def test_execute_offsetting_nonzero_legs_preserve_zero_net_cash() -> None:
    """Real buy/sell offsets are not an individual leg underflow."""
    book = BookState.zero(2, 1_000.0, np.array([100.0, 100.0]))
    book.execute(
        fill_prices=np.array([100.0, 100.0]),
        target_quantities=np.array([1.0, -1.0]),
        cost_amount=0.0,
        turnover=200.0,
    )
    np.testing.assert_array_equal(book.quantities, np.array([1.0, -1.0]))
    assert book.cash == 1_000.0
    assert book.fill_count == 2
    assert book.rebalance_events == 1


def test_constructor_rejects_nonzero_position_value_underflow() -> None:
    """A nonzero holding must not become zero in the valuation ledger."""
    with pytest.raises(ValueError, match="nonzero.*position.*value"):
        BookState(
            quantities=np.array([1.0]),
            cash=1_000.0,
            mark_prices=np.array([1e-308]),
            peak_value=1_000.0,
            contract_multipliers=np.array([1e-308]),
        )


def test_revalue_rejects_nonzero_position_value_underflow_atomically() -> None:
    """Revaluation must not erase an existing nonzero position economically."""
    book = BookState(
        quantities=np.array([1.0]),
        cash=1_000.0,
        mark_prices=np.array([1.0]),
        peak_value=1_001.0,
        contract_multipliers=np.array([1e-308]),
    )
    before_marks = book.mark_prices.copy()
    before_value = book.portfolio_value

    with pytest.raises(ValueError, match="nonzero.*position.*value"):
        book.revalue(np.array([1e-308]))

    np.testing.assert_array_equal(book.quantities, np.array([1.0]))
    np.testing.assert_array_equal(book.mark_prices, before_marks)
    assert book.portfolio_value == before_value


def test_mark_to_market_rejects_position_value_underflow_atomically() -> None:
    """A failed underflowing mark/funding transition must not partially mutate cash."""
    book = BookState(
        quantities=np.array([1.0]),
        cash=1_000.0,
        mark_prices=np.array([1.0]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1e-308]),
    )
    before_cash = book.cash
    before_marks = book.mark_prices.copy()
    before_funding = book.funding_pnl
    before_history = list(book.returns_history)

    with pytest.raises(ValueError, match="nonzero.*position.*value"):
        book.mark_to_market(
            mark_prices=np.array([1e-308]),
            funding_amount=5.0,
            period_start_value=1_000.0,
        )

    assert book.cash == before_cash
    np.testing.assert_array_equal(book.mark_prices, before_marks)
    assert book.funding_pnl == before_funding
    assert book.returns_history == before_history


def test_zero_position_allows_unrepresentable_unit_value() -> None:
    """A flat symbol does not need a nonzero representable position value."""
    book = BookState.zero(
        1,
        1_000.0,
        np.array([1e-308]),
        contract_multipliers=np.array([1e-308]),
    )
    book.revalue(np.array([1e-308]))
    assert book.quantities[0] == 0.0
    assert book.cash == 1_000.0
    assert book.portfolio_value == 1_000.0
