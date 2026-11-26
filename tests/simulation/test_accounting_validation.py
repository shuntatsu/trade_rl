from __future__ import annotations

import math
from array import array
from collections import UserList, deque
from decimal import Decimal
from fractions import Fraction

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
    with pytest.raises(ValueError, match="nonzero execution leg notional"):
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
def test_execute_fill_rejects_underflowed_nonzero_leg_atomically(sign: float) -> None:
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


@pytest.mark.parametrize("quantity", [1.0, -1.0])
def test_settle_positions_rejects_nonzero_proceeds_underflow_atomically(
    quantity: float,
) -> None:
    """A nonzero recovered settlement leg must not silently become cash zero."""
    book = BookState(
        quantities=np.array([quantity]),
        cash=1_000.0,
        mark_prices=np.array([1.0]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1e-308]),
    )
    before_cash = book.cash
    before_quantities = book.quantities.copy()
    before_exact = book._exact_quantities
    before_marks = book.mark_prices.copy()
    before_peak = book.peak_value
    before_drawdown = book.max_drawdown
    before_fills = book.fill_count

    with pytest.raises(ValueError, match="nonzero settlement proceeds"):
        book.settle_positions(
            mask=np.array([True]),
            prices=np.array([1e-308]),
            recovery=np.array([1e-308]),
        )

    assert book.cash == before_cash
    np.testing.assert_array_equal(book.quantities, before_quantities)
    np.testing.assert_array_equal(book.mark_prices, before_marks)
    assert book._exact_quantities == before_exact
    assert book.peak_value == before_peak
    assert book.max_drawdown == before_drawdown
    assert book.fill_count == before_fills


@pytest.mark.parametrize(
    "mask",
    [
        np.array([1, 0]),
        np.array([2, 0]),
        np.array([math.nan, 0.0]),
        np.array(["false", "true"]),
        np.array([None, True], dtype=object),
        np.array([[True, False]]),
        np.ma.array([True, False], mask=[True, False]),
        [np.ma.array(True, mask=True), False],
        np.ma.array([True, False], mask=[True, False]).astype(bool).view(np.ndarray),
        memoryview(np.ma.array([True, False], mask=[True, False])),
    ],
)
def test_settle_positions_refuses_non_boolean_or_non_vector_mask_atomically(
    mask: np.ndarray,
) -> None:
    """Never turn malformed delisting selectors into irreversible liquidation."""
    book = BookState.from_weights(
        weights=np.array([0.5, 0.0]),
        capital=1_000.0,
        prices=np.array([100.0, 50.0]),
    )
    before = book.clone()

    with pytest.raises(ValueError, match="settlement mask must be a boolean vector"):
        book.settle_positions(
            mask=mask,
            prices=np.array([100.0, 50.0]),
            recovery=np.array([0.5, 1.0]),
        )

    _assert_cash_and_inventory_unchanged(book, before)
    assert book._exact_quantities == before._exact_quantities
    assert book.total_cost == before.total_cost
    assert book.fill_count == before.fill_count


def test_settle_positions_explicit_boolean_mask_still_settles_selected_leg() -> None:
    book = BookState.from_weights(
        weights=np.array([0.5, 0.0]),
        capital=1_000.0,
        prices=np.array([100.0, 50.0]),
    )
    proceeds = book.settle_positions(
        mask=np.array([True, False]),
        prices=np.array([100.0, 50.0]),
        recovery=np.array([0.5, 1.0]),
    )
    assert proceeds == 250.0
    assert book.cash == 750.0
    np.testing.assert_array_equal(book.quantities, np.zeros(2))


def test_settle_positions_zero_recovery_is_legitimate() -> None:
    """Exact zero recovery is allowed to clear the position without proceeds."""
    book = BookState(
        quantities=np.array([1.0]),
        cash=1_000.0,
        mark_prices=np.array([1.0]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1e-308]),
    )
    proceeds = book.settle_positions(
        mask=np.array([True]),
        prices=np.array([1e-308]),
        recovery=np.array([0.0]),
    )
    assert proceeds == 0.0
    assert book.cash == 1_000.0
    assert book.quantities[0] == 0.0


def test_gemini4_execute_rejects_underflowed_residual_position_atomically() -> None:
    book = BookState(
        quantities=np.array([1.0]),
        cash=1_000.0,
        mark_prices=np.array([1.0]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1e-200]),
    )
    before = book.clone()

    with pytest.raises(ValueError, match="nonzero position value"):
        book.execute(
            fill_prices=np.array([1.0]),
            target_quantities=np.array([1e-200]),
            cost_amount=0.0,
            turnover=1.0,
        )

    _assert_cash_and_inventory_unchanged(book, before)


def test_gemini4_execute_fill_rejects_underflowed_valuation_atomically() -> None:
    book = BookState.zero(
        1,
        1_000.0,
        np.array([1e-200]),
        contract_multipliers=np.array([1e-200]),
    )
    before = book.clone()

    with pytest.raises(ValueError, match="nonzero position value"):
        book.execute_fill(
            symbol_index=0,
            quantity=1.0,
            fill_prices=np.array([1e200]),
            valuation_prices=np.array([1e-200]),
            cost_amount=0.0,
            turnover=1.0,
        )

    _assert_cash_and_inventory_unchanged(book, before)


def test_gemini4_settlement_rejects_float_conversion_overflow_atomically() -> None:
    book = BookState(
        quantities=np.array([1e200]),
        cash=1_000.0,
        mark_prices=np.array([1e-200]),
        peak_value=1_001.0,
        contract_multipliers=np.array([1.0]),
    )
    before = book.clone()

    with pytest.raises(ValueError, match="settlement.*(?:notional|proceeds)"):
        book.settle_positions(
            mask=np.array([True]),
            prices=np.array([1e200]),
            recovery=np.array([1.0]),
        )

    _assert_cash_and_inventory_unchanged(book, before)


def test_gemini4_settlement_rejects_fsum_overflow_atomically() -> None:
    book = BookState(
        quantities=np.array([1e308, 1e308]),
        cash=1_000.0,
        mark_prices=np.array([1e-308, 1e-308]),
        peak_value=1_002.0,
        contract_multipliers=np.ones(2),
    )
    before = book.clone()

    with pytest.raises(ValueError, match="settlement.*(?:notional|proceeds)"):
        book.settle_positions(
            mask=np.array([True, True]),
            prices=np.ones(2),
            recovery=np.ones(2),
        )

    _assert_cash_and_inventory_unchanged(book, before)


def test_gemini4_execute_rejects_fsum_overflow_atomically() -> None:
    book = BookState.zero(2, 1_000.0, np.ones(2))
    before = book.clone()

    with pytest.raises(ValueError, match="execution.*notional|account"):
        book.execute(
            fill_prices=np.ones(2),
            target_quantities=np.array([1e308, 1e308]),
            cost_amount=0.0,
            turnover=0.0,
        )

    _assert_cash_and_inventory_unchanged(book, before)


def test_gemini4_execute_rejects_nonfinite_signed_legs_before_netting() -> None:
    book = BookState.zero(
        2,
        1_000.0,
        np.ones(2),
        contract_multipliers=np.array([1e-200, 1e-200]),
    )
    before = book.clone()

    with pytest.raises(ValueError, match="execution.*notional"):
        book.execute(
            fill_prices=np.array([1e200, 1e200]),
            target_quantities=np.array([1e200, -1e200]),
            cost_amount=0.0,
            turnover=0.0,
        )

    _assert_cash_and_inventory_unchanged(book, before)


def test_gemini4_settlement_rejects_left_associative_overflow_atomically() -> None:
    book = BookState(
        quantities=np.array([1e200]),
        cash=1_000.0,
        mark_prices=np.array([1e-200]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1e-200]),
    )
    before = book.clone()

    with pytest.raises(ValueError, match="settlement.*(?:notional|proceeds)"):
        book.settle_positions(
            mask=np.array([True]),
            prices=np.array([1e200]),
            recovery=np.array([1.0]),
        )

    _assert_cash_and_inventory_unchanged(book, before)


@pytest.mark.parametrize("field", ["quantities", "mark_prices", "contract_multipliers"])
def test_book_constructor_rejects_masked_numeric_vector(field: str) -> None:
    # MaskedArray -> np.asarray discards the mask and exposes a hidden value.
    values: dict[str, object] = {
        "quantities": np.array([1.0, 0.0]),
        "cash": 1_000.0,
        "mark_prices": np.array([100.0, 50.0]),
        "contract_multipliers": np.ones(2),
        "peak_value": 1_100.0,
    }
    values[field] = np.ma.array(values[field], mask=[True, False])
    with pytest.raises(ValueError, match="masked"):
        BookState(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["prices", "recovery"])
def test_settlement_refuses_masked_price_or_recovery_without_mutation(
    field: str,
) -> None:
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0, 50.0]),
        peak_value=1_050.0,
    )
    before = book.clone()
    vectors = {
        "mask": np.array([True, False]),
        "prices": np.array([100.0, 50.0]),
        "recovery": np.array([1.0, 1.0]),
    }
    vectors[field] = np.ma.array(vectors[field], mask=[True, False])
    with pytest.raises(ValueError, match="masked"):
        book.settle_positions(**vectors)
    assert book.cash == before.cash
    assert book.exact_quantities == before.exact_quantities
    np.testing.assert_array_equal(book.quantities, before.quantities)
    np.testing.assert_array_equal(book.mark_prices, before.mark_prices)
    assert book.total_cost == before.total_cost
    assert book.fill_count == before.fill_count
    assert book.max_drawdown == before.max_drawdown
    assert book.termination_reason == before.termination_reason


@pytest.mark.parametrize(
    "operation", ["split", "dividend", "revalue", "mark_to_market"]
)
def test_account_transition_refuses_masked_numeric_input_atomically(
    operation: str,
) -> None:
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0, 50.0]),
        peak_value=1_100.0,
    )
    before = book.clone()
    masked = np.ma.array([2.0, 1.0], mask=[True, False])
    with pytest.raises(ValueError, match="masked"):
        if operation == "split":
            book.apply_split(masked)
        elif operation == "dividend":
            book.apply_dividend(masked)
        elif operation == "revalue":
            book.revalue(masked)
        else:
            book.mark_to_market(
                mark_prices=masked, funding_amount=10.0, period_start_value=1_050.0
            )
    assert book.cash == before.cash
    assert book.exact_quantities == before.exact_quantities
    np.testing.assert_array_equal(book.quantities, before.quantities)
    np.testing.assert_array_equal(book.mark_prices, before.mark_prices)
    assert book.total_cost == before.total_cost
    assert book.funding_pnl == before.funding_pnl
    assert book.returns_history == before.returns_history
    assert book.peak_value == before.peak_value
    assert book.termination_reason == before.termination_reason


@pytest.mark.parametrize("wrapper", ("list", "tuple", "nested_list"))
def test_book_constructor_rejects_nested_masked_numeric_vector(wrapper: str) -> None:
    hidden = np.ma.array([100.0, 50.0], mask=[True, False])
    value = (
        [hidden]
        if wrapper == "list"
        else (hidden,)
        if wrapper == "tuple"
        else [[hidden]]
    )
    with pytest.raises(ValueError, match="masked"):
        BookState.zero(2, 1_000.0, value)


def test_settlement_nested_masked_recovery_is_atomic() -> None:
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0, 50.0]),
        peak_value=1_100.0,
    )
    before = book.clone()
    wrapped_recovery = [[np.ma.array([0.5, 1.0], mask=[True, False])]]
    with pytest.raises(ValueError, match="masked"):
        book.settle_positions(
            mask=np.array([True, False]),
            prices=np.array([100.0, 50.0]),
            recovery=wrapped_recovery,
        )
    assert book.cash == before.cash
    assert book.exact_quantities == before.exact_quantities
    np.testing.assert_array_equal(book.mark_prices, before.mark_prices)
    assert book.total_cost == before.total_cost
    assert book.fill_count == before.fill_count
    assert book.termination_reason == before.termination_reason


def test_ordinary_nested_numeric_vectors_remain_valid() -> None:
    book = BookState.zero(2, 1_000.0, [[100.0, 50.0]])
    np.testing.assert_array_equal(book.mark_prices, [100.0, 50.0])


@pytest.mark.parametrize("wrapper", ("deque", "userlist", "nested_userlist"))
def test_book_constructor_rejects_other_sequence_masked_array(wrapper: str) -> None:
    hidden = np.ma.array([100.0, 50.0], mask=[True, False])
    if wrapper == "deque":
        input_values = deque([hidden])
    elif wrapper == "userlist":
        input_values = UserList([hidden])
    else:
        input_values = [UserList([deque([hidden])])]
    with pytest.raises(ValueError, match="masked"):
        BookState.zero(2, 1_000.0, input_values)


@pytest.mark.parametrize("wrapper", (deque, UserList))
def test_settlement_refuses_other_sequence_masked_recovery_atomically(
    wrapper: object,
) -> None:
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0, 50.0]),
        peak_value=1_100.0,
    )
    before = book.clone()
    hidden = np.ma.array([0.5, 1.0], mask=[True, False])
    with pytest.raises(ValueError, match="masked"):
        book.settle_positions(
            mask=np.array([True, False]),
            prices=np.array([100.0, 50.0]),
            recovery=wrapper([hidden]),
        )
    assert book.cash == before.cash
    assert book.exact_quantities == before.exact_quantities
    np.testing.assert_array_equal(book.mark_prices, before.mark_prices)
    assert book.total_cost == before.total_cost
    assert book.fill_count == before.fill_count
    assert book.termination_reason == before.termination_reason


def test_book_constructor_rejects_opaque_array_like_mask_adapter() -> None:
    class MaskAdapter:
        def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
            values = np.ma.array([100.0, 50.0], mask=[True, False])
            return values if dtype is None else values.astype(dtype)

    # A custom __array__ can discard hidden missingness during conversion.
    with pytest.raises(ValueError, match="array-like"):
        BookState.zero(2, 1_000.0, MaskAdapter())


@pytest.mark.parametrize("wrapper", (deque, UserList))
def test_ordinary_other_numeric_sequences_remain_valid(wrapper: object) -> None:
    book = BookState.zero(2, 1_000.0, wrapper([100.0, 50.0]))
    np.testing.assert_array_equal(book.mark_prices, [100.0, 50.0])


class _DuckVector:
    """An indexable NumPy input that is not a registered ABC Sequence."""

    def __init__(self, values: object) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> object:
        return self.values[index]


class _ArrayInterfaceVector:
    def __init__(self) -> None:
        self.values = np.array([100.0, 50.0], dtype=np.float64)
        self.mask = np.array([True, False])

    @property
    def __array_interface__(self) -> dict[str, object]:
        return {**self.values.__array_interface__, "mask": self.mask}


class _ArrayStructVector:
    def __init__(self) -> None:
        self.values = np.array([100.0, 50.0], dtype=np.float64)
        self.mask = np.array([True, False])

    @property
    def __array_struct__(self) -> object:
        return self.values.__array_struct__


@pytest.mark.parametrize("nested", [False, True])
def test_duck_sequence_rejects_nested_masked_prices(nested: bool) -> None:
    # NumPy consumes __len__/__getitem__ even without Sequence registration.
    wrapped = _DuckVector([np.ma.array([100.0, 50.0], mask=[True, False])])
    value = [wrapped] if nested else wrapped
    with pytest.raises(ValueError, match="masked"):
        BookState.zero(2, 1_000.0, value)


@pytest.mark.parametrize("wrapper", (_ArrayInterfaceVector, _ArrayStructVector))
@pytest.mark.parametrize("nested", [False, True])
def test_custom_numpy_buffer_protocols_refused_before_conversion(
    wrapper: object, nested: bool
) -> None:
    value = [wrapper()] if nested else wrapper()
    with pytest.raises(ValueError, match="opaque array-like"):
        BookState.zero(2, 1_000.0, value)


class _TransientOpaqueArrayInterface:
    """A descriptor that hides a masked backing scalar only on first lookup."""

    def __init__(self) -> None:
        self.masked = np.ma.array(100.0, mask=True)
        self.data = np.array(100.0)
        self.lookups = 0

    @property
    def __array_interface__(self) -> dict[str, object]:
        self.lookups += 1
        if self.lookups == 1:
            raise AttributeError("temporarily missing buffer protocol")
        return self.data.__array_interface__

    def __float__(self) -> float:
        return float(self.masked.data)


class _TransientOpaqueArrayStruct:
    def __init__(self) -> None:
        self.masked = np.ma.array(100.0, mask=True)
        self.data = np.array(100.0)
        self.lookups = 0

    @property
    def __array_struct__(self) -> object:
        self.lookups += 1
        if self.lookups == 1:
            raise AttributeError("temporarily missing buffer protocol")
        return self.data.__array_struct__

    def __float__(self) -> float:
        return float(self.masked.data)


@pytest.mark.parametrize(
    "adapter_class", (_TransientOpaqueArrayInterface, _TransientOpaqueArrayStruct)
)
@pytest.mark.parametrize("nested", (False, True))
def test_transient_array_protocol_mask_is_refused_before_numpy_cast(
    adapter_class: type,
    nested: bool,
) -> None:
    adapter = adapter_class()
    vector: object = [[adapter]] if nested else [adapter]
    with pytest.raises(ValueError, match="opaque array-like"):
        BookState.zero(1, 1_000.0, vector)
    # A protocol descriptor must not be invoked just to determine its presence.
    assert adapter.lookups == 0


class _DynamicArrayInterfaceGetattr:
    def __init__(self) -> None:
        self.masked = np.ma.array(100.0, mask=True)
        self.data = np.array(100.0)
        self.lookups = 0

    def __getattr__(self, name: str) -> object:
        if name != "__array_interface__":
            raise AttributeError(name)
        self.lookups += 1
        if self.lookups == 1:
            raise AttributeError("first dynamically supplied interface is unavailable")
        return self.data.__array_interface__

    def __float__(self) -> float:
        return float(self.masked.data)


class _DynamicArrayInterfaceGetattribute:
    def __init__(self) -> None:
        self.masked = np.ma.array(100.0, mask=True)
        self.data = np.array(100.0)
        self.lookups = 0

    def __getattribute__(self, name: str) -> object:
        if name == "__array_interface__":
            lookups = object.__getattribute__(self, "lookups") + 1
            object.__setattr__(self, "lookups", lookups)
            if lookups == 1:
                raise AttributeError(
                    "first dynamically supplied interface is unavailable"
                )
            return object.__getattribute__(self, "data").__array_interface__
        return object.__getattribute__(self, name)

    def __float__(self) -> float:
        return float(self.masked.data)


@pytest.mark.parametrize(
    "adapter_class", (_DynamicArrayInterfaceGetattr, _DynamicArrayInterfaceGetattribute)
)
@pytest.mark.parametrize("nested", (False, True))
def test_dynamic_numpy_array_protocol_mask_is_rejected_without_lookup(
    adapter_class: type, nested: bool
) -> None:
    adapter = adapter_class()
    value: object = [[adapter]] if nested else [adapter]
    with pytest.raises(ValueError, match="opaque array-like"):
        BookState.zero(1, 1_000.0, value)
    assert adapter.lookups == 0


@pytest.mark.parametrize(
    "value",
    (
        [Decimal("100.0")],
        deque([100.0]),
        range(100, 101),
        array("d", [100.0]),
        memoryview(array("d", [100.0])),
    ),
)
def test_builtin_numeric_collections_remain_accepted_with_opaque_protocol_guard(
    value: object,
) -> None:
    book = BookState.zero(1, 1_000.0, value)
    np.testing.assert_array_equal(book.mark_prices, [100.0])


def _masked_origin_memoryview(*, has_masked_entry: bool = True) -> memoryview:
    # memoryview retains its MaskedArray owner, even though indexed values
    # expose only the unmasked buffer and would otherwise lose missingness.
    return memoryview(np.ma.array([100.0, 50.0], mask=[has_masked_entry, False]))


@pytest.mark.parametrize("has_masked_entry", [True, False])
@pytest.mark.parametrize("wrapper", ["direct", "nested", "chained", "object_array"])
def test_book_rejects_memoryview_with_masked_array_owner(
    wrapper: str, has_masked_entry: bool
) -> None:
    view = _masked_origin_memoryview(has_masked_entry=has_masked_entry)
    assert np.ma.isMaskedArray(view.obj)
    if wrapper == "direct":
        value: object = view
    elif wrapper == "nested":
        value = [[view]]
    elif wrapper == "chained":
        value = memoryview(view)
    else:
        objects = np.empty(1, dtype=object)
        objects[0] = view  # Keep view identity rather than pre-casting its values.
        value = objects
    with pytest.raises(ValueError, match="masked array"):
        BookState.zero(2, 1_000.0, value)


def _ndarray_with_masked_base(*, masked: bool = True) -> np.ndarray:
    # Casting this MaskedArray to a plain ndarray preserves the masked owner
    # in the base chain, unlike caller-owned np.asarray materialization.
    source = np.ma.array([100.0, 50.0], mask=[masked, False])
    unmasked_type = source.astype(np.float64).view(np.ndarray)
    assert not np.ma.isMaskedArray(unmasked_type)
    assert np.ma.isMaskedArray(unmasked_type.base)
    return unmasked_type


@pytest.mark.parametrize("masked", (True, False))
@pytest.mark.parametrize(
    "wrapper",
    ("direct", "list", "chained", "object_array", "memoryview", "memoryview_list"),
)
def test_book_refuses_plain_ndarray_with_masked_base_provenance(
    masked: bool, wrapper: str
) -> None:
    values = _ndarray_with_masked_base(masked=masked)
    if wrapper == "direct":
        input_values: object = values
    elif wrapper == "list":
        input_values = [values]
    elif wrapper == "chained":
        input_values = values.view(np.ndarray)
    elif wrapper == "memoryview":
        input_values = memoryview(values)
    elif wrapper == "memoryview_list":
        input_values = [memoryview(values)]
    else:
        nested = np.empty(1, dtype=object)
        nested[0] = values  # Retain owner chain; do not pre-coerce to scalars.
        input_values = nested
    with pytest.raises(ValueError, match="masked array"):
        BookState.zero(2, 1_000.0, input_values)


@pytest.mark.parametrize("wrapped", [False, True])
def test_ndarray_with_masked_base_refuses_settlement_without_mutation(
    wrapped: bool,
) -> None:
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0, 50.0]),
        peak_value=1_100.0,
    )
    prior = book.clone()
    values = _ndarray_with_masked_base()
    with pytest.raises(ValueError, match="masked array"):
        book.settle_positions(
            mask=np.array([True, False]),
            prices=memoryview(values) if wrapped else values,
            recovery=np.ones(2),
        )
    _assert_cash_and_inventory_unchanged(book, prior)
    assert book.termination_reason == prior.termination_reason


def test_plain_ndarray_base_chain_without_masks_still_admitted() -> None:
    original = np.array([100.0, 50.0])
    values = original.view(np.ndarray)
    assert isinstance(values.base, np.ndarray)
    assert not np.ma.isMaskedArray(values.base)
    book = BookState.zero(2, 1_000.0, values)
    np.testing.assert_array_equal(book.mark_prices, original)


def test_memoryview_with_masked_owner_refuses_settlement_atomically() -> None:
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0, 50.0]),
        peak_value=1_100.0,
    )
    prior = book.clone()
    with pytest.raises(ValueError, match="masked array"):
        book.settle_positions(
            mask=np.array([True, False]),
            prices=_masked_origin_memoryview(),
            recovery=np.ones(2),
        )
    _assert_cash_and_inventory_unchanged(book, prior)
    assert book.termination_reason == prior.termination_reason


def test_plain_numeric_memoryview_remains_admitted_as_prices() -> None:
    values = array("d", [100.0, 50.0])
    book = BookState.zero(2, 1_000.0, memoryview(values))
    np.testing.assert_array_equal(book.mark_prices, [100.0, 50.0])


def test_transient_array_interface_settlement_is_atomic() -> None:
    book = BookState(
        quantities=np.array([1.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0]),
        peak_value=1_100.0,
    )
    prior = book.clone()
    adapter = _TransientOpaqueArrayInterface()
    with pytest.raises(ValueError, match="opaque array-like"):
        book.settle_positions(
            mask=np.array([True]),
            prices=[adapter],
            recovery=np.array([1.0]),
        )
    _assert_cash_and_inventory_unchanged(book, prior)
    assert book.termination_reason == prior.termination_reason


def test_settlement_refuses_duck_masked_recovery_atomically() -> None:
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0, 50.0]),
        peak_value=1_100.0,
    )
    prior = book.clone()
    masked = _DuckVector([np.ma.array([0.5, 1.0], mask=[True, False])])
    with pytest.raises(ValueError, match="masked"):
        book.settle_positions(
            mask=np.array([True, False]),
            prices=np.array([100.0, 50.0]),
            recovery=masked,
        )
    _assert_cash_and_inventory_unchanged(book, prior)
    assert book.termination_reason == prior.termination_reason


def test_numeric_duck_sequence_retains_prior_admission() -> None:
    book = BookState.zero(2, 1_000.0, _DuckVector([100.0, 50.0]))
    np.testing.assert_array_equal(book.mark_prices, [100.0, 50.0])


def test_nested_dynamic_sequence_keeps_visited_ancestors_alive() -> None:
    import weakref

    class DynamicVector:
        def __init__(self, depth: int, parent: "DynamicVector | None" = None) -> None:
            self.depth = depth
            self.parent = weakref.ref(parent) if parent is not None else None

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> object:
            if index != 0:
                raise IndexError(index)
            if self.parent is not None and self.parent() is None:
                raise AssertionError("sequence ancestor released before validation")
            if self.depth > 0:
                return DynamicVector(self.depth - 1, self)
            return np.ma.array([100.0, 50.0], mask=[True, False])

    with pytest.raises(ValueError, match="masked"):
        BookState.zero(2, 1_000.0, DynamicVector(16))


def test_opaque_interface_probe_error_is_fail_closed() -> None:
    class BrokenInterface:
        @property
        def __array_interface__(self) -> object:
            raise RuntimeError("buffer metadata cannot be inspected")

    with pytest.raises(ValueError, match="opaque"):
        BookState.zero(2, 1_000.0, BrokenInterface())


def test_duck_numeric_vector_uses_single_checked_snapshot() -> None:
    class SwitchingVector:
        def __init__(self) -> None:
            self.reads = 0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> object:
            if index != 0:
                raise IndexError(index)
            self.reads += 1
            if self.reads == 1:
                return [100.0, 50.0]
            # A second read would expose masked data and was silently cast.
            return np.ma.array([100.0, 50.0], mask=[True, False])

    input_values = SwitchingVector()
    book = BookState.zero(2, 1_000.0, input_values)
    np.testing.assert_array_equal(book.mark_prices, [100.0, 50.0])
    assert input_values.reads == 1


def test_shared_sequence_reference_keeps_numeric_array_semantics() -> None:
    shared = _DuckVector([100.0])
    book = BookState.zero(2, 1_000.0, [shared, shared])
    np.testing.assert_array_equal(book.mark_prices, [100.0, 100.0])


@pytest.mark.parametrize("kind", ("list", "object_array"))
def test_cyclic_numeric_container_is_refused_before_cast(kind: str) -> None:
    if kind == "list":
        values: list[object] = []
        values.append(values)
        input_values: object = values
    else:
        object_array = np.empty(1, dtype=object)
        object_array[0] = object_array
        input_values = object_array
    with pytest.raises(ValueError, match="cyclic"):
        BookState.zero(2, 1_000.0, input_values)


def test_normal_numeric_object_array_still_converts_to_prices() -> None:
    prices = np.array([100.0, 50.0], dtype=object)
    book = BookState.zero(2, 1_000.0, prices)
    np.testing.assert_array_equal(book.mark_prices, [100.0, 50.0])


class _MaskedFloatScalar:
    def __init__(self, value: float) -> None:
        self.hidden = np.ma.array(value, mask=True)

    def __float__(self) -> float:
        return float(self.hidden.data)


class _MaskedIndexScalar:
    def __init__(self, value: int) -> None:
        self.hidden = np.ma.array(value, mask=True)

    def __index__(self) -> int:
        return int(self.hidden.data)


@pytest.mark.parametrize("scalar_type", (_MaskedFloatScalar, _MaskedIndexScalar))
@pytest.mark.parametrize("wrapper", ("direct", "nested", "object_array"))
def test_untrusted_scalar_converter_cannot_hide_masked_financial_input(
    scalar_type: type, wrapper: str
) -> None:
    adapter = scalar_type(100)
    if wrapper == "direct":
        value: object = [adapter]
    elif wrapper == "nested":
        value = [[adapter]]
    else:
        objects = np.empty(1, dtype=object)
        objects[0] = adapter
        value = objects
    with pytest.raises(ValueError, match="opaque scalar"):
        BookState.zero(1, 1_000.0, value)


def test_opaque_masked_scalar_settlement_refuses_atomically() -> None:
    book = BookState(
        quantities=np.array([1.0]),
        cash=1_000.0,
        mark_prices=np.array([100.0]),
        peak_value=1_100.0,
    )
    before = book.clone()
    with pytest.raises(ValueError, match="opaque scalar"):
        book.settle_positions(
            mask=np.array([True]),
            prices=[_MaskedFloatScalar(100.0)],
            recovery=np.array([1.0]),
        )
    _assert_cash_and_inventory_unchanged(book, before)
    assert book.termination_reason == before.termination_reason


@pytest.mark.parametrize(
    "values",
    (
        [100.0, 50.0],
        [Decimal("100.0"), Decimal("50.0")],
        [Fraction(100, 1), Fraction(50, 1)],
        [np.float64(100.0), np.int64(50)],
    ),
)
def test_native_numeric_scalar_atoms_remain_admitted(values: object) -> None:
    book = BookState.zero(2, 1_000.0, values)
    np.testing.assert_array_equal(book.mark_prices, [100.0, 50.0])


class _MisreportedNumericDtype(np.ndarray):
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)


class _MisreportedArrayList(np.ndarray):
    def tolist(self) -> list[float]:
        return [100.0]


@pytest.mark.parametrize(
    ("subclass", "element"),
    (
        (_MisreportedNumericDtype, _MaskedFloatScalar(100.0)),
        (_MisreportedArrayList, np.ma.array(100.0, mask=True)),
    ),
)
@pytest.mark.parametrize("nested", (False, True))
def test_ndarray_subclass_cannot_override_native_object_admission(
    subclass: type, element: object, nested: bool
) -> None:
    raw = np.empty(1, dtype=object)
    raw[0] = element
    wrapped = raw.view(subclass)
    with pytest.raises(ValueError, match="masked array|opaque scalar"):
        BookState.zero(1, 1_000.0, [wrapped] if nested else wrapped)


def test_native_numeric_ndarray_subclass_remains_admitted() -> None:
    class NumericView(np.ndarray):
        pass

    raw = np.array([100.0, 50.0])
    book = BookState.zero(2, 1_000.0, raw.view(NumericView))
    np.testing.assert_array_equal(book.mark_prices, raw)


def test_untrusted_numpy_scalar_subclass_is_refused() -> None:
    class OpaqueScalar(np.float64):
        def __float__(self) -> float:
            return 100.0

    with pytest.raises(ValueError, match="opaque scalar"):
        BookState.zero(1, 1_000.0, [OpaqueScalar(100.0)])


def test_structured_object_dtype_cannot_bypass_scalar_admission() -> None:
    raw = np.empty(1, dtype=[("price", object)])
    raw["price"][0] = _MaskedFloatScalar(100.0)
    with pytest.raises(ValueError, match="opaque array-like"):
        BookState.zero(1, 1_000.0, raw)
    with pytest.raises(ValueError, match="opaque scalar"):
        BookState.zero(1, 1_000.0, [raw[0]])


@pytest.mark.parametrize("wrap_memoryview", [False, True])
@pytest.mark.parametrize("field", ["mask", "prices", "recovery"])
def test_opaque_stride_masked_owner_refuses_settlement_atomically(
    field: str, wrap_memoryview: bool
) -> None:
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1000.0,
        mark_prices=np.array([100.0, 50.0]),
        peak_value=1100.0,
    )
    before = book.clone()
    kwargs = dict(
        mask=np.array([True, False]),
        prices=np.array([100.0, 50.0]),
        recovery=np.ones(2),
    )
    owner = (
        np.ma.array(kwargs[field], mask=[True, False])
        .astype(kwargs[field].dtype)
        .view(np.ndarray)
    )
    hidden = np.lib.stride_tricks.as_strided(owner)
    assert np.ma.isMaskedArray(hidden.base.base.base)
    kwargs[field] = memoryview(hidden) if wrap_memoryview else hidden
    with pytest.raises(ValueError, match="masked array|opaque array ownership"):
        book.settle_positions(**kwargs)
    _assert_cash_and_inventory_unchanged(book, before)
    assert book.termination_reason == before.termination_reason


def test_book_refuses_opaque_stride_owner_even_without_observable_mask() -> None:
    values = np.lib.stride_tricks.as_strided(np.array([100.0, 50.0]))
    with pytest.raises(ValueError, match="opaque array ownership"):
        BookState.zero(2, 1000.0, values)
    # A caller-owned copy has deliberately discarded the ownership chain.
    book = BookState.zero(2, 1000.0, values.copy())
    np.testing.assert_array_equal(book.mark_prices, [100.0, 50.0])


def test_opaque_array_owner_is_rejected_without_reading_foreign_base() -> None:
    class Owner:
        def __init__(self) -> None:
            self.raw = np.array([100.0, 50.0])
            self.__array_interface__ = self.raw.__array_interface__

        @property
        def base(self) -> object:
            raise AssertionError("foreign base must not be read")

    owner = Owner()
    values = np.asarray(owner)
    assert values.base is owner
    with pytest.raises(ValueError, match="opaque array ownership"):
        BookState.zero(2, 1000.0, values)


@pytest.mark.parametrize("buffer_type", [bytes, bytearray, array])
def test_book_admits_known_native_buffer_owners(buffer_type: type) -> None:
    raw = np.array([100.0, 50.0])
    buffer = array("d", raw) if buffer_type is array else buffer_type(raw.tobytes())
    values = np.frombuffer(buffer, dtype=np.float64)
    book = BookState.zero(2, 1000.0, values)
    np.testing.assert_array_equal(book.mark_prices, raw)


def test_book_admits_native_memmap_owner(tmp_path) -> None:
    path = tmp_path / "prices.bin"
    values = np.memmap(path, dtype=np.float64, mode="w+", shape=(2,))
    values[:] = [100.0, 50.0]
    book = BookState.zero(2, 1000.0, values)
    np.testing.assert_array_equal(book.mark_prices, [100.0, 50.0])


class _SpoofedTypeEquality(type):
    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = type.__hash__


@pytest.mark.parametrize("path", ["owner", "scalar", "adapter"])
def test_array_admission_does_not_trust_metaclass_equality(path: str) -> None:
    class Owner(metaclass=_SpoofedTypeEquality):
        def __init__(self) -> None:
            self.raw = np.ma.array([100.0], mask=[True])
            self.__array_interface__ = self.raw.data.__array_interface__

    class Scalar(metaclass=_SpoofedTypeEquality):
        def __float__(self) -> float:
            return 100.0

    owner = Owner()
    if path == "owner":
        value = np.asarray(owner)
        assert value.base is owner
    elif path == "scalar":
        value = [Scalar()]
    else:
        value = owner
    with pytest.raises(ValueError, match="opaque"):
        BookState.zero(1, 1000.0, value)
