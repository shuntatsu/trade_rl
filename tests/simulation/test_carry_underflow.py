from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def _dataset(
    *,
    mark: float = 1.0,
    multiplier: float = 1.0,
    funding_rate: float = 0.0,
    funding_price_rate: float | None = None,
    borrow_rate: float = 0.0,
) -> MarketDataset:
    shape = (3, 1)
    prices = np.full(shape, mark, dtype=np.float64)
    rates = np.zeros(shape, dtype=np.float64)
    rates[1, 0] = funding_rate
    due = np.zeros(shape, dtype=np.bool_)
    due[1, 0] = True
    event_price = None
    if funding_price_rate is not None:
        event_price = np.zeros(shape, dtype=np.float64)
        event_price[1, 0] = funding_price_rate
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(3) * np.timedelta64(1, "h"),
        features=np.zeros((3, 1, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.full(shape, 1_000.0),
        funding_rate=rates,
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=8_760,
        funding_due=due,
        mark_price=prices.copy(),
        contract_multipliers=np.array([multiplier]),
        borrow_rate=np.full(shape, borrow_rate),
        funding_price_rate=event_price,
    )


def _book(*, quantity: float, cash: float, mark: float, multiplier: float) -> BookState:
    return BookState(
        quantities=np.array([quantity]),
        cash=cash,
        mark_prices=np.array([mark]),
        peak_value=max(cash + quantity * mark * multiplier, 1e-300),
        contract_multipliers=np.array([multiplier]),
    )


def _assert_unchanged(book: BookState, before: BookState) -> None:
    assert book.cash == before.cash
    assert book.funding_pnl == before.funding_pnl
    assert book.borrow_cost == before.borrow_cost
    assert book.exact_quantities == before.exact_quantities
    np.testing.assert_array_equal(book.quantities, before.quantities)
    np.testing.assert_array_equal(book.mark_prices, before.mark_prices)


def test_mark_price_funding_rejects_nonzero_underflow_before_mutation() -> None:
    dataset = _dataset(mark=1e-100, multiplier=1e-100, funding_rate=1e-200)
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = _book(quantity=1.0, cash=1_000.0, mark=1e-100, multiplier=1e-100)
    before = book.clone()
    with np.errstate(over="raise", under="raise", invalid="raise"):
        with pytest.raises(ValueError, match="funding.*represented"):
            executor._charge_carry(book, index=1, year_fraction=0.0)
    _assert_unchanged(book, before)


def test_event_price_funding_rejects_nonzero_underflow_before_mutation() -> None:
    dataset = _dataset(mark=100.0, multiplier=1e-200, funding_price_rate=1e-200)
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = _book(quantity=1.0, cash=1_000.0, mark=100.0, multiplier=1e-200)
    before = book.clone()
    with np.errstate(over="raise", under="raise", invalid="raise"):
        with pytest.raises(ValueError, match="funding.*represented"):
            executor._charge_carry(book, index=1, year_fraction=0.0)
    _assert_unchanged(book, before)


def test_borrow_rejects_nonzero_underflow_before_mutation() -> None:
    dataset = _dataset(mark=1e-100, multiplier=1e-100, borrow_rate=1e-200)
    executor = MarketExecutor(
        dataset, replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0)
    )
    book = _book(quantity=-1.0, cash=1_000.0, mark=1e-100, multiplier=1e-100)
    before = book.clone()
    with np.errstate(over="raise", under="raise", invalid="raise"):
        with pytest.raises(ValueError, match="borrow.*represented"):
            executor._charge_borrow(book, index=1, year_fraction=1.0)
    _assert_unchanged(book, before)


def test_dividend_rejects_nonzero_underflow_before_mutation() -> None:
    book = _book(quantity=1.0, cash=1_000.0, mark=1.0, multiplier=1e-200)
    before = book.clone()
    with np.errstate(over="raise", under="raise", invalid="raise"):
        with pytest.raises(ValueError, match="dividend.*represented"):
            book.apply_dividend(np.array([1e-200]))
    _assert_unchanged(book, before)


def test_cash_interest_rejects_nonzero_underflow_before_mutation() -> None:
    book = BookState.zero(1, 1e-200, np.array([1.0]))
    before = book.clone()
    with pytest.raises(ValueError, match="cash interest.*represented"):
        book.apply_cash_interest(1e-200, year_fraction=1.0)
    _assert_unchanged(book, before)


def test_dividend_cancellation_to_exact_zero_remains_valid() -> None:
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1_000.0,
        mark_prices=np.array([1.0, 1.0]),
        peak_value=1_000.0,
        contract_multipliers=np.ones(2),
    )
    before = book.clone()
    assert book.apply_dividend(np.array([0.25, 0.25])) == 0.0
    _assert_unchanged(book, before)


def test_zero_rate_cash_interest_is_identity() -> None:
    book = BookState.zero(1, 1e-200, np.array([1.0]))
    before = book.clone()
    assert book.apply_cash_interest(0.0, year_fraction=1.0) == 0.0
    _assert_unchanged(book, before)


def test_funding_cancellation_to_exact_zero_remains_valid() -> None:
    shape = (3, 2)
    prices = np.ones(shape)
    rates = np.zeros(shape)
    rates[1] = np.array([0.25, 0.25])
    due = np.zeros(shape, dtype=np.bool_)
    due[1] = True
    dataset = MarketDataset(
        dataset_id="a" * 64,
        symbols=("L", "S"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(3) * np.timedelta64(1, "h"),
        features=np.zeros((3, 2, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.full(shape, 1_000.0),
        funding_rate=rates,
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((3, 2, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=8_760,
        funding_due=due,
        mark_price=prices.copy(),
    )
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=1_000.0,
        mark_prices=np.ones(2),
        peak_value=1_000.0,
    )
    before = book.clone()
    funding, borrow = executor._charge_carry(book, index=1, year_fraction=0.0)
    assert funding == 0.0
    assert borrow == 0.0
    _assert_unchanged(book, before)


@pytest.mark.parametrize(
    ("rate", "year_fraction", "multiplier"),
    [
        (0.0, 1.0, 1.0),
        (1e-5, 0.0, 1.0),
        (1e-5, 1.0, 0.0),
    ],
)
def test_borrow_subnormal_position_zero_effect_remains_identity_under_strict_numpy(
    rate: float, year_fraction: float, multiplier: float
) -> None:
    dataset = _dataset(mark=1e-200, multiplier=1e-120, borrow_rate=rate)
    executor = MarketExecutor(
        dataset, replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=multiplier)
    )
    book = _book(quantity=-1.0, cash=1_000.0, mark=1e-200, multiplier=1e-120)
    before = book.clone()
    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert (
            executor._charge_borrow(book, index=1, year_fraction=year_fraction) == 0.0
        )
    _assert_unchanged(book, before)


def test_borrow_subnormal_position_underflow_fails_closed_under_strict_numpy() -> None:
    dataset = _dataset(mark=1e-200, multiplier=1e-120, borrow_rate=1e-5)
    executor = MarketExecutor(
        dataset, replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0)
    )
    book = _book(quantity=-1.0, cash=1_000.0, mark=1e-200, multiplier=1e-120)
    before = book.clone()
    with np.errstate(over="raise", under="raise", invalid="raise"):
        with pytest.raises(ValueError, match="borrow.*represented"):
            executor._charge_borrow(book, index=1, year_fraction=1.0)
    _assert_unchanged(book, before)


@pytest.mark.parametrize("quantity", [-1.0, 1.0])
def test_valid_subnormal_position_queries_work_under_strict_numpy(
    quantity: float,
) -> None:
    book = _book(quantity=quantity, cash=1_000.0, mark=1e-200, multiplier=1e-120)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        positions = book.position_values
        portfolio = book.portfolio_value
    assert positions[0] != 0.0
    assert np.sign(positions[0]) == np.sign(quantity)
    assert portfolio == 1_000.0


@pytest.mark.parametrize("operation", ["dividend", "cash_interest"])
def test_valid_subnormal_position_zero_carry_is_identity_under_strict_numpy(
    operation: str,
) -> None:
    book = _book(quantity=1.0, cash=1_000.0, mark=1e-200, multiplier=1e-120)
    before = book.clone()
    with np.errstate(over="raise", under="raise", invalid="raise"):
        amount = (
            book.apply_dividend(np.array([0.0]))
            if operation == "dividend"
            else book.apply_cash_interest(0.0, year_fraction=1.0)
        )
    assert amount == 0.0
    _assert_unchanged(book, before)


def test_valid_subnormal_position_nonzero_cash_interest_applies_atomically() -> None:
    book = _book(quantity=1.0, cash=1_000.0, mark=1e-200, multiplier=1e-120)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        amount = book.apply_cash_interest(0.01, year_fraction=1.0)
    assert amount == 10.0
    assert book.cash == 1_010.0


def test_valid_subnormal_position_nonzero_dividend_applies_atomically() -> None:
    book = _book(quantity=1.0, cash=1_000.0, mark=1e-200, multiplier=1e-120)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        amount = book.apply_dividend(np.array([1e121]))
    assert amount == pytest.approx(10.0)
    assert book.cash == pytest.approx(1_010.0)


@pytest.mark.parametrize("signed_quantity", [-1.0, 1.0])
@pytest.mark.parametrize("strict_underflow", [False, True])
def test_mutated_unrepresentable_position_query_fails_closed(
    signed_quantity: float, strict_underflow: bool
) -> None:
    book = _book(quantity=signed_quantity, cash=1_000.0, mark=1e-200, multiplier=1e-120)
    book.contract_multipliers[:] = 1e-200
    with np.errstate(
        over="raise", under="raise" if strict_underflow else "ignore", invalid="raise"
    ):
        with pytest.raises(ValueError, match="position value cannot be represented"):
            _ = book.position_values


@pytest.mark.parametrize("strict_underflow", [False, True])
def test_unrepresentable_leg_cannot_hide_behind_other_valid_position(
    strict_underflow: bool,
) -> None:
    book = BookState(
        quantities=np.array([1.0, 1.0]),
        cash=1_000.0,
        mark_prices=np.array([1.0, 1e-200]),
        peak_value=1_001.0,
        contract_multipliers=np.array([1.0, 1e-120]),
    )
    book.contract_multipliers[1] = 1e-200
    with np.errstate(
        over="raise", under="raise" if strict_underflow else "ignore", invalid="raise"
    ):
        with pytest.raises(ValueError, match="position value cannot be represented"):
            _ = book.portfolio_value


def test_borrow_does_not_silently_erase_poisoned_nonzero_short() -> None:
    dataset = _dataset(mark=1e-200, multiplier=1e-120, borrow_rate=1e-5)
    executor = MarketExecutor(
        dataset, replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0)
    )
    book = _book(quantity=-1.0, cash=1_000.0, mark=1e-200, multiplier=1e-120)
    book.contract_multipliers[:] = 1e-200
    before_cash = book.cash
    before_borrow_cost = book.borrow_cost
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        with pytest.raises(ValueError, match="position value cannot be represented"):
            executor._charge_borrow(book, index=1, year_fraction=1.0)
    assert book.cash == before_cash
    assert book.borrow_cost == before_borrow_cost
