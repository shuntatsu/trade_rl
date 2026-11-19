"""Stateful account/funding valuation must use BookState's numerical authority."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.simulation.stateful.bar_lifecycle import StatefulBarLifecycle
from trade_rl.simulation.stateful.runtime import StatefulExecutionRuntime


def _subnormal_dataset() -> MarketDataset:
    rows = 3
    shape = (rows, 1)
    prices = np.full(shape, 1e-160)
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(rows) * np.timedelta64(1, "h"),
        features=np.zeros((rows, 1, 1), dtype=np.float32),
        global_features=np.zeros((rows, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.full(shape, 100.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((rows, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        contract_multipliers=np.array([5e-164]),
        volume_units=(VolumeUnit.QUOTE_NOTIONAL,),
    )


@pytest.mark.parametrize("signed_position", (-1.0, 1.0))
def test_stateful_terminal_mark_uses_strict_safe_subnormal_valuation(
    signed_position: float,
) -> None:
    dataset = _subnormal_dataset()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState(
        quantities=np.array([signed_position]),
        cash=1_000.0,
        mark_prices=dataset.close[0],
        peak_value=1_000.0,
        contract_multipliers=dataset.contract_multipliers,
    )
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = executor.execute_orders(
            book, OrderBookState.empty(), (), start_index=0, bars=1
        )
        assert result.book.position_values[0] == signed_position * 5e-324
        assert result.book.portfolio_value == 1_000.0
    assert result.book.quantities.tolist() == [signed_position]
    assert result.book.cash == 1_000.0
    assert result.book.fill_count == 0
    assert result.book.returns_history == [0.0]
    assert book.quantities.tolist() == [signed_position]
    assert book.cash == 1_000.0


@pytest.mark.parametrize("strict", (False, True))
def test_mutated_active_position_valuation_rejects_underflow_to_zero(
    strict: bool,
) -> None:
    book = BookState.zero(
        1,
        1_000.0,
        np.array([1e-200]),
        contract_multipliers=np.array([1e-200]),
    )
    book.quantities[0] = 1e-200
    with np.errstate(all="raise" if strict else "ignore"):
        with pytest.raises(ValueError, match="nonzero position value"):
            _ = book.position_values
        with pytest.raises(ValueError, match="nonzero position value"):
            _ = book.portfolio_value
    assert book.quantities.tolist() == [1e-200]
    assert book.cash == 1_000.0


def test_mutated_cash_and_position_cannot_silently_report_infinite_equity() -> None:
    book = BookState.zero(1, 1_000.0, np.ones(1))
    book.quantities[0] = 1e308
    book.cash = 1e308
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError, match="portfolio value became non-finite"):
            _ = book.portfolio_value
    assert book.cash == 1e308
    assert book.quantities.tolist() == [1e308]


@pytest.mark.parametrize("quantity", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("strict", (False, True))
def test_mutated_nonfinite_position_read_fails_closed(
    quantity: float, strict: bool
) -> None:
    book = BookState.zero(1, 1_000.0, np.ones(1))
    book.quantities[0] = quantity
    with np.errstate(all="raise" if strict else "ignore"):
        with pytest.raises(
            ValueError, match="proposed account state became non-finite"
        ):
            _ = book.position_values
        with pytest.raises(ValueError):
            _ = book.portfolio_value
    assert book.cash == 1_000.0


@pytest.mark.parametrize("signed_position", (-1.0, 1.0))
def test_vectorized_multiasset_subnormal_read_matches_scalar_oracle(
    signed_position: float,
) -> None:
    n = 9  # Exercise the multiasset vector path, not the small-roster path.
    book = BookState.zero(
        n,
        1_000.0,
        np.full(n, 1e-160),
        contract_multipliers=np.full(n, 5e-164),
    )
    book.quantities[0] = signed_position
    with np.errstate(all="raise"):
        assert book.position_values[0] == signed_position * 5e-324
        assert book.portfolio_value == 1_000.0

    # A nonzero second holding must not vanish to 0 and be treated as FLAT.
    book.quantities[1] = 1e-160
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="nonzero position value"):
            _ = book.position_values


@pytest.mark.parametrize("quantity", (float("nan"), float("inf"), float("-inf")))
def test_multiasset_mutated_nonfinite_position_read_fails_closed(
    quantity: float,
) -> None:
    book = BookState.zero(9, 1_000.0, np.ones(9))
    book.quantities[8] = quantity
    with np.errstate(all="raise"):
        with pytest.raises(
            ValueError, match="proposed account state became non-finite"
        ):
            _ = book.position_values


@pytest.mark.parametrize("n_symbols", (1, 9))
def test_valuation_preserves_left_associated_binary64_order(
    n_symbols: int,
) -> None:
    book = BookState.zero(n_symbols, 1_000.0, np.ones(n_symbols))
    book.quantities[0] = 1e308
    book.mark_prices[0] = 1e-200
    book.contract_multipliers[0] = 1e-200
    # q * mark * multiplier survives; q * (mark * multiplier) vanishes.
    expected = (1e308 * 1e-200) * 1e-200
    assert expected > 0.0
    with np.errstate(all="raise"):
        assert book.position_values[0] == expected
    # Reversing the economic factors makes the *first* binary64 multiply
    # underflow; regrouping would incorrectly invent the positive exposure.
    book.quantities[0] = 1e-200
    book.mark_prices[0] = 1e-200
    book.contract_multipliers[0] = 1e308
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="nonzero position value"):
            _ = book.position_values


@pytest.mark.parametrize("n_symbols", (1, 9))
def test_post_admission_array_shape_change_fails_closed(n_symbols: int) -> None:
    book = BookState.zero(n_symbols, 1_000.0, np.ones(n_symbols))
    book.mark_prices = np.ones(n_symbols + 1)
    with pytest.raises(ValueError, match="matching vector shapes"):
        _ = book.position_values


@pytest.mark.parametrize("n_symbols", (1, 9))
def test_post_admission_float32_arrays_cannot_gain_precision(
    n_symbols: int,
) -> None:
    book = BookState.zero(n_symbols, 1_000.0, np.ones(n_symbols))
    book.quantities = np.full(n_symbols, 1e-18, dtype=np.float32)
    book.mark_prices = np.full(n_symbols, 1e-18, dtype=np.float32)
    book.contract_multipliers = np.full(n_symbols, 1e-18, dtype=np.float32)
    # Even if the float32 products vanish, reject dtype drift *before*
    # arithmetic. Another set of float32 values may be representable.
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="non-float64"):
            _ = book.position_values


def test_post_admission_empty_account_arrays_do_not_hide_holdings() -> None:
    book = BookState.zero(1, 1_000.0, np.ones(1))
    book.quantities = np.empty(0, dtype=np.float64)
    book.mark_prices = np.empty(0, dtype=np.float64)
    book.contract_multipliers = np.empty(0, dtype=np.float64)
    with pytest.raises(ValueError, match="matching nonempty vector shapes"):
        _ = book.position_values


@pytest.mark.parametrize("signed_position", (-1.0, 1.0))
def test_subnormal_portfolio_equity_retains_actual_position_value(
    signed_position: float,
) -> None:
    """Equity must distinguish a subnormal holding from a FLAT account."""
    unit = float(np.nextafter(0.0, 1.0))
    book = BookState(
        quantities=np.array([signed_position]),
        cash=3.0 * unit,
        mark_prices=np.array([1e-160]),
        contract_multipliers=np.array([5e-164]),
        peak_value=5.0 * unit,
    )
    with np.errstate(all="raise"):
        assert book.position_values[0] == signed_position * unit
        assert book.portfolio_value == (3.0 + signed_position) * unit


@pytest.mark.parametrize("signed_position", (-1.0, 1.0))
def test_subnormal_nonzero_funding_settles_under_strict_numpy(
    signed_position: float,
) -> None:
    dataset = replace(
        _subnormal_dataset(),
        funding_rate=np.ones((3, 1), dtype=np.float64),
    )
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState(
        quantities=np.array([signed_position]),
        cash=1_000.0,
        mark_prices=dataset.close[0],
        peak_value=1_000.0,
        contract_multipliers=dataset.contract_multipliers,
    )
    unit = float(np.nextafter(0.0, 1.0))
    with np.errstate(all="raise"):
        result = executor.execute_orders(
            book, OrderBookState.empty(), (), start_index=0, bars=1
        )
        assert result.book.position_values[0] == signed_position * unit
        assert result.book.funding_pnl == -signed_position * unit
        assert result.interval_funding == -signed_position * unit


@pytest.mark.parametrize("signed_position", (-1.0, 1.0))
def test_funding_book_write_failure_does_not_count_unsettled_funding(
    signed_position: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No funding aggregate can advance before the corresponding book credit."""
    dataset = replace(
        _subnormal_dataset(),
        funding_rate=np.ones((3, 1), dtype=np.float64),
    )
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState(
        quantities=np.array([signed_position]),
        cash=1_000.0,
        mark_prices=dataset.close[0],
        peak_value=1_000.0,
        contract_multipliers=dataset.contract_multipliers,
    )
    runtime = StatefulExecutionRuntime.create(executor, book, OrderBookState.empty())
    lifecycle = StatefulBarLifecycle(executor)
    context = lifecycle.begin_bar(runtime, previous_index=0, processing_index=1)
    before_cash = runtime.book.cash
    before_funding = runtime.book.funding_pnl
    calls: list[float] = []

    def injected_mark_failure(
        _book: BookState, *, funding_amount: float, **_kwargs: object
    ) -> float:
        calls.append(funding_amount)
        raise ValueError("synthetic book funding write refused")

    monkeypatch.setattr(BookState, "mark_to_market", injected_mark_failure)
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="synthetic book funding write refused"):
            lifecycle.finish_bar(runtime, context)
    assert calls == [-signed_position * float(np.nextafter(0.0, 1.0))]
    assert runtime.total_funding == 0.0
    assert runtime.book.funding_pnl == before_funding
    assert runtime.book.cash == before_cash
    assert runtime.funding_evidence == []


@pytest.mark.parametrize("n_symbols", (1, 9))
def test_integer_mutation_cannot_wrap_account_notional(
    n_symbols: int,
) -> None:
    book = BookState.zero(n_symbols, 1_000.0, np.ones(n_symbols))
    book.quantities = np.full(n_symbols, np.iinfo(np.int64).max, dtype=np.int64)
    book.mark_prices = np.full(n_symbols, 2, dtype=np.int64)
    book.contract_multipliers = np.ones(n_symbols, dtype=np.int64)
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="non-float64"):
            _ = book.position_values
        with pytest.raises(ValueError, match="non-float64"):
            _ = book.portfolio_value


def test_broadcastable_funding_mark_shape_refused_before_counting_payment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed mark view must not be accepted because NumPy broadcasts it."""
    dataset = _subnormal_dataset()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState(
        quantities=np.array([1.0]),
        cash=1_000.0,
        mark_prices=dataset.close[0],
        peak_value=1_000.0,
        contract_multipliers=dataset.contract_multipliers,
    )
    runtime = StatefulExecutionRuntime.create(executor, book, OrderBookState.empty())
    lifecycle = StatefulBarLifecycle(executor)
    context = lifecycle.begin_bar(runtime, previous_index=0, processing_index=1)
    observed_mark_requests: list[str] = []
    actual_resolved = MarketDataset.resolved_array

    def forged_mark(self: MarketDataset, name: str) -> np.ndarray:
        if name == "mark_price":
            observed_mark_requests.append(name)
            return np.full((3, 1, 1), 1e-160)
        return actual_resolved(self, name)

    monkeypatch.setattr(MarketDataset, "resolved_array", forged_mark)
    monkeypatch.setattr(
        executor,
        "_charge_carry",
        lambda *_args, **_kwargs: (-1.0, 0.0),
    )
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="pre-funding account valuation"):
            lifecycle.finish_bar(runtime, context)
    assert observed_mark_requests
    assert runtime.total_funding == 0.0
    assert runtime.book.funding_pnl == 0.0
    assert runtime.funding_evidence == []


@pytest.mark.parametrize("n_symbols", (1, 9))
@pytest.mark.parametrize("field", ("quantities", "mark_prices", "contract_multipliers"))
@pytest.mark.parametrize("dtype", (np.float32, np.int64, np.uint64))
def test_single_post_admission_dtype_replacement_is_rejected(
    n_symbols: int, field: str, dtype: type[np.generic]
) -> None:
    book = BookState.zero(n_symbols, 1_000.0, np.ones(n_symbols))
    setattr(book, field, np.ones(n_symbols, dtype=dtype))
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="non-float64"):
            _ = book.position_values
        with pytest.raises(ValueError, match="non-float64"):
            _ = book.portfolio_value


def test_mutated_missing_multiplier_read_fails_closed() -> None:
    book = BookState.zero(1, 1_000.0, np.ones(1))
    book.contract_multipliers = None
    with pytest.raises(ValueError, match="contract_multipliers"):
        _ = book.position_values
    with pytest.raises(ValueError, match="contract_multipliers"):
        _ = book.portfolio_value


def test_invalid_next_mark_rejected_before_real_borrow_is_charged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad process-bar marks must not leave real borrow debited on failure."""
    rows = 3
    shape = (rows, 1)
    prices = np.full(shape, 100.0)
    dataset = replace(
        _subnormal_dataset(),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        mark_price=prices.copy(),
        contract_multipliers=np.ones(1),
        borrow_rate=np.full(shape, 0.5),
        funding_price_rate=np.zeros(shape),
    )
    executor = MarketExecutor(
        dataset, replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0)
    )
    book = BookState(
        quantities=np.array([-1.0]),
        cash=1_000.0,
        mark_prices=prices[0],
        peak_value=1_000.0,
        contract_multipliers=np.ones(1),
    )
    runtime = StatefulExecutionRuntime.create(executor, book, OrderBookState.empty())
    lifecycle = StatefulBarLifecycle(executor)
    context = lifecycle.begin_bar(runtime, previous_index=0, processing_index=1)
    cash_before = runtime.book.cash
    marks_before = runtime.book.mark_prices.copy()
    borrow_before = runtime.book.borrow_cost
    original = MarketDataset.resolved_array

    def malformed_mark(self: MarketDataset, name: str) -> np.ndarray:
        if name == "mark_price":
            return np.full((rows, 1, 1), 100.0)
        return original(self, name)

    monkeypatch.setattr(MarketDataset, "resolved_array", malformed_mark)

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="pre-funding account valuation"):
            lifecycle.finish_bar(runtime, context)
    assert runtime.total_borrow == 0.0
    assert runtime.book.borrow_cost == borrow_before
    assert runtime.book.cash == cash_before
    np.testing.assert_array_equal(runtime.book.mark_prices, marks_before)
    assert runtime.total_funding == 0.0
    assert runtime.funding_evidence == []
