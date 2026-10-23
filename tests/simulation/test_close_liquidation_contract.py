from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def _dataset(
    *,
    volume: float = 1_000_000.0,
    tradable: bool = True,
    lot_size: float = 0.0,
    minimum_notional: float = 0.0,
    n_symbols: int = 1,
    volumes: tuple[float, ...] | None = None,
    volume_units: tuple[VolumeUnit, ...] = (),
    contract_multipliers: tuple[float, ...] | None = None,
) -> MarketDataset:
    n_bars = 3
    shape = (n_bars, n_symbols)
    close = np.full(shape, 100.0)
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=tuple(f"S{index}" for index in range(n_symbols)),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=(
            np.full(shape, volume)
            if volumes is None
            else np.tile(np.asarray(volumes, dtype=np.float64), (n_bars, 1))
        ),
        funding_rate=np.zeros(shape),
        tradable=np.full(shape, tradable, dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        lot_size=np.full(shape, lot_size),
        minimum_notional=np.full(shape, minimum_notional),
        volume_units=volume_units,
        contract_multipliers=(
            None
            if contract_multipliers is None
            else np.asarray(contract_multipliers, dtype=np.float64)
        ),
    )


def _book(quantity: float, *, capital: float, price: float = 100.0) -> BookState:
    book = BookState.zero(1, capital, np.array([price]))
    if quantity:
        book.execute_fill(
            symbol_index=0,
            quantity=quantity,
            fill_prices=np.array([price]),
            valuation_prices=np.array([price]),
            cost_amount=0.0,
            turnover=0.0,
        )
    return book


def test_close_forces_taker_fee_and_full_spread_for_limit_profile() -> None:
    dataset = _dataset()
    config = replace(
        ExecutionCostConfig.zero(),
        maker_fee_rate=0.003,
        taker_fee_rate=0.02,
        spread_rate=0.02,
        max_participation_rate=1.0,
        order_type="limit",
    )

    result = MarketExecutor(dataset, config).liquidate_at_close(
        _book(10.0, capital=1_000.0), index=1
    )

    assert result.book.exact_quantities == (Fraction(0),)
    assert result.interval_cost == pytest.approx(40.0)
    np.testing.assert_allclose(result.cost_by_symbol, [40.0])
    np.testing.assert_allclose(result.filled_notional_by_symbol, [1_000.0])
    assert result.book.cash == pytest.approx(960.0)
    assert result.termination_reason == EconomicTerminationReason.LIQUIDATION.value
    assert result.next_index == 1
    assert result.bars_advanced == 0


def test_whole_lot_subminimum_close_remains_unfilled_and_not_liquidated() -> None:
    dataset = _dataset(lot_size=0.1, minimum_notional=40.0)
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())

    result = executor.liquidate_at_close(_book(0.2, capital=1_000.0), index=1)
    empty = executor.liquidate_at_close(
        BookState.zero(1, 1_000.0, dataset.close[0]), index=1
    )

    assert result.book.exact_quantities == (Fraction(1, 5),)
    np.testing.assert_allclose(result.requested_notional_by_symbol, [20.0])
    np.testing.assert_allclose(result.filled_notional_by_symbol, [0.0])
    assert result.requested_turnover == pytest.approx(0.02)
    assert result.fill_ratio == pytest.approx(0.0)
    assert result.fill_count == 0
    assert result.termination_reason is None
    assert result.book.cash == pytest.approx(980.0)
    assert empty.fill_ratio == pytest.approx(1.0)


def test_full_capacity_off_grid_close_fills_only_whole_lots_in_one_batch() -> None:
    dataset = _dataset(volume=2.1, lot_size=1.0, n_symbols=2)
    initial = BookState(
        quantities=np.array([2.1, -2.1]),
        cash=1_000_000.0,
        mark_prices=np.array([100.0, 100.0]),
        peak_value=1_000_000.0,
    )

    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).liquidate_at_close(
        initial, index=1
    )

    assert result.book.exact_quantities == (Fraction(1, 10), Fraction(-1, 10))
    np.testing.assert_allclose(result.requested_notional_by_symbol, [210.0, 210.0])
    np.testing.assert_allclose(result.filled_notional_by_symbol, [200.0, 200.0])
    assert result.fill_count == 2
    assert result.rebalance_events == 1
    assert result.termination_reason is None


def test_unlotted_full_close_flattens_hidden_exact_tails_in_one_batch() -> None:
    quantities = np.array([1.2345678901234567, -2.345678901234567])
    marks = np.full(2, 100.0)
    book = BookState(
        quantities=quantities,
        cash=1_000_000.0 - float(np.sum(quantities * marks)),
        mark_prices=marks,
        peak_value=1_000_000.0,
    )
    book.apply_split(np.full(2, 1.1))
    assert any(
        Fraction(str(float(projected))) != exact
        for exact, projected in zip(book.exact_quantities, book.quantities, strict=True)
    )

    result = MarketExecutor(
        _dataset(n_symbols=2), ExecutionCostConfig.zero()
    ).liquidate_at_close(book, index=1)

    assert result.book.exact_quantities == (Fraction(0), Fraction(0))
    assert result.fill_count == 2
    assert result.rebalance_events == 1
    assert result.termination_reason == EconomicTerminationReason.LIQUIDATION.value


def test_close_respects_quote_and_contract_volume_units() -> None:
    dataset = _dataset(
        n_symbols=2,
        volumes=(5.0, 100.0),
        volume_units=(VolumeUnit.CONTRACTS, VolumeUnit.QUOTE_NOTIONAL),
        contract_multipliers=(0.1, 1.0),
    )
    initial = BookState(
        quantities=np.array([10.0, 1.0]),
        cash=1_000_000.0,
        mark_prices=np.array([100.0, 100.0]),
        peak_value=1_000_200.0,
        contract_multipliers=np.array([0.1, 1.0]),
    )

    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).liquidate_at_close(
        initial, index=1
    )

    assert result.book.exact_quantities == (Fraction(5), Fraction(0))
    np.testing.assert_allclose(result.requested_notional_by_symbol, [100.0, 100.0])
    np.testing.assert_allclose(result.filled_notional_by_symbol, [50.0, 100.0])
    assert result.fill_count == 2
    assert result.rebalance_events == 1


def test_close_cost_exhaustion_reason_survives_flattening() -> None:
    dataset = _dataset()
    config = replace(
        ExecutionCostConfig.zero(),
        fee_rate=1.0,
        max_participation_rate=1.0,
    )

    result = MarketExecutor(dataset, config).liquidate_at_close(
        _book(10.0, capital=1_000.0), index=1
    )

    assert result.book.exact_quantities == (Fraction(0),)
    assert result.book.insolvent
    assert (
        result.termination_reason
        == EconomicTerminationReason.EXECUTION_COST_EXHAUSTION.value
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_partial_lot_close_fills_only_capacity_and_preserves_exact_remainder(
    sign: int,
) -> None:
    dataset = _dataset(volume=1.6, lot_size=1.0)
    config = replace(
        ExecutionCostConfig.zero(),
        fee_rate=0.01,
        max_participation_rate=1.0,
    )
    initial = _book(sign * 2.1, capital=1_000_000.0)

    result = MarketExecutor(dataset, config).liquidate_at_close(initial, index=1)

    assert result.book.exact_quantities == (Fraction(sign * 11, 10),)
    np.testing.assert_allclose(result.requested_notional_by_symbol, [210.0])
    np.testing.assert_allclose(result.filled_notional_by_symbol, [100.0])
    np.testing.assert_allclose(result.cost_by_symbol, [1.0])
    assert result.requested_turnover == pytest.approx(210.0 / 1_000_000.0)
    assert result.filled_turnover == pytest.approx(100.0 / 1_000_000.0)
    assert result.fill_ratio == pytest.approx(100.0 / 210.0)
    assert result.interval_cost == pytest.approx(1.0)
    assert result.book.total_cost == pytest.approx(1.0)
    assert result.book.cash == pytest.approx(initial.cash + sign * 100.0 - 1.0)
    assert result.book.portfolio_value == pytest.approx(999_999.0)
    assert result.termination_reason is None


@pytest.mark.parametrize(
    ("volume", "tradable"),
    [(0.0, True), (1.6, False)],
    ids=["zero-capacity", "nontradable"],
)
def test_unfillable_off_grid_position_is_not_changed_by_lot_rounding(
    volume: float, tradable: bool
) -> None:
    dataset = _dataset(volume=volume, tradable=tradable, lot_size=1.0)
    initial = _book(2.1, capital=1_000_000.0)

    result = MarketExecutor(dataset, ExecutionCostConfig.zero()).liquidate_at_close(
        initial, index=1
    )

    assert result.book.exact_quantities == (Fraction(21, 10),)
    np.testing.assert_allclose(result.requested_notional_by_symbol, [210.0])
    np.testing.assert_allclose(result.filled_notional_by_symbol, [0.0])
    assert result.fill_count == 0
    assert result.book.fill_count == initial.fill_count
    assert result.book.cash == pytest.approx(initial.cash)
    assert result.book.total_cost == pytest.approx(initial.total_cost)
    assert result.termination_reason is None
