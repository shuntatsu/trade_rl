"""Public legacy close-liquidation accounting and capacity safety regression."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def _market(
    *,
    volume: float = 100.0,
    minimum_notional: float = 0.0,
    lot: float = 0.0,
    tick: float = 0.0,
) -> MarketDataset:
    shape = (5, 1)
    prices = np.full(shape, 100.0)
    return MarketDataset(
        dataset_id="9" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(5) * np.timedelta64(1, "h"),
        features=np.zeros((5, 1, 1), dtype=np.float32),
        global_features=np.zeros((5, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices + 1.0,
        low=prices - 1.0,
        close=prices.copy(),
        volume=np.full(shape, volume),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((5, 1, 1), dtype=np.bool_),
        feature_names=("f",),
        global_feature_names=("g",),
        periods_per_year=8_760,
        minimum_notional=np.full(shape, minimum_notional),
        lot_size=np.full(shape, lot),
        tick_size=np.full(shape, tick),
    )


def _book(quantity: float) -> BookState:
    return BookState(
        quantities=np.array([quantity]),
        cash=1_000.0 - quantity * 100.0,
        mark_prices=np.array([100.0]),
        peak_value=1_000.0,
    )


def test_forced_market_legacy_close_uses_taker_fee_and_full_spread() -> None:
    cost = replace(
        ExecutionCostConfig.zero(),
        order_type="limit",
        maker_fee_rate=0.003,
        taker_fee_rate=0.02,
        spread_rate=0.02,
        max_participation_rate=1.0,
    )
    book = _book(10.0)
    result = MarketExecutor(_market(), cost).liquidate_at_close(book, index=1)

    assert result.interval_cost == pytest.approx(40.0)
    assert result.book.cash == pytest.approx(960.0)
    assert result.book.quantities.tolist() == [0.0]
    assert result.termination_reason == EconomicTerminationReason.LIQUIDATION.value
    assert result.fill_ratio == pytest.approx(1.0)
    assert book.quantities.tolist() == [10.0]


def test_minimum_notional_refusal_is_not_reported_as_completed_liquidation() -> None:
    book = _book(0.2)
    result = MarketExecutor(
        _market(minimum_notional=40.0),
        replace(ExecutionCostConfig.zero(), max_participation_rate=1.0),
    ).liquidate_at_close(book, index=1)

    assert result.book.quantities[0] == pytest.approx(0.2)
    assert result.fill_count == 0
    assert result.interval_cost == 0.0
    assert result.termination_reason is None
    assert result.fill_ratio == pytest.approx(0.0)
    assert result.requested_turnover == pytest.approx(0.02)
    assert result.unfilled_turnover == pytest.approx(0.02)
    assert book.quantities[0] == pytest.approx(0.2)


@pytest.mark.parametrize("signed_quantity", [2.1, -2.1])
def test_off_grid_legacy_close_never_rounds_past_actual_liquidity_capacity(
    signed_quantity: float,
) -> None:
    book = _book(signed_quantity)
    result = MarketExecutor(
        _market(volume=1.6, lot=1.0),
        replace(ExecutionCostConfig.zero(), max_participation_rate=1.0),
    ).liquidate_at_close(book, index=1)

    assert result.book.quantities[0] == pytest.approx(
        1.0 if signed_quantity > 0.0 else -1.0
    )
    assert result.filled_turnover == pytest.approx(0.11)
    assert result.unfilled_turnover == pytest.approx(0.1)
    assert result.fill_ratio == pytest.approx(1.1 / 2.1)
    assert result.termination_reason is None
    assert result.max_participation <= 1.0
    assert book.quantities[0] == pytest.approx(signed_quantity)


@pytest.mark.parametrize("signed_quantity", [2.1, -2.1])
def test_unreachable_lot_step_retains_original_off_grid_book(
    signed_quantity: float,
) -> None:
    result = MarketExecutor(
        _market(volume=0.05, lot=1.0),
        replace(ExecutionCostConfig.zero(), max_participation_rate=1.0),
    ).liquidate_at_close(_book(signed_quantity), index=1)
    assert result.book.quantities[0] == pytest.approx(signed_quantity)
    assert result.fill_count == 0
    assert result.termination_reason is None
    assert result.fill_ratio == pytest.approx(0.0)


def test_ordinary_rested_limit_keeps_maker_fee_and_half_spread() -> None:
    dataset = _market()
    cost = replace(
        ExecutionCostConfig.zero(),
        order_type="limit",
        maker_fee_rate=0.003,
        taker_fee_rate=0.02,
        spread_rate=0.02,
        max_participation_rate=1.0,
        limit_offset_rate=0.0,
    )
    executor = MarketExecutor(dataset, cost)
    result = executor._fill_toward_quantities(
        _book(10.0),
        np.zeros(1),
        prices=dataset.close[1],
        capacity_volume=dataset.volume[1],
        tradable=dataset.tradable[1],
        turnover_denominator=1_000.0,
        market_index=1,
        force_market=False,
    )
    assert result.filled_notional == pytest.approx(1_000.0)
    assert result.cost_amount == pytest.approx(13.0)


@pytest.mark.parametrize(
    ("capacity_base", "expected_abs_remaining"),
    [
        (0.05, 2.1),
        (1.6, 1.0),
        (2.0, 1.0),
        (2.1, 0.0),
        (10.0, 0.0),
    ],
)
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_legacy_close_capacity_never_overfills_after_grid_quantization(
    capacity_base: float, expected_abs_remaining: float, direction: float
) -> None:
    dataset = _market(volume=capacity_base, lot=1.0)
    original = 2.1 * direction
    result = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), max_participation_rate=1.0),
    ).liquidate_at_close(_book(original), index=1)

    assert abs(result.book.quantities[0]) == pytest.approx(expected_abs_remaining)
    actual_notional = abs(original - float(result.book.quantities[0])) * 100.0
    assert actual_notional <= capacity_base * 100.0
    assert result.filled_turnover == pytest.approx(actual_notional / 1_000.0)
    assert result.book.quantities[0] * direction >= 0.0
    if expected_abs_remaining:
        assert result.termination_reason is None
    else:
        assert result.termination_reason == EconomicTerminationReason.LIQUIDATION.value


def test_completed_liquidation_ratio_uses_actual_rounded_reference_price() -> None:
    result = MarketExecutor(
        _market(tick=0.3),
        replace(ExecutionCostConfig.zero(), max_participation_rate=1.0),
    ).liquidate_at_close(_book(10.0), index=1)

    assert result.book.exact_quantities == (0,)
    assert result.termination_reason == EconomicTerminationReason.LIQUIDATION.value
    assert result.fill_ratio == pytest.approx(1.0)
    assert result.unfilled_turnover == pytest.approx(0.0)
    assert result.requested_turnover == pytest.approx(result.filled_turnover)
