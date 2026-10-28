from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    TimeInForce,
)


def _market(
    *,
    gap_before_insolvency: bool = False,
    insolvency_price: float = 400.0,
    insolvency_borrow_rate: float = 0.5,
    processing_cash_rate: float = 0.0,
) -> MarketDataset:
    close = np.array(
        [[100.0], [100.0], [insolvency_price], [insolvency_price], [insolvency_price]]
    )
    shape = close.shape
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(
        close.shape[0]
    ) * np.timedelta64(1, "h")
    cash_rate = np.zeros(shape[0])
    if gap_before_insolvency:
        timestamps = np.datetime64("2026-01-01", "ns") + np.array(
            [0, 1, 8_761, 8_762, 8_763]
        ) * np.timedelta64(1, "h")
        cash_rate[2] = -2.0
    else:
        cash_rate[2] = processing_cash_rate
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTC",),
        timestamps=timestamps,
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((shape[0], 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.array([[0.0], [0.0], [0.1], [0.2], [0.2]]),
        funding_due=np.array([[False], [False], [True], [True], [True]]),
        borrow_rate=np.array([[0.0], [0.0], [insolvency_borrow_rate], [1.0], [1.0]]),
        cash_rate=cash_rate,
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((*shape, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        calendar_kind="session_calendar"
        if gap_before_insolvency
        else "continuous_24_7",
        nominal_bar_hours=1.0 if gap_before_insolvency else None,
    )


def _executor(
    dataset: MarketDataset,
    *,
    insolvency_valuation: str = "floor_zero",
    fee_rate: float = 0.0,
    borrow_rate_multiplier: float = 0.0,
) -> MarketExecutor:
    cost = replace(
        ExecutionCostConfig.zero(),
        allow_short=True,
        borrow_rate_multiplier=borrow_rate_multiplier,
        fee_rate=fee_rate,
        max_leverage=2.0,
        maintenance_margin_rate=0.0,
        processing_bar_volume_capacity=True,
    )
    return MarketExecutor(
        dataset,
        cost,
        insolvency_valuation=insolvency_valuation,
    )


def _intent(
    executor: MarketExecutor,
    *,
    quantity: float,
    target_identity: str,
) -> OrderIntent:
    return OrderIntent.create(
        dataset_id=executor.dataset.dataset_id,
        target_identity=target_identity,
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )


def _short_intent(executor: MarketExecutor) -> OrderIntent:
    return _intent(
        executor,
        quantity=-5.0,
        target_identity="insolvency-valuation-short",
    )


def _open_short(executor: MarketExecutor):
    dataset = executor.dataset
    return executor.execute_orders(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        OrderBookState.empty(),
        (_short_intent(executor),),
        start_index=0,
        bars=1,
    )


def test_floor_zero_is_the_legacy_default_and_preserves_its_policy_digest() -> None:
    dataset = _market()
    cost = replace(ExecutionCostConfig.zero(), allow_short=True, max_leverage=2.0)
    executor = MarketExecutor(dataset, cost)

    assert executor.insolvency_valuation == "floor_zero"
    assert executor.execution_policy_digest == cost.execution_policy_digest


@pytest.mark.parametrize("invalid", [None, True, "floor", "retain", ""])
def test_insolvency_valuation_rejects_unsupported_values(invalid: object) -> None:
    with pytest.raises(ValueError, match="insolvency_valuation"):
        MarketExecutor(_market(), insolvency_valuation=invalid)


def test_retain_debt_policy_digest_separates_intent_identity() -> None:
    dataset = _market()
    legacy = _executor(dataset)
    retained = _executor(dataset, insolvency_valuation="retain_debt")

    assert retained.execution_policy_digest != legacy.execution_policy_digest


def test_opposite_mode_intent_is_rejected_before_any_fill() -> None:
    dataset = _market()
    legacy = _executor(dataset)
    retained = _executor(dataset, insolvency_valuation="retain_debt")

    result = retained.execute_orders(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        OrderBookState.empty(),
        (_short_intent(legacy),),
        start_index=0,
        bars=1,
    )

    assert result.fill_count == 0
    assert result.book.quantities.tolist() == [0.0]
    assert len(result.order_events) == 2
    assert result.order_events[-1].event_type == "rejected"
    assert result.order_events[-1].reason == "identity_mismatch"


def test_retain_debt_stops_at_first_insolvency_without_later_carry_or_bars() -> None:
    dataset = _market()
    executor = _executor(dataset, insolvency_valuation="retain_debt")
    opened = _open_short(executor)

    assert opened.book.quantities.tolist() == [-5.0]
    assert opened.book.cash == pytest.approx(1_500.0)
    assert opened.book.portfolio_value == pytest.approx(1_000.0)

    result = executor.execute_orders(
        opened.book,
        opened.order_book,
        (),
        start_index=1,
        bars=3,
    )

    assert result.termination_reason == EconomicTerminationReason.INSOLVENCY.value
    assert result.bars_advanced == 1
    assert result.next_index == 2
    assert result.book.quantities.tolist() == [0.0]
    assert result.book.cash == pytest.approx(-500.0)
    assert result.book.portfolio_value == pytest.approx(-500.0)
    assert result.book.mark_prices.tolist() == [400.0]
    assert result.book.margin_used == 0.0
    assert result.book.maintenance_requirement == 0.0
    assert result.order_book.active_orders == ()
    assert result.interval_funding == 0.0
    assert result.interval_borrow_cost == 0.0
    assert result.funding_evidence == ()


def test_retain_debt_terminal_book_is_not_resumable_or_mutated_by_rejection() -> None:
    dataset = _market()
    executor = _executor(dataset, insolvency_valuation="retain_debt")
    opened = _open_short(executor)
    terminal = executor.execute_orders(
        opened.book,
        opened.order_book,
        (),
        start_index=1,
        bars=3,
    )
    terminal_cash = terminal.book.cash
    terminal_orders = terminal.order_book

    with pytest.raises(ValueError, match="positive starting equity"):
        executor.execute_orders(
            terminal.book,
            terminal.order_book,
            (),
            start_index=terminal.next_index,
            bars=2,
        )

    assert terminal.book.cash == terminal_cash
    assert terminal.book.portfolio_value == terminal_cash
    assert terminal.book.quantities.tolist() == [0.0]
    assert terminal.order_book is terminal_orders
    assert terminal.order_book.active_orders == ()


def test_retain_debt_stops_after_borrow_insolvency_before_funding_credit() -> None:
    dataset = _market(insolvency_price=299.0, insolvency_borrow_rate=200.0)
    executor = _executor(
        dataset,
        insolvency_valuation="retain_debt",
        borrow_rate_multiplier=1.0,
    )
    opened = _open_short(executor)

    result = executor.execute_orders(
        opened.book,
        opened.order_book,
        (),
        start_index=1,
        bars=3,
    )
    expected_borrow = 5.0 * 299.0 * 200.0 / dataset.periods_per_year

    assert result.termination_reason == EconomicTerminationReason.INSOLVENCY.value
    assert result.bars_advanced == 1
    assert result.next_index == 2
    assert result.interval_borrow_cost == pytest.approx(expected_borrow)
    expected_terminal_value = 5.0 - expected_borrow
    assert result.book.cash == pytest.approx(expected_terminal_value)
    assert result.book.portfolio_value == pytest.approx(expected_terminal_value)
    assert result.interval_funding == 0.0
    assert result.funding_evidence == ()
    assert result.book.quantities.tolist() == [0.0]
    assert result.order_book.active_orders == ()


def test_retain_debt_stops_after_processing_cash_interest_before_carry() -> None:
    dataset = _market(insolvency_price=100.0, processing_cash_rate=-6_000.0)
    executor = _executor(
        dataset,
        insolvency_valuation="retain_debt",
        borrow_rate_multiplier=1.0,
    )
    opened = _open_short(executor)

    result = executor.execute_orders(
        opened.book,
        opened.order_book,
        (),
        start_index=1,
        bars=3,
    )
    expected_interest = (
        opened.book.cash * dataset.cash_rate[2] * dataset.elapsed_year_fraction(1, 2)
    )

    assert result.termination_reason == EconomicTerminationReason.INSOLVENCY.value
    assert result.bars_advanced == 1
    assert result.next_index == 2
    assert result.interval_cash_interest == pytest.approx(expected_interest)
    assert result.interval_borrow_cost == 0.0
    assert result.interval_funding == 0.0
    expected_terminal_value = 1_000.0 + expected_interest
    assert result.book.cash == pytest.approx(expected_terminal_value)
    assert result.book.portfolio_value == pytest.approx(expected_terminal_value)
    assert result.book.quantities.tolist() == [0.0]
    assert result.order_book.active_orders == ()


def test_retain_debt_stops_after_gap_insolvency_before_borrow_or_open_mark() -> None:
    dataset = _market(gap_before_insolvency=True)
    executor = _executor(dataset, insolvency_valuation="retain_debt")
    opened = _open_short(executor)

    result = executor.execute_orders(
        opened.book,
        opened.order_book,
        (),
        start_index=1,
        bars=3,
    )
    elapsed_hours = dataset.elapsed_hours(1, 2)
    elapsed_year_fraction = dataset.elapsed_year_fraction(1, 2)
    gap_year_fraction = elapsed_year_fraction - (
        elapsed_year_fraction * dataset.bar_hours / elapsed_hours
    )
    expected_gap_interest = opened.book.cash * dataset.cash_rate[2] * gap_year_fraction

    assert result.bars_advanced == 1
    assert result.next_index == 2
    assert result.book.insolvent is True
    assert result.book.termination_reason is EconomicTerminationReason.INSOLVENCY
    assert result.book.cash == pytest.approx(
        opened.book.portfolio_value + expected_gap_interest
    )
    assert result.book.portfolio_value == pytest.approx(
        opened.book.portfolio_value + expected_gap_interest
    )
    assert result.book.mark_prices.tolist() == [100.0]
    assert result.interval_cash_interest == pytest.approx(expected_gap_interest)
    assert result.interval_borrow_cost == 0.0
    assert result.interval_funding == 0.0


def test_retain_debt_cancels_other_same_bar_orders_after_first_insolvency() -> None:
    dataset = _market()
    executor = _executor(
        dataset,
        insolvency_valuation="retain_debt",
        fee_rate=11.0,
    )
    intents = (
        _intent(executor, quantity=1.0, target_identity="first-cost-exhaustion"),
        _intent(executor, quantity=1.0, target_identity="later-same-bar-order"),
    )

    result = executor.execute_orders(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        OrderBookState.empty(),
        intents,
        start_index=0,
        bars=1,
    )

    fills = tuple(
        event
        for event in result.order_events
        if event.event_type in {"filled", "partial_fill"}
    )
    cancellations = tuple(
        event
        for event in result.order_events
        if event.event_type == "cancelled" and event.reason == "economic_termination"
    )
    assert result.fill_count == 1
    assert len(fills) == 1
    assert len(cancellations) == 1
    assert result.book.insolvent is True
    assert result.book.quantities.tolist() == [0.0]
    assert result.book.cash == pytest.approx(-100.0)
    assert result.order_book.active_orders == ()


def test_floor_zero_keeps_legacy_flattening_and_requested_bar_count() -> None:
    dataset = _market()
    executor = _executor(dataset)
    opened = _open_short(executor)

    result = executor.execute_orders(
        opened.book,
        opened.order_book,
        (),
        start_index=1,
        bars=3,
    )

    assert result.termination_reason == EconomicTerminationReason.INSOLVENCY.value
    assert result.bars_advanced == 3
    assert result.next_index == 4
    assert result.book.quantities.tolist() == [0.0]
    assert result.book.cash == 0.0
    assert result.book.portfolio_value == 0.0
