from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    run_scripted_shared_cash_replay,
    run_shared_cash_replay,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.strategies.position_intent import PositionIntent


@dataclass(frozen=True)
class FixedIntent:
    intent: PositionIntent

    def decide(self, observation: object) -> PositionIntent:
        del observation
        return self.intent


def _constant_price_market() -> MarketDataset:
    n_bars = 6
    prices = np.full((n_bars, 1), 100.0, dtype=np.float64)
    volume = np.asarray(
        [[2.0], [2.0], [2.0], [2.0], [1_000.0], [1_000.0]],
        dtype=np.float64,
    )
    zero_rates = np.zeros((n_bars, 1), dtype=np.float64)
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("AAA",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=volume,
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("literal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=zero_rates,
        maker_fee_rate=zero_rates,
        taker_fee_rate=zero_rates,
        spread_rate=zero_rates,
        contract_multipliers=np.ones(1, dtype=np.float64),
    )


def _two_symbol_constant_price_market(
    *, reverse_symbols: bool = False
) -> MarketDataset:
    n_bars = 6
    prices = np.tile(np.asarray([[100.0, 200.0]], dtype=np.float64), (n_bars, 1))
    volume = np.tile(np.asarray([[2.0, 1.0]], dtype=np.float64), (n_bars, 1))
    volume[-2:, :] = 1_000.0
    symbol_order = (1, 0) if reverse_symbols else (0, 1)
    symbols = ("AAA", "BBB")
    prices = prices[:, symbol_order]
    volume = volume[:, symbol_order]
    zero_rates = np.zeros((n_bars, 2), dtype=np.float64)
    return MarketDataset(
        dataset_id=("e" if reverse_symbols else "d") * 64,
        symbols=tuple(symbols[index] for index in symbol_order),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=volume,
        funding_rate=zero_rates.copy(),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("literal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=zero_rates.copy(),
        maker_fee_rate=zero_rates.copy(),
        taker_fee_rate=zero_rates.copy(),
        spread_rate=zero_rates.copy(),
        contract_multipliers=np.ones(2, dtype=np.float64),
    )


def _delayed_two_symbol_split_market() -> MarketDataset:
    n_bars = 7
    close = np.tile(np.asarray([[100.0, 200.0]], dtype=np.float64), (n_bars, 1))
    close[4:, 0] = 50.0
    open_prices = np.vstack((close[0], close[:-1]))
    open_prices[4:, 0] = 50.0
    volume = np.full((n_bars, 2), 1_000_000.0, dtype=np.float64)
    volume[1:3] = ((2.0, 1.0), (2.0, 1.0))
    funding_rate = np.zeros((n_bars, 2), dtype=np.float64)
    funding_rate[3] = (0.02, 0.005)
    funding_due = np.zeros((n_bars, 2), dtype=np.bool_)
    funding_due[3] = True
    borrow_rate = np.zeros((n_bars, 2), dtype=np.float64)
    borrow_rate[:, 1] = 0.365
    split_factor = np.ones((n_bars, 2), dtype=np.float64)
    split_factor[4, 0] = 2.0
    zero_rates = np.zeros((n_bars, 2), dtype=np.float64)
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=("AAA", "BBB"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_prices,
        high=np.maximum(open_prices, close),
        low=np.minimum(open_prices, close),
        close=close,
        volume=volume,
        funding_rate=funding_rate,
        funding_due=funding_due,
        borrow_rate=borrow_rate,
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("literal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=zero_rates.copy(),
        maker_fee_rate=zero_rates.copy(),
        taker_fee_rate=zero_rates.copy(),
        spread_rate=zero_rates.copy(),
        contract_multipliers=np.ones(2, dtype=np.float64),
        split_factor=split_factor,
    )


def test_shared_cash_terminal_cash_matches_independent_raw_fill_fee_oracle() -> None:
    initial_cash = Fraction(1_000)
    fee_rate = Fraction(1, 100)
    dataset = _constant_price_market()
    execution_cost = ExecutionCostConfig(
        fee_rate=float(fee_rate),
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=0.5,
        slippage_std=0.0,
        max_leverage=1.0,
        maintenance_margin_rate=0.0,
        order_latency_bars=0,
    )

    result = run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG),),
        start_index=0,
        stop_index=5,
        gross_budget=0.2,
        initial_capital=float(initial_cash),
        execution_cost=execution_cost,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    fill_events = tuple(
        event
        for interval in ledger.intervals
        for event in interval.order_events
        if event.filled_quantity != 0.0
    )
    assert tuple(event.event_type for event in fill_events) == (
        "partial_fill",
        "filled",
        "filled",
    )

    multiplier = Fraction(1)
    signed_cash_flow = sum(
        (
            -Fraction(str(event.filled_quantity))
            * Fraction(str(event.execution_price))
            * multiplier
            for event in fill_events
        ),
        start=Fraction(0),
    )
    fee_total = sum(
        (
            abs(Fraction(str(event.filled_quantity)))
            * Fraction(str(event.execution_price))
            * multiplier
            * fee_rate
            for event in fill_events
        ),
        start=Fraction(0),
    )
    independently_reconciled_cash = initial_cash + signed_cash_flow - fee_total

    assert tuple(event.filled_quantity for event in fill_events) == (1.0, 1.0, -2.0)
    assert signed_cash_flow == 0
    assert fee_total == 4
    assert independently_reconciled_cash == 996
    assert result.book.cash == pytest.approx(float(independently_reconciled_cash))
    assert ledger.final_cash == pytest.approx(float(independently_reconciled_cash))
    assert result.diagnostics.total_cost == pytest.approx(float(fee_total))
    assert ledger.terminal_exact_quantities == ("0",)
    assert result.book.quantities == pytest.approx((0.0,))
    assert ledger.active_order_remainders == ()


@pytest.mark.parametrize("reverse_symbols", (False, True))
def test_two_symbol_partial_fill_fees_reconcile_under_symbol_axis_reversal(
    reverse_symbols: bool,
) -> None:
    # At constant marks, 20% per symbol means $200 each: 2 AAA and 1 BBB.
    # Participation-limited volumes fill 1 AAA and 0.5 BBB on each entry bar.
    initial_cash = Fraction(1_000)
    fee_rate = Fraction(1, 100)
    expected_fills = {
        "AAA": (Fraction(1), Fraction(1), Fraction(-2)),
        "BBB": (Fraction(1, 2), Fraction(1, 2), Fraction(-1)),
    }
    expected_event_types = ("partial_fill", "filled", "filled")
    execution_cost = ExecutionCostConfig(
        fee_rate=float(fee_rate),
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=0.5,
        slippage_std=0.0,
        max_leverage=1.0,
        maintenance_margin_rate=0.0,
        order_latency_bars=0,
    )
    dataset = _two_symbol_constant_price_market(reverse_symbols=reverse_symbols)
    result = run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.LONG)),
        start_index=0,
        stop_index=5,
        gross_budget=0.2,
        initial_capital=float(initial_cash),
        execution_cost=execution_cost,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    fills_by_symbol: dict[str, list[OrderEvent]] = {
        symbol: [] for symbol in dataset.symbols
    }
    signed_cash_flow = Fraction(0)
    filled_notional = Fraction(0)
    fee_total = Fraction(0)
    for interval in ledger.intervals:
        for event in interval.order_events:
            if event.filled_quantity == 0.0:
                continue
            assert event.execution_price is not None
            quantity = Fraction(str(event.filled_quantity))
            price = Fraction(str(event.execution_price))
            notional = abs(quantity) * price
            fills_by_symbol[dataset.symbols[event.symbol_index]].append(event)
            signed_cash_flow -= quantity * price
            filled_notional += notional
            fee_total += notional * fee_rate
            assert Fraction(str(event.filled_notional)) == notional

    assert {
        symbol: tuple(Fraction(str(event.filled_quantity)) for event in events)
        for symbol, events in fills_by_symbol.items()
    } == expected_fills
    assert all(
        tuple(event.event_type for event in events) == expected_event_types
        and events[0].processing_index
        < events[1].processing_index
        < events[2].processing_index
        for events in fills_by_symbol.values()
    )
    assert tuple(event.processing_index for event in fills_by_symbol["AAA"]) == tuple(
        event.processing_index for event in fills_by_symbol["BBB"]
    )

    # Signed TRADE-notional cash flow is zero before fees; fees move cash by -$8.
    # The fills buy and sell $400 each, so the 1% fee on $800 is $8.
    expected_cash = initial_cash - Fraction(8)
    assert signed_cash_flow == 0
    assert filled_notional == 800
    assert fee_total == 8
    assert expected_cash == 992
    assert result.book.cash == pytest.approx(float(expected_cash), abs=1e-10, rel=0.0)
    assert result.book.portfolio_value == pytest.approx(
        float(expected_cash), abs=1e-10, rel=0.0
    )
    assert ledger.final_cash == pytest.approx(float(expected_cash), abs=1e-10, rel=0.0)
    assert ledger.final_portfolio_value == pytest.approx(
        float(expected_cash), abs=1e-10, rel=0.0
    )
    assert result.diagnostics.total_cost == pytest.approx(
        float(fee_total), abs=1e-10, rel=0.0
    )
    assert result.book.total_cost == pytest.approx(float(fee_total), abs=1e-10, rel=0.0)
    assert ledger.final_total_cost == pytest.approx(
        float(fee_total), abs=1e-10, rel=0.0
    )
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert result.book.quantities == pytest.approx((0.0, 0.0))
    assert ledger.active_order_remainders == ()
    assert ledger.final_funding_pnl == 0.0
    assert ledger.final_borrow_cost == 0.0


def test_shared_cash_partial_fills_reconcile_fees_funding_and_short_borrow() -> None:
    initial_cash = Fraction(1_000)
    fee_rate = Fraction(1, 100)
    annual_borrow_rate = Fraction(365, 1_000)
    dataset = _two_symbol_constant_price_market()

    funding_rate = np.zeros((dataset.n_bars, dataset.n_symbols), dtype=np.float64)
    funding_rate[3] = (float(Fraction(1, 100)), float(Fraction(1, 200)))
    funding_due = np.zeros((dataset.n_bars, dataset.n_symbols), dtype=np.bool_)
    funding_due[3] = True
    borrow_rate = np.zeros((dataset.n_bars, dataset.n_symbols), dtype=np.float64)
    borrow_rate[:, 1] = float(annual_borrow_rate)
    dataset = replace(
        dataset,
        funding_rate=funding_rate,
        funding_due=funding_due,
        borrow_rate=borrow_rate,
    )

    execution_cost = ExecutionCostConfig(
        fee_rate=float(fee_rate),
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=0.5,
        slippage_std=0.0,
        max_leverage=1.0,
        maintenance_margin_rate=0.0,
        borrow_rate_multiplier=1.0,
        order_latency_bars=0,
        allow_short=True,
    )
    result = run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.SHORT)),
        start_index=0,
        stop_index=5,
        gross_budget=0.2,
        initial_capital=float(initial_cash),
        execution_cost=execution_cost,
        minimum_hold_bars=0,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    expected_fills = {
        "AAA": (Fraction(1), Fraction(1), Fraction(-2)),
        "BBB": (Fraction(-1, 2), Fraction(-1, 2), Fraction(1)),
    }
    fills_by_symbol: dict[str, list[OrderEvent]] = {
        symbol: [] for symbol in dataset.symbols
    }
    signed_trade_cash_flow = Fraction(0)
    filled_notional = Fraction(0)
    fee_by_symbol = {symbol: Fraction(0) for symbol in dataset.symbols}
    for interval in ledger.intervals:
        for event in interval.order_events:
            if event.filled_quantity == 0.0:
                continue
            assert event.execution_price is not None
            quantity = Fraction(str(event.filled_quantity))
            price = Fraction(str(event.execution_price))
            notional = abs(quantity) * price
            symbol = dataset.symbols[event.symbol_index]
            fills_by_symbol[symbol].append(event)
            signed_trade_cash_flow -= quantity * price
            filled_notional += notional
            fee_by_symbol[symbol] += notional * fee_rate
            assert Fraction(str(event.filled_notional)) == notional

    assert {
        symbol: tuple(Fraction(str(event.filled_quantity)) for event in events)
        for symbol, events in fills_by_symbol.items()
    } == expected_fills
    assert all(
        tuple(event.event_type for event in events)
        == ("partial_fill", "filled", "filled")
        for events in fills_by_symbol.values()
    )
    assert all(
        tuple(event.processing_index for event in events) == (1, 2, 5)
        for events in fills_by_symbol.values()
    )
    assert filled_notional == 800
    assert signed_trade_cash_flow == 0
    assert fee_by_symbol == {"AAA": Fraction(4), "BBB": Fraction(4)}
    expected_fee_total = Fraction(8)

    # Row 3 is the only funding boundary: +$200 long pays $2 and -$200 short
    # receives $1. Borrow is charged on the processing bars after fills at rows
    # 1-4, then the terminal close leaves no short position to charge at row 5.
    expected_funding = -Fraction(200) * Fraction(1, 100) + Fraction(200) * Fraction(
        1, 200
    )
    expected_short_notionals = (
        Fraction(100),
        Fraction(200),
        Fraction(200),
        Fraction(200),
    )
    assert sum(expected_short_notionals, start=Fraction(0)) == 700
    expected_borrow = Fraction(700) * annual_borrow_rate / 8_760
    assert expected_funding == -1
    assert expected_borrow == Fraction(7, 240)
    expected_cash = (
        initial_cash
        + signed_trade_cash_flow
        - expected_fee_total
        + expected_funding
        - expected_borrow
    )
    assert expected_cash == Fraction(237_833, 240)

    funding_intervals = [
        interval for interval in ledger.intervals if interval.funding_events
    ]
    assert len(funding_intervals) == 1
    funding_interval = funding_intervals[0]
    boundary = funding_interval.funding_events[0]
    assert funding_interval.next_index == 3
    assert boundary.processing_index == 3
    assert boundary.timestamp_ns == int(
        np.datetime64("2026-01-01T03:00:00", "ns").astype(np.int64)
    )
    assert boundary.funding_due == (True, True)
    assert tuple(Fraction(str(value)) for value in boundary.signed_quantities) == (
        Fraction(2),
        Fraction(-1),
    )
    assert tuple(Fraction(str(value)) for value in boundary.mark_prices) == (
        Fraction(100),
        Fraction(200),
    )
    assert tuple(Fraction(str(value)) for value in boundary.funding_rates) == (
        Fraction(1, 100),
        Fraction(1, 200),
    )
    assert boundary.funding_amount == pytest.approx(
        float(expected_funding), abs=1e-10, rel=0.0
    )
    assert funding_interval.interval_funding == pytest.approx(
        float(expected_funding), abs=1e-10, rel=0.0
    )
    assert all(
        interval.interval_funding == pytest.approx(0.0, abs=1e-10, rel=0.0)
        for interval in ledger.intervals
        if interval.next_index != 3
    )

    expected_borrow_by_index = {
        1: Fraction(100) * annual_borrow_rate / 8_760,
        2: Fraction(200) * annual_borrow_rate / 8_760,
        3: Fraction(200) * annual_borrow_rate / 8_760,
        4: Fraction(200) * annual_borrow_rate / 8_760,
        5: Fraction(0),
    }
    assert {interval.next_index for interval in ledger.intervals} == set(
        expected_borrow_by_index
    )
    for interval in ledger.intervals:
        assert interval.interval_borrow_cost == pytest.approx(
            float(expected_borrow_by_index[interval.next_index]),
            abs=1e-10,
            rel=0.0,
        )

    assert result.book.cash == pytest.approx(float(expected_cash), abs=1e-10, rel=0.0)
    assert result.book.portfolio_value == pytest.approx(
        float(expected_cash), abs=1e-10, rel=0.0
    )
    assert ledger.final_cash == pytest.approx(float(expected_cash), abs=1e-10, rel=0.0)
    assert ledger.final_portfolio_value == pytest.approx(
        float(expected_cash), abs=1e-10, rel=0.0
    )
    assert result.diagnostics.total_cost == pytest.approx(
        float(expected_fee_total), abs=1e-10, rel=0.0
    )
    assert result.book.total_cost == pytest.approx(
        float(expected_fee_total), abs=1e-10, rel=0.0
    )
    assert ledger.final_total_cost == pytest.approx(
        float(expected_fee_total), abs=1e-10, rel=0.0
    )
    assert result.diagnostics.funding_pnl == pytest.approx(
        float(expected_funding), abs=1e-10, rel=0.0
    )
    assert result.book.funding_pnl == pytest.approx(
        float(expected_funding), abs=1e-10, rel=0.0
    )
    assert ledger.final_funding_pnl == pytest.approx(
        float(expected_funding), abs=1e-10, rel=0.0
    )
    assert result.diagnostics.borrow_cost == pytest.approx(
        float(expected_borrow), abs=1e-10, rel=0.0
    )
    assert result.book.borrow_cost == pytest.approx(
        float(expected_borrow), abs=1e-10, rel=0.0
    )
    assert ledger.final_borrow_cost == pytest.approx(
        float(expected_borrow), abs=1e-10, rel=0.0
    )
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert result.book.quantities == pytest.approx((0.0, 0.0), abs=1e-10, rel=0.0)
    assert ledger.active_order_remainders == ()


def test_delayed_two_symbol_raw_events_reconcile_accounting_across_split() -> None:
    initial_cash = Fraction(1_000)
    fee_rate = Fraction(1, 100)
    annual_borrow_rate = Fraction(365, 1_000)
    hourly_year_fraction = Fraction(1, 8_760)
    result = run_scripted_shared_cash_replay(
        _delayed_two_symbol_split_market(),
        actions_by_symbol={
            "AAA": (PositionIntent.LONG, None, None, None),
            "BBB": (PositionIntent.SHORT, None, None, None),
        },
        start_index=0,
        stop_index=6,
        gross_budget=0.2,
        initial_capital=float(initial_cash),
        execution_cost=ExecutionCostConfig(
            fee_rate=float(fee_rate),
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=0.5,
            slippage_std=0.0,
            max_leverage=1.0,
            maintenance_margin_rate=0.0,
            borrow_rate_multiplier=1.0,
            order_latency_bars=1,
            allow_short=True,
        ),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    assert tuple(interval.next_index for interval in ledger.intervals) == tuple(
        range(1, 7)
    )

    fills_by_symbol: dict[str, list[OrderEvent]] = {"AAA": [], "BBB": []}
    for interval in ledger.intervals:
        for event in interval.order_events:
            if event.filled_quantity != 0.0:
                fills_by_symbol[("AAA", "BBB")[event.symbol_index]].append(event)

    expected_fills = {
        "AAA": (Fraction(1), Fraction(1), Fraction(-4)),
        "BBB": (Fraction(-1, 2), Fraction(-1, 2), Fraction(1)),
    }
    entry_order_ids: set[str] = set()
    for symbol, events in fills_by_symbol.items():
        assert (
            tuple(Fraction(str(event.filled_quantity)) for event in events)
            == expected_fills[symbol]
        )
        assert tuple(event.processing_index for event in events) == (1, 2, 5)
        assert tuple(event.event_type for event in events) == (
            "partial_fill",
            "filled",
            "filled",
        )
        assert events[0].order_id == events[1].order_id
        assert abs(Fraction(str(events[0].remaining_quantity))) > 0
        assert Fraction(str(events[1].remaining_quantity)) == 0
        entry_order_ids.add(events[0].order_id)
    assert len(entry_order_ids) == 2
    assert not any(
        event.event_type == "cancelled" and event.order_id in entry_order_ids
        for interval in ledger.intervals
        for event in interval.order_events
    )

    # Reconstruct each row from only raw fill events and literal fixture facts.
    # The split changes AAA's units at row 4; it does not change cash.
    quantities = [Fraction(0), Fraction(0)]
    expected_cash = initial_cash
    expected_fee_total = Fraction(0)
    expected_funding_total = Fraction(0)
    expected_borrow_total = Fraction(0)
    filled_notional_total = Fraction(0)
    signed_trade_cash_total = Fraction(0)
    expected_borrow_by_index: dict[int, Fraction] = {}
    for interval in ledger.intervals:
        processing_index = interval.next_index
        if processing_index == 4:
            quantities[0] *= 2

        interval_cash_flow = Fraction(0)
        interval_fee = Fraction(0)
        for event in interval.order_events:
            if event.filled_quantity == 0.0:
                continue
            quantity = Fraction(str(event.filled_quantity))
            assert event.execution_price is not None
            price = Fraction(str(event.execution_price))
            notional = abs(quantity) * price
            assert Fraction(str(event.filled_notional)) == notional
            interval_cash_flow -= quantity * price
            interval_fee += notional * fee_rate
            filled_notional_total += notional
            quantities[event.symbol_index] += quantity

        funding = Fraction(0)
        if processing_index == 3:
            # Literal row-3 financing: +2 AAA at $100 pays 2%; -1 BBB at
            # $200 receives 0.5%.
            funding = -(
                quantities[0] * 100 * Fraction(1, 50)
                + quantities[1] * 200 * Fraction(1, 200)
            )
        short_notional = max(-quantities[1], Fraction(0)) * 200
        borrow = short_notional * annual_borrow_rate * hourly_year_fraction
        expected_borrow_by_index[processing_index] = borrow

        signed_trade_cash_total += interval_cash_flow
        expected_cash += interval_cash_flow - interval_fee + funding - borrow
        expected_fee_total += interval_fee
        expected_funding_total += funding
        expected_borrow_total += borrow

        marks = (50 if processing_index >= 4 else 100, 200)
        expected_equity = expected_cash + sum(
            quantity * mark for quantity, mark in zip(quantities, marks, strict=True)
        )
        assert tuple(
            Fraction(value) for value in interval.exact_quantities_after
        ) == tuple(quantities)
        assert interval.cash_after == pytest.approx(
            float(expected_cash), abs=1e-10, rel=0.0
        )
        assert interval.portfolio_value_after == pytest.approx(
            float(expected_equity), abs=1e-10, rel=0.0
        )
        assert interval.interval_cost == pytest.approx(
            float(interval_fee), abs=1e-10, rel=0.0
        )
        assert interval.interval_funding == pytest.approx(
            float(funding), abs=1e-10, rel=0.0
        )
        assert interval.interval_borrow_cost == pytest.approx(
            float(borrow), abs=1e-10, rel=0.0
        )

    assert signed_trade_cash_total == 0
    assert filled_notional_total == 800
    assert expected_fee_total == 8
    assert expected_funding_total == -3
    assert expected_borrow_total == Fraction(7, 240)
    assert expected_borrow_by_index == {
        1: Fraction(73, 17_520),
        2: Fraction(73, 8_760),
        3: Fraction(73, 8_760),
        4: Fraction(73, 8_760),
        5: Fraction(0),
        6: Fraction(0),
    }
    assert expected_cash == Fraction(237_353, 240)
    assert float(expected_cash) == pytest.approx(ledger.final_cash, abs=1e-10, rel=0.0)
    assert expected_cash == pytest.approx(result.book.cash, abs=1e-10, rel=0.0)
    assert expected_cash == pytest.approx(
        ledger.final_portfolio_value, abs=1e-10, rel=0.0
    )
    assert expected_fee_total == pytest.approx(ledger.final_total_cost, abs=1e-10)
    assert expected_funding_total == pytest.approx(ledger.final_funding_pnl, abs=1e-10)
    assert expected_borrow_total == pytest.approx(ledger.final_borrow_cost, abs=1e-10)
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert ledger.active_order_remainders == ()
