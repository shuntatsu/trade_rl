from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    run_scripted_shared_cash_replay,
    run_shared_cash_replay,
    run_shared_portfolio_replay,
)
from trade_rl.simulation import (
    BookState,
    ExecutionCostConfig,
    ExecutionResult,
    MarketExecutor,
)
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.simulation.stateful.execution import StatefulExecutionObservation
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.position_intent import PositionIntent


@dataclass(frozen=True)
class FixedIntent:
    intent: PositionIntent

    def decide(self, observation: object) -> PositionIntent:
        del observation
        return self.intent


@dataclass(frozen=True)
class FixedPortfolioTarget:
    target_weights: tuple[float, ...]

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        del snapshot
        return PortfolioTargetProposal(self.target_weights)


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
    n_bars = 8
    close = np.tile(np.asarray([[100.0, 200.0]], dtype=np.float64), (n_bars, 1))
    close[4:, 0] = 50.0
    open_prices = np.vstack((close[0], close[:-1]))
    open_prices[4:, 0] = 50.0
    volume = np.full((n_bars, 2), 1_000_000.0, dtype=np.float64)
    volume[1:6] = (
        (2.0, 1.0),
        (1.0, 0.25),
        (0.5, 0.25),
        (1_000_000.0, 0.25),
        (1_000_000.0, 0.25),
    )
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

    # The incremental API must satisfy this independent raw-fill money oracle,
    # not only agree with the batch path that now delegates to the same session.
    from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession

    session = SharedPortfolioSession(
        dataset,
        start_index=0,
        stop_index=5,
        initial_capital=float(initial_cash),
        execution_cost=execution_cost,
        settle_terminal_position=True,
    )
    fixed_quantities = (2.0, -1.0)
    while session.can_decide:
        book = session.book
        prices = dataset.resolved_array("mark_price")[session.current_index]
        target_weights = tuple(
            quantity * float(price) / book.portfolio_value
            for quantity, price in zip(fixed_quantities, prices, strict=True)
        )
        session.advance(target_weights)
    while session.can_settle_terminal:
        session.settle_next_interval()

    session_book = session.book
    assert session.finished
    assert session_book.cash == pytest.approx(float(expected_cash), abs=1e-10, rel=0.0)
    assert session_book.total_cost == pytest.approx(
        float(expected_fee_total), abs=1e-10, rel=0.0
    )
    assert session_book.funding_pnl == pytest.approx(
        float(expected_funding), abs=1e-10, rel=0.0
    )
    assert session_book.borrow_cost == pytest.approx(
        float(expected_borrow), abs=1e-10, rel=0.0
    )
    assert session_book.quantities == pytest.approx((0.0, 0.0), abs=0.0)

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
            "AAA": (PositionIntent.LONG, None, None, None, None),
            "BBB": (PositionIntent.SHORT, None, None, None, None),
        },
        start_index=0,
        stop_index=7,
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
        range(1, 8)
    )

    fills_by_symbol: dict[str, list[OrderEvent]] = {"AAA": [], "BBB": []}
    for interval in ledger.intervals:
        for event in interval.order_events:
            if event.filled_quantity != 0.0:
                fills_by_symbol[("AAA", "BBB")[event.symbol_index]].append(event)

    expected_fills = {
        "AAA": (
            Fraction(1),
            Fraction(1, 2),
            Fraction(1, 4),
            Fraction(1, 2),
            Fraction(-4),
        ),
        "BBB": (
            Fraction(-1, 2),
            Fraction(-1, 8),
            Fraction(-1, 8),
            Fraction(-1, 8),
            Fraction(-1, 8),
            Fraction(1),
        ),
    }
    initial_order_ids: dict[str, str] = {}
    for symbol, events in fills_by_symbol.items():
        assert (
            tuple(Fraction(str(event.filled_quantity)) for event in events)
            == expected_fills[symbol]
        )
        initial_order_ids[symbol] = events[0].order_id

    aaa_fills = fills_by_symbol["AAA"]
    assert tuple(event.processing_index for event in aaa_fills) == (1, 2, 3, 5, 6)
    assert tuple(event.event_type for event in aaa_fills) == (
        "partial_fill",
        "partial_fill",
        "partial_fill",
        "filled",
        "filled",
    )
    assert (
        tuple(event.order_id for event in aaa_fills[:3])
        == (initial_order_ids["AAA"],) * 3
    )
    assert Fraction(str(aaa_fills[2].remaining_quantity)) == Fraction(1, 4)
    assert aaa_fills[3].order_id != initial_order_ids["AAA"]
    assert aaa_fills[4].order_id != aaa_fills[3].order_id
    assert tuple(event.processing_index for event in fills_by_symbol["BBB"]) == (
        1,
        2,
        3,
        4,
        5,
        6,
    )
    assert tuple(event.event_type for event in fills_by_symbol["BBB"]) == (
        "partial_fill",
        "partial_fill",
        "partial_fill",
        "partial_fill",
        "filled",
        "filled",
    )
    assert (
        tuple(event.order_id for event in fills_by_symbol["BBB"][:5])
        == (initial_order_ids["BBB"],) * 5
    )
    assert fills_by_symbol["BBB"][5].order_id != initial_order_ids["BBB"]
    assert Fraction(str(fills_by_symbol["BBB"][2].remaining_quantity)) == Fraction(
        -1, 4
    )

    split_cancellations = tuple(
        event
        for interval in ledger.intervals
        for event in interval.order_events
        if event.event_type == "cancelled"
        and event.reason == "split_adjustment_required"
    )
    assert len(split_cancellations) == 1
    assert split_cancellations[0].processing_index == 4
    assert split_cancellations[0].symbol_index == 0
    assert split_cancellations[0].order_id == initial_order_ids["AAA"]
    assert Fraction(str(split_cancellations[0].remaining_quantity)) == Fraction(1, 4)
    assert not any(
        event.event_type == "cancelled" and event.order_id == initial_order_ids["BBB"]
        for interval in ledger.intervals
        for event in interval.order_events
    )

    split_interval = ledger.intervals[3]
    assert split_interval.exact_quantities_before == ("7/4", "-3/4")
    assert split_interval.exact_quantities_after == ("7/2", "-7/8")
    surviving_bbb_fill = next(
        event
        for event in split_interval.order_events
        if event.symbol_index == 1 and event.filled_quantity != 0.0
    )
    assert surviving_bbb_fill.order_id == initial_order_ids["BBB"]
    assert Fraction(str(surviving_bbb_fill.filled_quantity)) == Fraction(-1, 8)
    assert Fraction(str(surviving_bbb_fill.remaining_quantity)) == Fraction(-1, 8)
    post_split_interval = ledger.intervals[4]
    assert post_split_interval.exact_quantities_before == ("7/2", "-7/8")
    assert post_split_interval.exact_quantities_after == ("4", "-1")
    rebound_submissions = tuple(
        event
        for event in post_split_interval.order_events
        if event.symbol_index == 0
        and event.event_type == "submitted"
        and event.processing_index == 4
    )
    assert len(rebound_submissions) == 1
    assert rebound_submissions[0].order_id == aaa_fills[3].order_id
    assert Fraction(str(rebound_submissions[0].remaining_quantity)) == Fraction(1, 2)
    assert fills_by_symbol["BBB"][4].order_id == initial_order_ids["BBB"]
    assert Fraction(str(fills_by_symbol["BBB"][4].remaining_quantity)) == 0

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
            # Literal row-3 financing: +7/4 AAA at $100 pays 2%; -3/4 BBB
            # at $200 receives 0.5%.
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
    assert expected_funding_total == Fraction(-11, 4)
    assert expected_borrow_total == Fraction(1, 32)
    assert expected_borrow_by_index == {
        1: Fraction(73, 17_520),
        2: Fraction(73, 14_016),
        3: Fraction(1, 160),
        4: Fraction(511, 70_080),
        5: Fraction(73, 8_760),
        6: Fraction(0),
        7: Fraction(0),
    }
    assert expected_cash == Fraction(31_655, 32)
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


@pytest.mark.parametrize("initial_capital", (True, "1000", None))
def test_portfolio_session_rejects_non_numeric_initial_capital(
    initial_capital: object,
) -> None:
    from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession

    with pytest.raises(TypeError, match="initial_capital must be a real numeric value"):
        SharedPortfolioSession(
            _constant_price_market(),
            start_index=0,
            stop_index=5,
            initial_capital=initial_capital,  # type: ignore[arg-type]
        )


def test_portfolio_session_normalizes_integer_subclass_indices() -> None:
    from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession

    class RowIndex(IntEnum):
        FIRST = 0
        STOP = 5

    session = SharedPortfolioSession(
        _constant_price_market(),
        start_index=RowIndex.FIRST,
        stop_index=RowIndex.STOP,
        initial_capital=1_000.0,
    )
    assert type(session.current_index) is int
    session.advance((0.0,))
    assert session.current_index == 1
    assert type(session.current_index) is int


def test_incremental_portfolio_session_matches_batch_across_split_and_terminal() -> (
    None
):
    from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession

    dataset = _delayed_two_symbol_split_market()
    execution_cost = ExecutionCostConfig(
        fee_rate=0.01,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=0.5,
        slippage_std=0.0001,
        random_seed=17,
        max_leverage=1.0,
        maintenance_margin_rate=0.0,
        borrow_rate_multiplier=1.0,
        order_latency_bars=1,
        allow_short=True,
    )
    direct_observations: list[StatefulExecutionObservation] = []
    direct_steps: list[ExecutionResult] = []
    execution_calls: list[tuple[MarketExecutor, BookState, BookState, int]] = []
    original_execute_interval = MarketExecutor.execute_interval

    def record_interval(
        executor: MarketExecutor,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
        hold_current_quantity_symbols: tuple[int, ...] = (),
    ) -> ExecutionResult:
        result = original_execute_interval(
            executor,
            book,
            target,
            start_index=start_index,
            bars=bars,
            hold_current_quantity_symbols=hold_current_quantity_symbols,
        )
        execution_calls.append((executor, book, result.book, start_index))
        return result

    direct_session = SharedPortfolioSession(
        dataset,
        start_index=0,
        stop_index=7,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        settle_terminal_position=True,
        execution_observer=direct_observations.append,
    )
    detached_book = direct_session.book
    detached_book.quantities[0] = 99.0
    assert direct_session.book.quantities == pytest.approx((0.0, 0.0), abs=0.0)
    with pytest.raises(ValueError, match="target weights must be finite"):
        direct_session.advance((float("nan"), -0.2))
    assert direct_session.current_index == 0
    assert not direct_session.poisoned
    assert direct_observations == []

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(MarketExecutor, "execute_interval", record_interval)
        while direct_session.can_decide:
            _, execution, _ = direct_session.advance((0.2, -0.2))
            direct_steps.append(execution)
        assert direct_session.current_index == direct_session.agent_stop_index
        book_at_actor_stop = direct_session.book
        with pytest.raises(RuntimeError, match="not accepting actor decisions"):
            direct_session.advance((0.0, 0.0))
        assert direct_session.current_index == direct_session.agent_stop_index
        assert direct_session.book.cash == book_at_actor_stop.cash
        while direct_session.can_settle_terminal:
            _, execution, _ = direct_session.settle_next_interval()
            direct_steps.append(execution)

    assert direct_session.finished
    assert len(direct_steps) == 7
    assert len(direct_observations) == 7
    assert len(execution_calls) == 7
    assert all(call[0] is execution_calls[0][0] for call in execution_calls)
    assert all(
        execution_calls[index + 1][1] is execution_calls[index][2]
        for index in range(len(execution_calls) - 1)
    )

    batch = run_shared_portfolio_replay(
        dataset,
        FixedPortfolioTarget((0.2, -0.2)),
        start_index=0,
        stop_index=7,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert batch.ledger_evidence is not None
    intervals = batch.ledger_evidence.intervals.intervals
    assert len(intervals) == len(direct_steps)
    assert tuple(step.next_index for step in direct_steps) == tuple(
        interval.next_index for interval in intervals
    )
    assert tuple(step.book.cash for step in direct_steps) == pytest.approx(
        tuple(interval.cash_after for interval in intervals), abs=0.0
    )
    assert tuple(step.book.portfolio_value for step in direct_steps) == pytest.approx(
        tuple(interval.portfolio_value_after for interval in intervals), abs=0.0
    )
    assert tuple(step.book.total_cost for step in direct_steps) == pytest.approx(
        tuple(interval.total_cost_after for interval in intervals), abs=0.0
    )
    assert tuple(step.book.funding_pnl for step in direct_steps) == pytest.approx(
        tuple(interval.funding_pnl_after for interval in intervals), abs=0.0
    )
    assert tuple(step.book.borrow_cost for step in direct_steps) == pytest.approx(
        tuple(interval.borrow_cost_after for interval in intervals), abs=0.0
    )
    assert tuple(step.book.turnover_total for step in direct_steps) == pytest.approx(
        tuple(interval.turnover_total_after for interval in intervals), abs=0.0
    )
    assert tuple(step.book.max_drawdown for step in direct_steps) == pytest.approx(
        tuple(interval.max_drawdown_after for interval in intervals), abs=0.0
    )
    assert tuple(
        tuple(str(value) for value in step.book.exact_quantities)
        for step in direct_steps
    ) == tuple(interval.exact_quantities_after for interval in intervals)
    assert tuple(step.interval_net_return for step in direct_steps) == pytest.approx(
        tuple(interval.interval_net_return for interval in intervals), abs=0.0
    )
    assert tuple(
        tuple(event.canonical_payload() for event in observation.order_events)
        for observation in direct_observations
    ) == tuple(
        tuple(event.canonical_payload() for event in interval.order_events)
        for interval in intervals
    )
    assert any(
        event.event_type == "cancelled"
        and event.reason == "split_adjustment_required"
        and event.symbol_index == 0
        for interval in intervals
        for event in interval.order_events
    )
    assert batch.book.cash == pytest.approx(direct_session.book.cash, abs=0.0)
    assert batch.book.max_drawdown == pytest.approx(
        direct_session.book.max_drawdown, abs=0.0
    )
    assert batch.book.quantities == pytest.approx((0.0, 0.0), abs=0.0)


def test_portfolio_session_is_poisoned_after_observer_failure() -> None:
    from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession

    calls = 0

    def fail_after_execution(observation: StatefulExecutionObservation) -> None:
        nonlocal calls
        del observation
        calls += 1
        raise RuntimeError("observer failed after native execution")

    session = SharedPortfolioSession(
        _constant_price_market(),
        start_index=0,
        stop_index=5,
        initial_capital=1_000.0,
        execution_observer=fail_after_execution,
    )
    with pytest.raises(RuntimeError, match="observer failed"):
        session.advance((0.2,))
    assert calls == 1
    assert session.current_index == 0
    assert session.poisoned
    assert session.finished
    with pytest.raises(RuntimeError, match="not accepting actor decisions"):
        session.advance((0.0,))
    assert calls == 1
