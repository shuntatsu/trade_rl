from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_cash_replay
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

    # $400 is bought and $400 sold; a 1% fee on $800 of raw fills is $8.
    expected_cash = initial_cash - Fraction(8)
    assert signed_cash_flow == 0
    assert filled_notional == 800
    assert fee_total == 8
    assert expected_cash == 992
    assert result.book.cash == pytest.approx(float(expected_cash))
    assert result.book.portfolio_value == pytest.approx(float(expected_cash))
    assert ledger.final_cash == pytest.approx(float(expected_cash))
    assert ledger.final_portfolio_value == pytest.approx(float(expected_cash))
    assert result.diagnostics.total_cost == pytest.approx(float(fee_total))
    assert result.book.total_cost == pytest.approx(float(fee_total))
    assert ledger.final_total_cost == pytest.approx(float(fee_total))
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert result.book.quantities == pytest.approx((0.0, 0.0))
    assert ledger.active_order_remainders == ()
    assert ledger.final_funding_pnl == 0.0
    assert ledger.final_borrow_cost == 0.0
