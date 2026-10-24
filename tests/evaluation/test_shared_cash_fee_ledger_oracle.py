from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.simulation import ExecutionCostConfig
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
