"""Terminal settlement must close residuals below the venue minimum notional.

A position can decay below the exchange minimum notional after it was opened
(adverse price move, drawdown de-leveraging). The reserved terminal interval is a
risk-owned reduce-only close; a venue does not apply the minimum-notional filter to
reduce-only orders, so the book must end flat and the independent ledger
reconciliation (terminal equity from raw fills) must pass.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import numpy as np

from tests.evaluation.trend_stop_support import synthetic_dataset
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.dev_realdata_study.runner import reconcile_ledger
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy

SYMBOLS = ("AAAUSDT", "BBBUSDT")
MINIMUM_NOTIONAL = 5.0
CAPITAL = 100.0
DROP_ROW = 8


def _dust_dataset() -> MarketDataset:
    """Flat 100.0 prices that halve at ``DROP_ROW``; lot 0.01, minimum notional 5."""

    base = synthetic_dataset(
        "4h",
        SYMBOLS,
        first_close="2021-01-01T00:00:00",
        last_close="2021-01-05T00:00:00",
    )
    shape = (base.n_bars, len(SYMBOLS))
    price = np.full(shape, 100.0)
    price[DROP_ROW:] = 50.0
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
    fields.update(
        open=price.copy(),
        high=price.copy(),
        low=price.copy(),
        close=price.copy(),
        lot_size=np.full(shape, 0.01),
        minimum_notional=np.full(shape, MINIMUM_NOTIONAL),
    )
    return MarketDataset(**fields)


def _replay(dataset: MarketDataset):
    # Two positions of 6 USDT each pass the 5 USDT entry filter, then fall to 3 USDT.
    return run_shared_portfolio_replay(
        dataset,
        BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, gross_budget=0.12),
        start_index=1,
        stop_index=dataset.n_bars - 1,
        initial_capital=CAPITAL,
        execution_cost=ExecutionCostConfig(max_participation_rate=0.5),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )


def test_terminal_settlement_closes_positions_below_minimum_notional() -> None:
    dataset = _dust_dataset()
    result = _replay(dataset)
    ledger = result.ledger_evidence
    assert ledger is not None
    opened = [
        event
        for interval in ledger.intervals.intervals[:3]
        for event in interval.order_events
        if event.filled_quantity
    ]
    assert len(opened) == len(SYMBOLS)  # entries really filled at 6 USDT each
    assert all(
        Fraction(value) == 0 for value in ledger.intervals.terminal_exact_quantities
    )


def test_ledger_reconciliation_passes_when_terminal_notional_is_below_minimum() -> None:
    dataset = _dust_dataset()
    protocol = dataclasses.replace(PROTOCOL, initial_capital=CAPITAL)
    reconciliation = reconcile_ledger(_replay(dataset), dataset, protocol)
    assert reconciliation["terminal_flat"] is True
    assert reconciliation["terminal_residual_quantities"] == {}
    assert reconciliation["passed"] is True
    assert (
        abs(float(reconciliation["equity_residual"]))
        <= protocol.reconciliation_tolerance
    )
