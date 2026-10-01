from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.position_intent import PositionIntent


@dataclass
class _AlwaysLong:
    def decide(self, observation: object) -> PositionIntent:
        del observation
        return PositionIntent.LONG


def _two_symbol_market() -> MarketDataset:
    close = np.asarray(
        [
            [100.0, 200.0],
            [100.0, 200.0],
            [110.0, 190.0],
            [120.0, 180.0],
            [140.0, 160.0],
            [140.0, 160.0],
        ],
        dtype=np.float64,
    )
    open_price = np.vstack((close[0], close[:-1]))
    n_bars, n_symbols = close.shape
    return MarketDataset(
        dataset_id="e" * 64,
        symbols=("ALPHAUSDT", "BETAUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((n_bars, n_symbols), 1_000_000.0),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        mark_price=close.copy(),
    )


def test_shared_cash_terminal_cash_and_cost_match_hand_calculation() -> None:
    result = run_shared_cash_replay(
        _two_symbol_market(),
        (_AlwaysLong(), _AlwaysLong()),
        start_index=0,
        stop_index=5,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.01,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
        ),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=0.75,
                max_abs_weight=0.5,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    # Both 25% targets buy $250 at the next open: 2.5 ALPHA and 1.25 BETA.
    # Price P&L by the final mark is +$50 and $0, respectively. Entry fees are
    # $5 total; the $550 terminal sale incurs $5.50 in fees.
    expected_terminal_cash = 1_000.0 + 50.0 - 5.0 - 5.5
    expected_total_cost = 5.0 + 5.5

    ledger = result.ledger_evidence
    assert ledger is not None
    assert len(result.decisions) == 4
    assert len(ledger.intervals) == 5
    assert ledger.intervals[0].exact_quantities_after == ("5/2", "5/4")
    assert [interval.interval_cost for interval in ledger.intervals] == pytest.approx(
        [5.0, 0.0, 0.0, 0.0, 5.5]
    )
    np.testing.assert_allclose(result.book.quantities, np.zeros(2))
    assert result.diagnostics.total_cost == pytest.approx(expected_total_cost)
    assert result.book.total_cost == pytest.approx(expected_total_cost)
    assert result.book.cash == pytest.approx(expected_terminal_cash)
    assert ledger.final_total_cost == pytest.approx(expected_total_cost)
    assert ledger.final_cash == pytest.approx(expected_terminal_cash)
