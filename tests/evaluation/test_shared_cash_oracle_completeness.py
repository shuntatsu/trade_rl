from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.position_intent import PositionIntent


@dataclass
class _ConstantIntent:
    intent: PositionIntent

    def decide(self, observation: object) -> PositionIntent:
        del observation
        return self.intent


def _market(*, volume: float = 1_000_000.0) -> MarketDataset:
    n_bars = 6
    close = np.tile(np.asarray((100.0, 200.0)), (n_bars, 1))
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("ALPHAUSDT", "BETAUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, 2), volume, dtype=np.float64),
        funding_rate=np.zeros((n_bars, 2), dtype=np.float64),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        mark_price=close.copy(),
    )


def _risk() -> PreTradeRisk:
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.75,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )


def _strategies() -> tuple[_ConstantIntent, _ConstantIntent]:
    return (
        _ConstantIntent(PositionIntent.LONG),
        _ConstantIntent(PositionIntent.FLAT),
    )


def test_shared_cash_funding_debit_matches_hand_calculation() -> None:
    base = _market()
    funding_rate = np.zeros((base.n_bars, base.n_symbols), dtype=np.float64)
    funding_rate[2, 0] = 0.04
    funding_due = np.zeros_like(funding_rate, dtype=np.bool_)
    funding_due[2, 0] = True
    dataset = replace(base, funding_rate=funding_rate, funding_due=funding_due)

    result = run_shared_cash_replay(
        dataset,
        _strategies(),
        start_index=0,
        stop_index=3,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=_risk(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    assert result.ledger_evidence is not None
    assert result.ledger_evidence.final_funding_pnl == pytest.approx(-10.0)
    assert result.ledger_evidence.final_cash == pytest.approx(990.0)
    assert result.book.cash == pytest.approx(990.0)
    np.testing.assert_allclose(result.book.quantities, np.zeros(2))


def test_shared_cash_partial_fill_residual_is_settled_at_terminal() -> None:
    base = _market()
    volume = base.volume.copy()
    volume[1:5, 0] = 1.0
    volume[5, 0] = 10.0
    dataset = replace(base, volume=volume)

    result = run_shared_cash_replay(
        dataset,
        _strategies(),
        start_index=0,
        stop_index=5,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            max_participation_rate=0.5,
        ),
        risk=_risk(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
        ohlc_drawdown_stress=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    partial_fills = [
        event
        for interval in ledger.intervals
        for event in interval.order_events
        if event.event_type == "partial_fill"
    ]
    assert partial_fills
    assert partial_fills[0].filled_quantity == pytest.approx(0.5)
    assert partial_fills[0].remaining_quantity == pytest.approx(2.0)
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert ledger.active_order_remainders == ()
    assert ledger.final_cash == pytest.approx(1_000.0)
    assert ledger.final_portfolio_value == pytest.approx(1_000.0)
    np.testing.assert_allclose(result.book.quantities, np.zeros(2))
    fill_event_sequences = {
        event.sequence
        for interval in ledger.intervals
        for event in interval.order_events
        if event.event_type in {"filled", "partial_fill"}
    }
    after_fill_stress_sequences = {
        transition.evidence["fill_event_sequence"]
        for interval in ledger.intervals
        for transition in interval.accounting_transitions
        if transition.transition_type == "ohlc_drawdown_stress"
        and transition.evidence["phase"] == "after_fill"
    }
    assert fill_event_sequences == after_fill_stress_sequences


def test_shared_cash_spread_and_impact_costs_match_hand_calculation() -> None:
    result = run_shared_cash_replay(
        _market(volume=10.0),
        _strategies(),
        start_index=0,
        stop_index=2,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            spread_rate=0.01,
            impact_rate=0.04,
            max_participation_rate=1.0,
        ),
        risk=_risk(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    assert result.ledger_evidence is not None
    # Each $250 fill is 25% of 10 available units. Impact is 4% * sqrt(0.25)
    # = 2%; adding 1% spread costs $7.50 on entry and $7.50 on settlement.
    assert result.ledger_evidence.final_total_cost == pytest.approx(15.0)
    assert result.ledger_evidence.final_cash == pytest.approx(985.0)
    assert result.book.cash == pytest.approx(985.0)
    np.testing.assert_allclose(result.book.quantities, np.zeros(2))
