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


def test_shared_cash_split_preserves_hand_calculated_inventory_value() -> None:
    base = _market()
    price_fields = {
        field: getattr(base, field).copy()
        for field in ("open", "high", "low", "close", "mark_price")
    }
    for prices in price_fields.values():
        prices[2:, 0] /= 2.0
    split_factor = base.resolved_array("split_factor").copy()
    split_factor[2, 0] = 2.0
    dataset = replace(base, **price_fields, split_factor=split_factor)

    result = run_shared_cash_replay(
        dataset,
        _strategies(),
        start_index=0,
        stop_index=3,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=_risk(),
        settle_terminal_position=False,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    transition = next(
        transition
        for interval in ledger.intervals
        for transition in interval.accounting_transitions
        if transition.transition_type == "split"
    )
    assert transition.state_before.exact_quantities == ("5/2", "0")
    assert transition.state_after.exact_quantities == ("5", "0")
    assert transition.state_before.mark_prices == pytest.approx((100.0, 200.0))
    assert transition.state_after.mark_prices == pytest.approx((50.0, 200.0))
    assert transition.state_before.cash == pytest.approx(750.0)
    assert transition.state_after.cash == pytest.approx(750.0)
    assert 2.5 * 100.0 == pytest.approx(5.0 * 50.0)
    assert ledger.final_portfolio_value == pytest.approx(1_000.0)


def test_shared_cash_delisting_recovery_matches_hand_calculation() -> None:
    base = _market()
    asset_active = base.resolved_array("asset_active").copy()
    asset_active[2:, 0] = False
    tradable = base.tradable.copy()
    tradable[2:, 0] = False
    feature_available = base.feature_available.copy()
    feature_available[2:, 0] = False
    feature_staleness = base.resolved_array("feature_staleness").copy()
    feature_staleness[2:, 0] = 1.0
    information_available = base.resolved_array("information_available").copy()
    information_available[2:, 0] = False
    recovery = base.resolved_array("delisting_recovery").copy()
    recovery[2, 0] = 0.5
    dataset = replace(
        base,
        asset_active=asset_active,
        symbol_active=asset_active,
        tradable=tradable,
        feature_available=feature_available,
        feature_staleness=feature_staleness,
        information_available=information_available,
        delisting_recovery=recovery,
    )

    result = run_shared_cash_replay(
        dataset,
        _strategies(),
        start_index=0,
        stop_index=3,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=_risk(),
        settle_terminal_position=False,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    transition = next(
        transition
        for interval in ledger.intervals
        for transition in interval.accounting_transitions
        if transition.transition_type == "delisting_settlement"
    )
    assert transition.state_before.exact_quantities == ("5/2", "0")
    assert transition.state_after.exact_quantities == ("0", "0")
    expected_recovery = 2.5 * 100.0 * 0.5
    assert transition.state_after.cash - transition.state_before.cash == pytest.approx(
        expected_recovery
    )
    assert transition.state_before.cash + 2.5 * 100.0 == pytest.approx(1_000.0)
    assert transition.state_after.cash == pytest.approx(875.0)
    assert ledger.final_portfolio_value == pytest.approx(875.0)


def test_shared_cash_short_borrow_matches_hand_calculation() -> None:
    base = _market()
    borrow_rate = np.zeros((base.n_bars, base.n_symbols), dtype=np.float64)
    borrow_rate[2, 0] = 0.1
    dataset = replace(base, borrow_rate=borrow_rate)

    result = run_shared_cash_replay(
        dataset,
        (
            _ConstantIntent(PositionIntent.SHORT),
            _ConstantIntent(PositionIntent.FLAT),
        ),
        start_index=0,
        stop_index=3,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            borrow_rate_multiplier=1.0,
        ),
        risk=_risk(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    expected_borrow = 250.0 * 0.1 / 8_760.0
    borrow_transitions = [
        transition
        for interval in ledger.intervals
        for transition in interval.accounting_transitions
        if transition.transition_type == "borrow_charge"
    ]
    actual_borrow = sum(
        float(transition.evidence["borrow_amount"]) for transition in borrow_transitions
    )
    assert actual_borrow == pytest.approx(expected_borrow)
    assert ledger.final_borrow_cost == pytest.approx(expected_borrow)
    assert ledger.final_cash == pytest.approx(1_000.0 - expected_borrow)
    assert ledger.terminal_exact_quantities == ("0", "0")


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
