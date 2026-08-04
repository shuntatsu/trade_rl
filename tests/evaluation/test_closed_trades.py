from __future__ import annotations

import pytest

from trade_rl.evaluation.closed_trades import (
    ClosedTradeDiagnostics,
    ClosedTradeTracker,
)


def test_tracker_groups_partial_closes_into_one_position_cycle() -> None:
    tracker = ClosedTradeTracker([1.0])

    tracker.record_fill(symbol=0, quantity=10.0, price=100.0, execution_cost=10.0)
    tracker.record_fill(symbol=0, quantity=-4.0, price=110.0, execution_cost=4.0)
    assert tracker.diagnostics().closed_trades == 0
    tracker.record_fill(symbol=0, quantity=-6.0, price=90.0, execution_cost=6.0)

    result = tracker.diagnostics()
    assert result.closed_trades == 1
    assert result.winning_trades == 0
    assert result.losing_trades == 1
    assert result.net_realized_pnl == pytest.approx(-40.0)
    assert result.gross_loss == pytest.approx(40.0)
    assert result.win_rate == pytest.approx(0.0)
    assert result.profit_factor == pytest.approx(0.0)
    assert result.open_positions == 0


def test_tracker_closes_and_opens_cycles_at_a_sign_reversal() -> None:
    tracker = ClosedTradeTracker([1.0])

    tracker.record_fill(symbol=0, quantity=2.0, price=100.0, execution_cost=2.0)
    tracker.record_fill(symbol=0, quantity=-3.0, price=110.0, execution_cost=3.0)
    tracker.record_fill(symbol=0, quantity=1.0, price=100.0, execution_cost=1.0)

    result = tracker.diagnostics()
    assert result.closed_trades == 2
    assert result.winning_trades == 2
    assert result.net_realized_pnl == pytest.approx(24.0)
    assert result.gross_profit == pytest.approx(24.0)
    assert result.gross_loss == pytest.approx(0.0)
    assert result.win_rate == pytest.approx(1.0)
    assert result.average_realized_pnl == pytest.approx(12.0)
    assert result.profit_factor is None
    assert result.open_positions == 0


def test_tracker_seeds_positions_at_the_evaluation_boundary() -> None:
    tracker = ClosedTradeTracker([1.0])
    tracker.seed_positions(quantities=[2.0], prices=[100.0])

    tracker.record_fill(symbol=0, quantity=-2.0, price=110.0, execution_cost=2.0)

    result = tracker.diagnostics()
    assert result.closed_trades == 1
    assert result.winning_trades == 1
    assert result.net_realized_pnl == pytest.approx(18.0)
    assert result.open_positions == 0


def test_tracker_rejects_reseeding_after_a_fill() -> None:
    tracker = ClosedTradeTracker([1.0])
    tracker.record_fill(symbol=0, quantity=1.0, price=100.0, execution_cost=0.0)

    with pytest.raises(RuntimeError, match="already initialized"):
        tracker.seed_positions(quantities=[1.0], prices=[100.0])


def test_closed_trade_diagnostics_combine_recomputes_derived_metrics() -> None:
    combined = ClosedTradeDiagnostics.combine(
        (
            ClosedTradeDiagnostics(
                closed_trades=1,
                winning_trades=1,
                net_realized_pnl=5.0,
                gross_profit=5.0,
            ),
            ClosedTradeDiagnostics(
                closed_trades=2,
                losing_trades=1,
                breakeven_trades=1,
                net_realized_pnl=-2.0,
                gross_loss=2.0,
                open_positions=1,
            ),
        )
    )

    assert combined.closed_trades == 3
    assert combined.win_rate == pytest.approx(1.0 / 3.0)
    assert combined.average_realized_pnl == pytest.approx(1.0)
    assert combined.profit_factor == pytest.approx(2.5)
    assert combined.open_positions == 1
    assert combined.digest_payload()["profit_factor"] == pytest.approx(2.5)
