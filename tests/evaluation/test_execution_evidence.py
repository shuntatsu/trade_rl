from __future__ import annotations

import pytest

from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult, stitch_oos


def returns(*values: float) -> ReturnSeries:
    return ReturnSeries(
        values=values,
        kind=ReturnKind.BASE_BAR,
        periods_per_year=8_760,
    )


def test_execution_diagnostics_combine_without_losing_economic_evidence() -> None:
    combined = ExecutionDiagnostics.combine(
        (
            ExecutionDiagnostics(
                turnover_total=0.3,
                total_cost=12.0,
                funding_pnl=-2.0,
                borrow_cost=1.5,
                n_trades=2,
                rebalance_events=1,
                termination_reasons=("margin_call",),
            ),
            ExecutionDiagnostics(
                turnover_total=0.4,
                total_cost=8.0,
                funding_pnl=3.0,
                borrow_cost=0.5,
                n_trades=3,
                rebalance_events=2,
                termination_reasons=("drawdown_stop",),
            ),
        )
    )
    assert combined.turnover_total == pytest.approx(0.7)
    assert combined.total_cost == pytest.approx(20.0)
    assert combined.funding_pnl == pytest.approx(1.0)
    assert combined.borrow_cost == pytest.approx(2.0)
    assert combined.n_trades == 5
    assert combined.rebalance_events == 3
    assert combined.termination_reasons == ("margin_call", "drawdown_stop")


def test_stitching_and_metrics_preserve_execution_diagnostics() -> None:
    stitched = stitch_oos(
        (
            FoldOOSResult(
                fold_index=0,
                start=10,
                stop=12,
                returns=returns(0.01, -0.005),
                diagnostics=ExecutionDiagnostics(
                    turnover_total=0.2,
                    total_cost=5.0,
                    funding_pnl=-1.0,
                    borrow_cost=0.25,
                    n_trades=2,
                    rebalance_events=1,
                ),
            ),
            FoldOOSResult(
                fold_index=1,
                start=12,
                stop=14,
                returns=returns(0.02, 0.0),
                diagnostics=ExecutionDiagnostics(
                    turnover_total=0.4,
                    total_cost=7.0,
                    funding_pnl=2.0,
                    borrow_cost=0.75,
                    n_trades=3,
                    rebalance_events=2,
                    termination_reasons=("minimum_equity",),
                ),
            ),
        )
    )
    diagnostics = stitched.diagnostics
    metrics = evaluate_performance(
        stitched.returns,
        turnover_total=diagnostics.turnover_total,
        total_cost=diagnostics.total_cost,
        funding_pnl=diagnostics.funding_pnl,
        borrow_cost=diagnostics.borrow_cost,
        n_trades=diagnostics.n_trades,
        rebalance_events=diagnostics.rebalance_events,
        termination_count=diagnostics.termination_count,
    )
    assert metrics.turnover_total == pytest.approx(0.6)
    assert metrics.total_cost == pytest.approx(12.0)
    assert metrics.funding_pnl == pytest.approx(1.0)
    assert metrics.borrow_cost == pytest.approx(1.0)
    assert metrics.n_trades == 5
    assert metrics.rebalance_events == 3
    assert metrics.termination_count == 1
