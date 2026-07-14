from __future__ import annotations

import pytest

from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.fold_metrics import summarize_independent_folds
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult


def _fold(
    index: int, start: int, values: tuple[float, ...], cost: float
) -> FoldOOSResult:
    return FoldOOSResult(
        fold_index=index,
        start=start,
        stop=start + len(values),
        returns=ReturnSeries(
            values=values, kind=ReturnKind.BASE_BAR, periods_per_year=365
        ),
        diagnostics=ExecutionDiagnostics(
            total_cost=cost, turnover_total=2 * cost, n_trades=index + 1
        ),
    )


def test_independent_fold_summary_preserves_execution_evidence() -> None:
    summary = summarize_independent_folds(
        (_fold(0, 10, (0.10, -0.05), 2.0), _fold(1, 20, (0.02, 0.03), 3.0))
    )
    assert summary.fold_count == 2
    assert summary.total_cost == 5.0
    assert summary.turnover_total == 10.0
    assert summary.n_trades == 3
    assert summary.worst_fold_return <= summary.median_fold_return


def test_independent_summary_does_not_claim_continuous_drawdown() -> None:
    summary = summarize_independent_folds((_fold(0, 10, (0.1,), 0.0),))
    with pytest.raises(ValueError, match="continuous-account"):
        summary.continuous_max_drawdown()
