from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "tests/evaluation/test_walk_forward_metric_semantics.py"


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_tests() -> None:
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(
        '''from __future__ import annotations

import pytest

from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult, StitchMode, stitch_oos
from trade_rl.workflows.walk_forward import evaluate_walk_forward_performance


def series(*values: float) -> ReturnSeries:
    return ReturnSeries(values=values, kind=ReturnKind.BASE_BAR, periods_per_year=365)


def test_independent_folds_do_not_report_cross_reset_compounding() -> None:
    folds = (
        FoldOOSResult(fold_index=0, start=10, stop=11, returns=series(0.10)),
        FoldOOSResult(fold_index=1, start=11, stop=12, returns=series(0.10)),
    )
    stitched = stitch_oos(folds, mode=StitchMode.INDEPENDENT_FOLDS)
    metrics = evaluate_walk_forward_performance(stitched, folds)

    assert metrics.total_return == pytest.approx(0.10)
    assert metrics.metric_semantics == "independent_fold_distribution"
    assert metrics.fold_count == 2
    assert metrics.worst_fold_return == pytest.approx(0.10)
    assert metrics.positive_fold_fraction == pytest.approx(1.0)


def test_continuous_account_retains_true_compounding() -> None:
    folds = (
        FoldOOSResult(
            fold_index=0,
            start=10,
            stop=11,
            returns=series(0.10),
            opening_state_digest="open",
            closing_state_digest="handoff",
        ),
        FoldOOSResult(
            fold_index=1,
            start=11,
            stop=12,
            returns=series(0.10),
            opening_state_digest="handoff",
            closing_state_digest="close",
        ),
    )
    stitched = stitch_oos(folds, mode=StitchMode.CONTINUOUS_ACCOUNT)
    metrics = evaluate_walk_forward_performance(stitched, folds)

    assert metrics.total_return == pytest.approx(0.21)
    assert metrics.metric_semantics == "continuous_account_series"


def test_independent_fold_coverage_exposes_gaps() -> None:
    folds = (
        FoldOOSResult(fold_index=0, start=10, stop=12, returns=series(0.01, 0.02)),
        FoldOOSResult(fold_index=1, start=16, stop=18, returns=series(-0.01, 0.03)),
    )
    stitched = stitch_oos(folds, mode=StitchMode.INDEPENDENT_FOLDS)
    metrics = evaluate_walk_forward_performance(stitched, folds)

    assert stitched.calendar_span_periods == 8
    assert stitched.observed_periods == 4
    assert stitched.coverage_fraction == pytest.approx(0.5)
    assert metrics.coverage_fraction == pytest.approx(0.5)
    assert metrics.worst_fold_return < 0.03
''',
        encoding="utf-8",
    )


def apply_implementation() -> None:
    replace_once(
        "trade_rl/evaluation/metrics.py",
        '''    return_kind: ReturnKind
    periods_per_year: int
''',
        '''    return_kind: ReturnKind
    periods_per_year: int
    metric_semantics: str = "continuous_series"
    fold_count: int = 1
    coverage_fraction: float = 1.0
    worst_fold_return: float | None = None
    positive_fold_fraction: float | None = None
''',
    )
    replace_once(
        "trade_rl/evaluation/metrics.py",
        '''        return_kind=returns.kind,
        periods_per_year=returns.periods_per_year,
    )
''',
        '''        return_kind=returns.kind,
        periods_per_year=returns.periods_per_year,
        metric_semantics="continuous_series",
    )
''',
    )

    replace_once(
        "trade_rl/evaluation/walk_forward/stitching.py",
        '''    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)


def stitch_oos(
''',
        '''    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)

    @property
    def observed_periods(self) -> int:
        return len(self.returns.values)

    @property
    def calendar_span_periods(self) -> int:
        return self.boundaries[-1][1] - self.boundaries[0][0]

    @property
    def coverage_fraction(self) -> float:
        return self.observed_periods / self.calendar_span_periods


def stitch_oos(
''',
    )

    replace_once(
        "trade_rl/workflows/walk_forward.py",
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, replace\nfrom statistics import fmean\n",
    )
    marker = '''def run_walk_forward(
    config: WalkForwardWorkflowConfig,
'''
    helper = '''def _diagnostic_performance(stitched: StitchedOOS) -> PerformanceMetrics:
    diagnostics = stitched.diagnostics
    return evaluate_performance(
        stitched.returns,
        turnover_total=diagnostics.turnover_total,
        total_cost=diagnostics.total_cost,
        funding_pnl=diagnostics.funding_pnl,
        borrow_cost=diagnostics.borrow_cost,
        n_trades=diagnostics.n_trades,
        rebalance_events=diagnostics.rebalance_events,
        termination_count=diagnostics.termination_count,
    )


def evaluate_walk_forward_performance(
    stitched: StitchedOOS,
    folds: tuple[FoldOOSResult, ...],
) -> PerformanceMetrics:
    """Evaluate continuous accounts or aggregate independent fold distributions."""

    if not folds:
        raise ValueError("walk-forward performance requires fold results")
    if stitched.mode is StitchMode.CONTINUOUS_ACCOUNT:
        return replace(
            _diagnostic_performance(stitched),
            metric_semantics="continuous_account_series",
            fold_count=len(folds),
            coverage_fraction=stitched.coverage_fraction,
        )
    fold_metrics = tuple(
        evaluate_performance(
            fold.returns,
            turnover_total=fold.diagnostics.turnover_total,
            total_cost=fold.diagnostics.total_cost,
            funding_pnl=fold.diagnostics.funding_pnl,
            borrow_cost=fold.diagnostics.borrow_cost,
            n_trades=fold.diagnostics.n_trades,
            rebalance_events=fold.diagnostics.rebalance_events,
            termination_count=fold.diagnostics.termination_count,
        )
        for fold in folds
    )
    fold_returns = tuple(metric.total_return for metric in fold_metrics)
    diagnostics = stitched.diagnostics
    first = fold_metrics[0]
    return PerformanceMetrics(
        total_return=fmean(fold_returns),
        sharpe=fmean(metric.sharpe for metric in fold_metrics),
        sortino=fmean(metric.sortino for metric in fold_metrics),
        max_drawdown=max(metric.max_drawdown for metric in fold_metrics),
        turnover_total=diagnostics.turnover_total,
        total_cost=diagnostics.total_cost,
        funding_pnl=diagnostics.funding_pnl,
        borrow_cost=diagnostics.borrow_cost,
        n_trades=diagnostics.n_trades,
        rebalance_events=diagnostics.rebalance_events,
        termination_count=diagnostics.termination_count,
        n_periods=sum(metric.n_periods for metric in fold_metrics),
        return_kind=first.return_kind,
        periods_per_year=first.periods_per_year,
        metric_semantics="independent_fold_distribution",
        fold_count=len(folds),
        coverage_fraction=stitched.coverage_fraction,
        worst_fold_return=min(fold_returns),
        positive_fold_fraction=sum(value > 0.0 for value in fold_returns) / len(fold_returns),
    )


'''
    replace_once("trade_rl/workflows/walk_forward.py", marker, helper + marker)
    replace_once(
        "trade_rl/workflows/walk_forward.py",
        '''        metrics=evaluate_performance(
            stitched.returns,
            turnover_total=stitched.diagnostics.turnover_total,
            total_cost=stitched.diagnostics.total_cost,
            funding_pnl=stitched.diagnostics.funding_pnl,
            borrow_cost=stitched.diagnostics.borrow_cost,
            n_trades=stitched.diagnostics.n_trades,
            rebalance_events=stitched.diagnostics.rebalance_events,
            termination_count=stitched.diagnostics.termination_count,
        ),
''',
        '''        metrics=evaluate_walk_forward_performance(stitched, tuple(results)),
''',
    )
    replace_once(
        "trade_rl/workflows/walk_forward.py",
        '''        selected_metrics=evaluate_performance(
            selected_stitched.returns,
            turnover_total=selected_stitched.diagnostics.turnover_total,
            total_cost=selected_stitched.diagnostics.total_cost,
            funding_pnl=selected_stitched.diagnostics.funding_pnl,
            borrow_cost=selected_stitched.diagnostics.borrow_cost,
            n_trades=selected_stitched.diagnostics.n_trades,
            rebalance_events=selected_stitched.diagnostics.rebalance_events,
            termination_count=selected_stitched.diagnostics.termination_count,
        ),
        baseline_metrics=evaluate_performance(
            baseline_stitched.returns,
            turnover_total=baseline_stitched.diagnostics.turnover_total,
            total_cost=baseline_stitched.diagnostics.total_cost,
            funding_pnl=baseline_stitched.diagnostics.funding_pnl,
            borrow_cost=baseline_stitched.diagnostics.borrow_cost,
            n_trades=baseline_stitched.diagnostics.n_trades,
            rebalance_events=baseline_stitched.diagnostics.rebalance_events,
            termination_count=baseline_stitched.diagnostics.termination_count,
        ),
''',
        '''        selected_metrics=evaluate_walk_forward_performance(
            selected_stitched,
            tuple(selected_results),
        ),
        baseline_metrics=evaluate_walk_forward_performance(
            baseline_stitched,
            tuple(baseline_results),
        ),
''',
    )
    replace_once(
        "trade_rl/workflows/walk_forward.py",
        '''            "selected_diagnostics": selected_stitched.diagnostics.digest_payload(),
            "stitch_mode": config.stitch_mode.value,
''',
        '''            "selected_diagnostics": selected_stitched.diagnostics.digest_payload(),
            "selected_metric_semantics": (
                "continuous_account_series"
                if config.stitch_mode is StitchMode.CONTINUOUS_ACCOUNT
                else "independent_fold_distribution"
            ),
            "stitch_mode": config.stitch_mode.value,
''',
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: fold_metrics.py tests|implementation")
    if sys.argv[1] == "tests":
        write_tests()
    else:
        apply_implementation()


if __name__ == "__main__":
    main()
