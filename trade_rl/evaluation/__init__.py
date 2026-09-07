"""Lean evaluation primitives for trading research."""

from trade_rl.evaluation.bootstrap import BootstrapResult, moving_block_mean_test
from trade_rl.evaluation.capacity import CapacityCurve, CapacityPoint, evaluate_capacity_grid
from trade_rl.evaluation.closed_trades import ClosedTradeDiagnostics, ClosedTradeTracker
from trade_rl.evaluation.comparisons import PairedComparison, compare_paired_returns
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.fold_metrics import IndependentFoldSummary, summarize_independent_folds
from trade_rl.evaluation.gates import resolve_gate
from trade_rl.evaluation.metrics import PerformanceMetrics, compound_return, evaluate_performance
from trade_rl.evaluation.perfect_information_bound import (
    PERFECT_INFORMATION_BOUND_SCHEMA,
    PerfectInformationBoundConfig,
    PerfectInformationBoundResult,
    solve_perfect_information_bound,
)
from trade_rl.evaluation.seed_robustness import (
    SeedEvaluation,
    SeedResult,
    SeedRobustnessSummary,
    summarize_seed_robustness,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries

__all__ = [
    "BootstrapResult",
    "CapacityCurve",
    "CapacityPoint",
    "ClosedTradeDiagnostics",
    "ClosedTradeTracker",
    "ExecutionDiagnostics",
    "IndependentFoldSummary",
    "PERFECT_INFORMATION_BOUND_SCHEMA",
    "PairedComparison",
    "PerformanceMetrics",
    "PerfectInformationBoundConfig",
    "PerfectInformationBoundResult",
    "ReturnKind",
    "ReturnSeries",
    "SeedEvaluation",
    "SeedResult",
    "SeedRobustnessSummary",
    "compare_paired_returns",
    "compound_return",
    "evaluate_capacity_grid",
    "evaluate_performance",
    "moving_block_mean_test",
    "resolve_gate",
    "solve_perfect_information_bound",
    "summarize_independent_folds",
    "summarize_seed_robustness",
]
