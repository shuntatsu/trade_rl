"""Lean evaluation primitives for trading research."""

from trade_rl.evaluation.comparison.bootstrap import (
    BootstrapResult,
    moving_block_mean_test,
)
from trade_rl.evaluation.comparison.paired import (
    PairedComparison,
    compare_paired_returns,
)
from trade_rl.evaluation.comparison.seed_robustness import (
    SeedEvaluation,
    SeedResult,
    SeedRobustnessSummary,
    summarize_seed_robustness,
)
from trade_rl.evaluation.comparison.strategies import (
    StrategyComparison,
    StrategyComparisonEntry,
    SymbolStrategyComparison,
    UniversalStrategyComparison,
    compare_strategies,
    compare_strategies_by_symbol,
)
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.gates import resolve_gate
from trade_rl.evaluation.metrics import (
    PerformanceMetrics,
    compound_return,
    evaluate_performance,
)
from trade_rl.evaluation.replay import (
    ReplayDecision,
    SingleSymbolReplayResult,
    run_single_symbol_replay,
)
from trade_rl.evaluation.robustness.capacity import (
    CapacityCurve,
    CapacityPoint,
    evaluate_capacity_grid,
)
from trade_rl.evaluation.robustness.closed_trades import (
    ClosedTradeDiagnostics,
    ClosedTradeTracker,
)
from trade_rl.evaluation.robustness.fold_metrics import (
    IndependentFoldSummary,
    summarize_independent_folds,
)
from trade_rl.evaluation.robustness.perfect_information.bound import (
    PERFECT_INFORMATION_BOUND_SCHEMA,
    PerfectInformationBoundConfig,
    PerfectInformationBoundResult,
    solve_perfect_information_bound,
)
from trade_rl.evaluation.runs.candidate_suite import (
    LeanCandidateConfig,
    run_lean_candidate_suite,
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
    "LeanCandidateConfig",
    "PERFECT_INFORMATION_BOUND_SCHEMA",
    "PairedComparison",
    "PerformanceMetrics",
    "PerfectInformationBoundConfig",
    "PerfectInformationBoundResult",
    "ReplayDecision",
    "ReturnKind",
    "ReturnSeries",
    "SeedEvaluation",
    "SeedResult",
    "SeedRobustnessSummary",
    "SingleSymbolReplayResult",
    "StrategyComparison",
    "StrategyComparisonEntry",
    "SymbolStrategyComparison",
    "UniversalStrategyComparison",
    "compare_paired_returns",
    "compare_strategies",
    "compare_strategies_by_symbol",
    "compound_return",
    "evaluate_capacity_grid",
    "evaluate_performance",
    "moving_block_mean_test",
    "resolve_gate",
    "run_lean_candidate_suite",
    "run_single_symbol_replay",
    "solve_perfect_information_bound",
    "summarize_independent_folds",
    "summarize_seed_robustness",
]
