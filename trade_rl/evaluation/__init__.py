"""Lean evaluation primitives for trading research."""

from trade_rl.evaluation.comparison.bootstrap import (
    BootstrapResult,
    moving_block_mean_test,
)
from trade_rl.evaluation.comparison.leaderboard import (
    FamilyLeaderboard,
    LeaderboardEntry,
    VarianceSource,
    build_family_leaderboard,
)
from trade_rl.evaluation.comparison.multiple_testing import (
    SharpeMoments,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_moments_from_returns,
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
    SharedCashStrategyComparisonEntry,
    StrategyComparison,
    StrategyComparisonEntry,
    SymbolStrategyComparison,
    UniversalStrategyComparison,
    compare_shared_cash_strategy_factories,
    compare_strategies,
    compare_strategies_by_symbol,
    compare_strategy_factories_by_symbol,
)
from trade_rl.evaluation.comparison.trial_ledger import (
    TrialLedger,
    TrialLedgerError,
    TrialRecord,
    TrialStatus,
    decode_trial_ledger_jsonl,
    encode_trial_ledger_jsonl,
)
from trade_rl.evaluation.decision_value_fit import (
    fit_conservative_linear_decision_value,
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
from trade_rl.evaluation.runs import (
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
    "FamilyLeaderboard",
    "IndependentFoldSummary",
    "LeaderboardEntry",
    "LeanCandidateConfig",
    "PERFECT_INFORMATION_BOUND_SCHEMA",
    "PairedComparison",
    "PerfectInformationBoundConfig",
    "PerfectInformationBoundResult",
    "PerformanceMetrics",
    "ReplayDecision",
    "ReturnKind",
    "ReturnSeries",
    "SeedEvaluation",
    "SeedResult",
    "SeedRobustnessSummary",
    "SharedCashStrategyComparisonEntry",
    "SharpeMoments",
    "SingleSymbolReplayResult",
    "StrategyComparison",
    "StrategyComparisonEntry",
    "SymbolStrategyComparison",
    "TrialLedger",
    "TrialLedgerError",
    "TrialRecord",
    "TrialStatus",
    "UniversalStrategyComparison",
    "VarianceSource",
    "build_family_leaderboard",
    "compare_paired_returns",
    "compare_shared_cash_strategy_factories",
    "compare_strategies",
    "compare_strategies_by_symbol",
    "compare_strategy_factories_by_symbol",
    "compound_return",
    "decode_trial_ledger_jsonl",
    "deflated_sharpe_ratio",
    "encode_trial_ledger_jsonl",
    "evaluate_capacity_grid",
    "evaluate_performance",
    "expected_maximum_sharpe",
    "fit_conservative_linear_decision_value",
    "moving_block_mean_test",
    "probabilistic_sharpe_ratio",
    "resolve_gate",
    "run_lean_candidate_suite",
    "run_single_symbol_replay",
    "sharpe_moments_from_returns",
    "solve_perfect_information_bound",
    "summarize_independent_folds",
    "summarize_seed_robustness",
]
