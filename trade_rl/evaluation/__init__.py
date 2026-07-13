"""Unified metrics, comparisons, bootstrap, and gates."""

from trade_rl.evaluation.bootstrap import BootstrapResult, moving_block_mean_test
from trade_rl.evaluation.comparisons import PairedComparison, compare_paired_returns
from trade_rl.evaluation.gates import resolve_gate
from trade_rl.evaluation.metrics import (
    PerformanceMetrics,
    compound_return,
    evaluate_performance,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries

__all__ = [
    "BootstrapResult",
    "PairedComparison",
    "PerformanceMetrics",
    "ReturnKind",
    "ReturnSeries",
    "compare_paired_returns",
    "compound_return",
    "evaluate_performance",
    "moving_block_mean_test",
    "resolve_gate",
]
