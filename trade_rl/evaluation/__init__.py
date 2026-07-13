"""Unified metrics, comparisons, bootstrap, capacity and gates."""

from trade_rl.evaluation.bootstrap import BootstrapResult, moving_block_mean_test
from trade_rl.evaluation.capacity import CapacityCurve, CapacityPoint, evaluate_capacity_grid
from trade_rl.evaluation.comparisons import PairedComparison, compare_paired_returns
from trade_rl.evaluation.gates import resolve_gate
from trade_rl.evaluation.metrics import PerformanceMetrics, compound_return, evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries

__all__ = [
    "BootstrapResult",
    "CapacityCurve",
    "CapacityPoint",
    "PairedComparison",
    "PerformanceMetrics",
    "ReturnKind",
    "ReturnSeries",
    "compare_paired_returns",
    "compound_return",
    "evaluate_capacity_grid",
    "evaluate_performance",
    "moving_block_mean_test",
    "resolve_gate",
]
