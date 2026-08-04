"""Compatibility exports for lower portfolio performance contracts."""

from trade_rl.simulation.performance import (
    PerformanceMetrics,
    compound_return,
    evaluate_performance,
)

__all__ = [
    "PerformanceMetrics",
    "compound_return",
    "evaluate_performance",
]
