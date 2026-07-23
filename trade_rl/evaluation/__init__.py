"""Unified metrics, comparisons, bootstrap, capacity and gates."""

from trade_rl.evaluation.bootstrap import BootstrapResult, moving_block_mean_test
from trade_rl.evaluation.capacity import (
    CapacityCurve,
    CapacityPoint,
    evaluate_capacity_grid,
)
from trade_rl.evaluation.comparisons import PairedComparison, compare_paired_returns
from trade_rl.evaluation.confirmation import (
    FRESH_CONFIRMATION_SCHEMA,
    FreshConfirmationEvidence,
    load_confirmation_evidence,
    write_confirmation_evidence,
)
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.gates import resolve_gate
from trade_rl.evaluation.metrics import (
    PerformanceMetrics,
    compound_return,
    evaluate_performance,
)
from trade_rl.evaluation.paper_reconciliation import (
    PAPER_RECONCILIATION_FILE_NAME,
    PAPER_RECONCILIATION_SCHEMA,
    PaperReconciliationEvidence,
    load_paper_reconciliation_evidence,
    write_paper_reconciliation_evidence,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries

__all__ = [
    "BootstrapResult",
    "CapacityCurve",
    "CapacityPoint",
    "ExecutionDiagnostics",
    "FRESH_CONFIRMATION_SCHEMA",
    "FreshConfirmationEvidence",
    "PAPER_RECONCILIATION_FILE_NAME",
    "PAPER_RECONCILIATION_SCHEMA",
    "PairedComparison",
    "PaperReconciliationEvidence",
    "PerformanceMetrics",
    "ReturnKind",
    "ReturnSeries",
    "compare_paired_returns",
    "compound_return",
    "evaluate_capacity_grid",
    "evaluate_performance",
    "load_confirmation_evidence",
    "load_paper_reconciliation_evidence",
    "moving_block_mean_test",
    "resolve_gate",
    "write_confirmation_evidence",
    "write_paper_reconciliation_evidence",
]
