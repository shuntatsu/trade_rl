"""Unified metrics, comparisons, bootstrap, capacity and gates."""

from trade_rl.evaluation.bootstrap import BootstrapResult, moving_block_mean_test
from trade_rl.evaluation.capacity import (
    CapacityCurve,
    CapacityPoint,
    evaluate_capacity_grid,
)
from trade_rl.evaluation.causal_scenario_artifact import (
    CAUSAL_SCENARIO_ARRAYS_NAME,
    CAUSAL_SCENARIO_MANIFEST_NAME,
    CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA,
    CausalScenarioValueArtifactManifest,
    load_causal_scenario_value_artifact,
    write_causal_scenario_value_artifact,
)
from trade_rl.evaluation.causal_scenario_values import (
    CAUSAL_SCENARIO_EVALUATOR_SCHEMA,
    CausalQuerySnapshot,
    CausalScenarioEvaluationResult,
    CausalScenarioEvaluatorConfig,
    CausalScenarioSet,
    ProjectedResidualCandidate,
    ScenarioRolloutEvidence,
    evaluate_causal_scenario_actions,
    generate_residual_candidates,
)
from trade_rl.evaluation.closed_trades import (
    ClosedTradeDiagnostics,
    ClosedTradeTracker,
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
from trade_rl.evaluation.perfect_information_bound import (
    PERFECT_INFORMATION_BOUND_SCHEMA,
    PerfectInformationBoundConfig,
    PerfectInformationBoundResult,
    solve_perfect_information_bound,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries

__all__ = [
    "BootstrapResult",
    "CAUSAL_SCENARIO_ARRAYS_NAME",
    "CAUSAL_SCENARIO_EVALUATOR_SCHEMA",
    "CAUSAL_SCENARIO_MANIFEST_NAME",
    "CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA",
    "CapacityCurve",
    "CapacityPoint",
    "CausalQuerySnapshot",
    "CausalScenarioEvaluationResult",
    "CausalScenarioEvaluatorConfig",
    "CausalScenarioSet",
    "CausalScenarioValueArtifactManifest",
    "ClosedTradeDiagnostics",
    "ClosedTradeTracker",
    "ExecutionDiagnostics",
    "FRESH_CONFIRMATION_SCHEMA",
    "FreshConfirmationEvidence",
    "PAPER_RECONCILIATION_FILE_NAME",
    "PAPER_RECONCILIATION_SCHEMA",
    "PERFECT_INFORMATION_BOUND_SCHEMA",
    "PairedComparison",
    "PaperReconciliationEvidence",
    "PerfectInformationBoundConfig",
    "PerfectInformationBoundResult",
    "PerformanceMetrics",
    "ProjectedResidualCandidate",
    "ReturnKind",
    "ReturnSeries",
    "ScenarioRolloutEvidence",
    "compare_paired_returns",
    "compound_return",
    "evaluate_capacity_grid",
    "evaluate_causal_scenario_actions",
    "evaluate_performance",
    "generate_residual_candidates",
    "load_causal_scenario_value_artifact",
    "load_confirmation_evidence",
    "load_paper_reconciliation_evidence",
    "moving_block_mean_test",
    "resolve_gate",
    "solve_perfect_information_bound",
    "write_causal_scenario_value_artifact",
    "write_confirmation_evidence",
    "write_paper_reconciliation_evidence",
]
