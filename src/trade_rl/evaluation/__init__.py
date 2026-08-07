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
from trade_rl.evaluation.causal_scenario_c3_artifact import (
    C3_AGGREGATE_REPORT_ARTIFACT_SCHEMA,
    PHASE_A_GATE_ARTIFACT_SCHEMA,
    load_c3_aggregate_report_artifact,
    load_phase_a_gate_artifact,
    write_c3_aggregate_report_artifact,
    write_phase_a_gate_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    C3_CONFIG_SCHEMA,
    C3_DECISION_SCHEMA,
    C3_QUERY_COMPARISON_SCHEMA,
    C3_REALIZED_OUTCOME_SCHEMA,
    C3_REPLAY_IDENTITY_SCHEMA,
    C3ReplayIdentity,
    CausalScenarioC3Config,
    CausalScenarioQueryComparison,
    PerfectInformationComparison,
    PerfectInformationComparisonReason,
    PerfectInformationComparisonStatus,
    PersistedScenarioDecision,
    RealizedPolicyOutcome,
)
from trade_rl.evaluation.causal_scenario_c3_decision_artifact import (
    LoadedC3Decision,
    load_c3_decision_artifact,
    write_c3_decision_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_gate import (
    PHASE_A_ENTRY_GATE_SCHEMA,
    GateConditionResult,
    PhaseAEntryGateEvidence,
    evaluate_phase_a_entry_gate,
)
from trade_rl.evaluation.causal_scenario_c3_perfect_information import (
    PERFECT_INFORMATION_COMPATIBILITY_SCHEMA,
    PerfectInformationCompatibilityEvidence,
    evaluate_perfect_information_compatibility,
)
from trade_rl.evaluation.causal_scenario_c3_prediction import (
    C3_PREDICTION_EVIDENCE_SCHEMA,
    C3PredictionEvidence,
    build_c3_prediction_evidence,
    create_c3_prediction_evidence,
)
from trade_rl.evaluation.causal_scenario_c3_report import (
    C3_AGGREGATE_REPORT_SCHEMA,
    C3_CALIBRATION_BUCKET_SCHEMA,
    C3_EXECUTION_SUMMARY_SCHEMA,
    C3_FOLD_REPORT_SCHEMA,
    C3CalibrationBucket,
    C3PolicyExecutionSummary,
    CausalScenarioAggregateReport,
    CausalScenarioFoldReport,
    build_c3_aggregate_report,
    build_c3_fold_report,
)
from trade_rl.evaluation.causal_scenario_c3_runner import (
    build_persisted_scenario_decision,
    run_c3_query_comparison,
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
    "C3_AGGREGATE_REPORT_ARTIFACT_SCHEMA",
    "C3_AGGREGATE_REPORT_SCHEMA",
    "C3_CALIBRATION_BUCKET_SCHEMA",
    "C3_CONFIG_SCHEMA",
    "C3_DECISION_SCHEMA",
    "C3_EXECUTION_SUMMARY_SCHEMA",
    "C3_FOLD_REPORT_SCHEMA",
    "C3_PREDICTION_EVIDENCE_SCHEMA",
    "C3_QUERY_COMPARISON_SCHEMA",
    "C3_REALIZED_OUTCOME_SCHEMA",
    "C3_REPLAY_IDENTITY_SCHEMA",
    "C3CalibrationBucket",
    "C3PolicyExecutionSummary",
    "C3PredictionEvidence",
    "C3ReplayIdentity",
    "CAUSAL_SCENARIO_ARRAYS_NAME",
    "CAUSAL_SCENARIO_EVALUATOR_SCHEMA",
    "CAUSAL_SCENARIO_MANIFEST_NAME",
    "CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA",
    "CapacityCurve",
    "CapacityPoint",
    "CausalQuerySnapshot",
    "CausalScenarioAggregateReport",
    "CausalScenarioC3Config",
    "CausalScenarioEvaluationResult",
    "CausalScenarioEvaluatorConfig",
    "CausalScenarioFoldReport",
    "CausalScenarioQueryComparison",
    "CausalScenarioSet",
    "CausalScenarioValueArtifactManifest",
    "ClosedTradeDiagnostics",
    "ClosedTradeTracker",
    "ExecutionDiagnostics",
    "FRESH_CONFIRMATION_SCHEMA",
    "FreshConfirmationEvidence",
    "GateConditionResult",
    "LoadedC3Decision",
    "PAPER_RECONCILIATION_FILE_NAME",
    "PAPER_RECONCILIATION_SCHEMA",
    "PERFECT_INFORMATION_BOUND_SCHEMA",
    "PERFECT_INFORMATION_COMPATIBILITY_SCHEMA",
    "PHASE_A_ENTRY_GATE_SCHEMA",
    "PHASE_A_GATE_ARTIFACT_SCHEMA",
    "PairedComparison",
    "PaperReconciliationEvidence",
    "PerfectInformationBoundConfig",
    "PerfectInformationBoundResult",
    "PerfectInformationComparison",
    "PerfectInformationComparisonReason",
    "PerfectInformationComparisonStatus",
    "PerfectInformationCompatibilityEvidence",
    "PerformanceMetrics",
    "PersistedScenarioDecision",
    "PhaseAEntryGateEvidence",
    "ProjectedResidualCandidate",
    "RealizedPolicyOutcome",
    "ReturnKind",
    "ReturnSeries",
    "ScenarioRolloutEvidence",
    "build_c3_aggregate_report",
    "build_c3_fold_report",
    "build_c3_prediction_evidence",
    "build_persisted_scenario_decision",
    "compare_paired_returns",
    "compound_return",
    "create_c3_prediction_evidence",
    "evaluate_capacity_grid",
    "evaluate_causal_scenario_actions",
    "evaluate_perfect_information_compatibility",
    "evaluate_performance",
    "evaluate_phase_a_entry_gate",
    "generate_residual_candidates",
    "load_c3_aggregate_report_artifact",
    "load_c3_decision_artifact",
    "load_causal_scenario_value_artifact",
    "load_confirmation_evidence",
    "load_paper_reconciliation_evidence",
    "load_phase_a_gate_artifact",
    "moving_block_mean_test",
    "resolve_gate",
    "run_c3_query_comparison",
    "solve_perfect_information_bound",
    "write_c3_aggregate_report_artifact",
    "write_c3_decision_artifact",
    "write_causal_scenario_value_artifact",
    "write_confirmation_evidence",
    "write_paper_reconciliation_evidence",
    "write_phase_a_gate_artifact",
]
