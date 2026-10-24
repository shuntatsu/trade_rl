"""Tier-2 public facade for immutable candidate-run contracts and execution."""

from trade_rl.evaluation.runs.artifact import (
    CandidateRunArtifactIdentity,
    LoadedCandidateRun,
    PublishedCandidateRun,
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
    supports_shared_cash_result_schema,
)
from trade_rl.evaluation.runs.candidate_suite import (
    LeanCandidateConfig,
    run_lean_candidate_suite,
)
from trade_rl.evaluation.runs.config import (
    CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY as CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
)
from trade_rl.evaluation.runs.config import (
    LEGACY_DATASET_EXECUTION_OVERLAY as LEGACY_DATASET_EXECUTION_OVERLAY,
)
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    ResolvedCandidateRunSpec,
    parse_candidate_run_config,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import CandidateRunResult, execute_candidate_run
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
from trade_rl.evaluation.runs.ridge_artifact_comparison import (
    RidgeArtifactComparison,
    RidgeArtifactComparisonConfig,
    compare_ridge_artifact_shared_cash,
)
from trade_rl.evaluation.runs.ridge_comparison_artifact import (
    LoadedRidgeComparisonArtifact,
    load_ridge_comparison_artifact,
    publish_ridge_comparison_artifact,
)
from trade_rl.evaluation.runs.ridge_comparison_attempt import (
    PublishedRidgeComparisonAttempt,
    run_ridge_comparison_attempt,
)

__all__ = [
    "CandidateRunArtifactIdentity",
    "CandidateRunConfig",
    "CandidateRunResult",
    "LeanCandidateConfig",
    "LoadedRidgeComparisonArtifact",
    "LoadedCandidateRun",
    "PublishedCandidateRun",
    "PublishedRidgeComparisonAttempt",
    "ResolvedCandidateRunSpec",
    "RidgeArtifactComparison",
    "RidgeArtifactComparisonConfig",
    "build_candidate_run_provenance",
    "compare_ridge_artifact_shared_cash",
    "execute_candidate_run",
    "inspect_candidate_run_artifact",
    "load_candidate_run_artifact",
    "load_ridge_comparison_artifact",
    "parse_candidate_run_config",
    "publish_ridge_comparison_artifact",
    "publish_candidate_run",
    "resolve_candidate_run_spec",
    "run_lean_candidate_suite",
    "run_ridge_comparison_attempt",
    "supports_shared_cash_result_schema",
]
