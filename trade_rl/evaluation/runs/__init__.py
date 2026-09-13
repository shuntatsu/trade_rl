"""Tier-2 public facade for immutable candidate-run contracts and execution."""

from trade_rl.evaluation.runs.artifact import (
    CandidateRunArtifactIdentity,
    LoadedCandidateRun,
    PublishedCandidateRun,
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
)
from trade_rl.evaluation.runs.candidate_suite import (
    LeanCandidateConfig,
    run_lean_candidate_suite,
)
from trade_rl.evaluation.runs.config import (
    CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
    LEGACY_DATASET_EXECUTION_OVERLAY,
    CandidateRunConfig,
    ResolvedCandidateRunSpec,
    parse_candidate_run_config,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import CandidateRunResult, execute_candidate_run
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance

__all__ = [
    "CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY",
    "LEGACY_DATASET_EXECUTION_OVERLAY",
    "CandidateRunArtifactIdentity",
    "CandidateRunConfig",
    "CandidateRunResult",
    "LeanCandidateConfig",
    "LoadedCandidateRun",
    "PublishedCandidateRun",
    "ResolvedCandidateRunSpec",
    "build_candidate_run_provenance",
    "execute_candidate_run",
    "inspect_candidate_run_artifact",
    "load_candidate_run_artifact",
    "parse_candidate_run_config",
    "publish_candidate_run",
    "resolve_candidate_run_spec",
    "run_lean_candidate_suite",
]
