from __future__ import annotations

from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
RUNS = PACKAGE / "evaluation" / "runs"

RUNS_EXPORTS = {
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
}

FORBIDDEN_OWNER_MODULES = (
    "trade_rl.evaluation.runs.artifact",
    "trade_rl.evaluation.runs.candidate_suite",
    "trade_rl.evaluation.runs.config",
    "trade_rl.evaluation.runs.execute",
    "trade_rl.evaluation.runs.provenance",
)

EVALUATION_EXPORTS = {
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
}


def test_runs_facade_exports_exact_surface_and_owner_identity() -> None:
    import trade_rl.evaluation.runs as facade
    from trade_rl.evaluation.runs import (
        artifact,
        candidate_suite,
        config,
        execute,
        provenance,
    )

    assert set(facade.__all__) == RUNS_EXPORTS
    assert facade.CandidateRunConfig is config.CandidateRunConfig
    assert facade.ResolvedCandidateRunSpec is config.ResolvedCandidateRunSpec
    assert facade.parse_candidate_run_config is config.parse_candidate_run_config
    assert facade.resolve_candidate_run_spec is config.resolve_candidate_run_spec
    assert facade.LeanCandidateConfig is candidate_suite.LeanCandidateConfig
    assert facade.run_lean_candidate_suite is candidate_suite.run_lean_candidate_suite
    assert facade.CandidateRunResult is execute.CandidateRunResult
    assert facade.execute_candidate_run is execute.execute_candidate_run
    assert facade.CandidateRunArtifactIdentity is artifact.CandidateRunArtifactIdentity
    assert facade.LoadedCandidateRun is artifact.LoadedCandidateRun
    assert facade.PublishedCandidateRun is artifact.PublishedCandidateRun
    assert (
        facade.inspect_candidate_run_artifact is artifact.inspect_candidate_run_artifact
    )
    assert facade.load_candidate_run_artifact is artifact.load_candidate_run_artifact
    assert facade.publish_candidate_run is artifact.publish_candidate_run
    assert (
        facade.build_candidate_run_provenance
        is provenance.build_candidate_run_provenance
    )
    assert not hasattr(facade, "load_candidate_run_config")
    assert not hasattr(facade, "PROVENANCE_SCHEMA")


def test_production_outside_runs_imports_run_core_through_facade() -> None:
    collector = ImportCollector(PACKAGE)
    offenders: list[tuple[str, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.is_relative_to(RUNS):
            continue
        for imported in sorted(collector.collect_direct(path)):
            if any(within_module(imported, owner) for owner in FORBIDDEN_OWNER_MODULES):
                offenders.append((path.relative_to(ROOT).as_posix(), imported))
    assert offenders == []


def test_evaluation_tier1_public_api_is_unchanged() -> None:
    import trade_rl.evaluation as evaluation

    assert set(evaluation.__all__) == EVALUATION_EXPORTS
