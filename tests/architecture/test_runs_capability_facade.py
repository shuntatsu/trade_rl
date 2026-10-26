from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
RUNS = PACKAGE / "evaluation" / "runs"

RUNS_EXPORTS = {
    "CandidateRunArtifactIdentity",
    "CandidateRunConfig",
    "CandidateRunResult",
    "ForecastErrorMetrics",
    "ForecastSymbolDiagnostics",
    "FourFamilyPortfolioAttemptContext",
    "FourFamilyPortfolioAttemptIdentity",
    "FourFamilyPortfolioReplayResult",
    "LeanCandidateConfig",
    "LoadedFourFamilyPortfolioAttempt",
    "LoadedFourFamilyPortfolioResultArtifact",
    "LoadedRidgeComparisonArtifact",
    "LoadedCandidateRun",
    "PublishedCandidateRun",
    "PublishedFourFamilyPortfolioAttempt",
    "PublishedRidgeComparisonAttempt",
    "PortfolioFamilyReplayOutcome",
    "ResolvedCandidateRunSpec",
    "RidgeArtifactComparison",
    "RidgeArtifactComparisonConfig",
    "RidgeForecastDiagnostics",
    "build_candidate_run_provenance",
    "compare_ridge_artifact_shared_cash",
    "execute_candidate_run",
    "inspect_candidate_run_artifact",
    "load_candidate_run_artifact",
    "load_four_family_portfolio_attempt",
    "load_four_family_portfolio_result_artifact",
    "load_ridge_comparison_artifact",
    "parse_candidate_run_config",
    "publish_candidate_run",
    "publish_ridge_comparison_artifact",
    "resolve_candidate_run_spec",
    "run_lean_candidate_suite",
    "run_four_family_shared_portfolio_replay",
    "run_four_family_portfolio_attempt",
    "run_ridge_comparison_attempt",
    "supports_shared_cash_result_schema",
}

FORBIDDEN_OWNER_MODULES = (
    "trade_rl.evaluation.runs.artifact",
    "trade_rl.evaluation.runs.candidate_suite",
    "trade_rl.evaluation.runs.config",
    "trade_rl.evaluation.runs.execute",
    "trade_rl.evaluation.runs.provenance",
    "trade_rl.evaluation.runs.ridge_artifact_comparison",
    "trade_rl.evaluation.runs.ridge_comparison_attempt",
    "trade_rl.evaluation.runs.portfolio_family_comparison",
    "trade_rl.evaluation.runs.ridge_comparison_artifact",
    "trade_rl.evaluation.runs.portfolio_family_attempt",
    "trade_rl.evaluation.runs.portfolio_family_result_artifact",
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
    "SharedCashStrategyComparisonEntry",
    "SingleSymbolReplayResult",
    "StrategyComparison",
    "StrategyComparisonEntry",
    "SymbolStrategyComparison",
    "UniversalStrategyComparison",
    "compare_paired_returns",
    "compare_shared_cash_strategy_factories",
    "compare_strategies",
    "compare_strategies_by_symbol",
    "compare_strategy_factories_by_symbol",
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
    from inspect import Parameter, signature

    import trade_rl.evaluation.runs as facade
    from trade_rl.evaluation.runs import (
        artifact,
        candidate_suite,
        config,
        execute,
        portfolio_family_attempt,
        portfolio_family_comparison,
        portfolio_family_result_artifact,
        provenance,
        ridge_artifact_comparison,
        ridge_comparison_artifact,
        ridge_comparison_attempt,
    )

    assert set(facade.__all__) == RUNS_EXPORTS
    assert facade.CandidateRunConfig is config.CandidateRunConfig
    assert (
        facade.FourFamilyPortfolioAttemptIdentity
        is portfolio_family_result_artifact.FourFamilyPortfolioAttemptIdentity
    )
    assert (
        facade.LoadedFourFamilyPortfolioResultArtifact
        is portfolio_family_result_artifact.LoadedFourFamilyPortfolioResultArtifact
    )
    assert (
        facade.load_four_family_portfolio_result_artifact
        is portfolio_family_result_artifact.load_four_family_portfolio_result_artifact
    )
    assert (
        facade.FourFamilyPortfolioAttemptContext
        is portfolio_family_attempt.FourFamilyPortfolioAttemptContext
    )
    assert (
        facade.LoadedFourFamilyPortfolioAttempt
        is portfolio_family_attempt.LoadedFourFamilyPortfolioAttempt
    )
    assert (
        facade.PublishedFourFamilyPortfolioAttempt
        is portfolio_family_attempt.PublishedFourFamilyPortfolioAttempt
    )
    assert (
        facade.run_four_family_portfolio_attempt
        is portfolio_family_attempt.run_four_family_portfolio_attempt
    )
    assert (
        facade.load_four_family_portfolio_attempt
        is portfolio_family_attempt.load_four_family_portfolio_attempt
    )
    result_digest = signature(facade.load_four_family_portfolio_attempt).parameters[
        "expected_result_digest"
    ]
    assert result_digest.kind is Parameter.KEYWORD_ONLY
    assert result_digest.default is Parameter.empty
    assert (
        facade.RidgeArtifactComparison
        is ridge_artifact_comparison.RidgeArtifactComparison
    )
    assert (
        facade.RidgeArtifactComparisonConfig
        is ridge_artifact_comparison.RidgeArtifactComparisonConfig
    )
    assert (
        facade.FourFamilyPortfolioReplayResult
        is portfolio_family_comparison.FourFamilyPortfolioReplayResult
    )
    assert (
        facade.PortfolioFamilyReplayOutcome
        is portfolio_family_comparison.PortfolioFamilyReplayOutcome
    )
    assert (
        facade.run_four_family_shared_portfolio_replay
        is portfolio_family_comparison.run_four_family_shared_portfolio_replay
    )
    assert facade.ForecastErrorMetrics is ridge_artifact_comparison.ForecastErrorMetrics
    assert (
        facade.ForecastSymbolDiagnostics
        is ridge_artifact_comparison.ForecastSymbolDiagnostics
    )
    assert (
        facade.RidgeForecastDiagnostics
        is ridge_artifact_comparison.RidgeForecastDiagnostics
    )
    assert (
        facade.compare_ridge_artifact_shared_cash
        is ridge_artifact_comparison.compare_ridge_artifact_shared_cash
    )
    assert (
        facade.LoadedRidgeComparisonArtifact
        is ridge_comparison_artifact.LoadedRidgeComparisonArtifact
    )
    assert (
        facade.load_ridge_comparison_artifact
        is ridge_comparison_artifact.load_ridge_comparison_artifact
    )
    assert (
        facade.publish_ridge_comparison_artifact
        is ridge_comparison_artifact.publish_ridge_comparison_artifact
    )
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
        facade.PublishedRidgeComparisonAttempt
        is ridge_comparison_attempt.PublishedRidgeComparisonAttempt
    )
    assert (
        facade.run_ridge_comparison_attempt
        is ridge_comparison_attempt.run_ridge_comparison_attempt
    )
    assert (
        facade.inspect_candidate_run_artifact is artifact.inspect_candidate_run_artifact
    )
    assert facade.load_candidate_run_artifact is artifact.load_candidate_run_artifact
    assert facade.publish_candidate_run is artifact.publish_candidate_run
    assert (
        facade.supports_shared_cash_result_schema
        is artifact.supports_shared_cash_result_schema
    )
    assert (
        facade.build_candidate_run_provenance
        is provenance.build_candidate_run_provenance
    )
    assert not hasattr(facade, "load_candidate_run_config")
    assert not hasattr(facade, "PROVENANCE_SCHEMA")


def test_run_schema_capability_is_available_through_public_facade() -> None:
    from trade_rl.evaluation.runs import supports_shared_cash_result_schema

    assert supports_shared_cash_result_schema("lean_candidate_result_v7")
    assert not supports_shared_cash_result_schema("lean_candidate_result_v8")
    assert supports_shared_cash_result_schema("lean_candidate_result_v9")
    assert not supports_shared_cash_result_schema(None)


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


def test_runs_facade_keeps_joint_rl_runtime_imports_lazy() -> None:
    script = (
        "import sys; import trade_rl.evaluation.runs; "
        "assert 'torch' not in sys.modules; "
        "assert not any(name == 'stable_baselines3' or "
        "name.startswith('stable_baselines3.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
