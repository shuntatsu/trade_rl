from __future__ import annotations

from pathlib import Path

import trade_rl.evaluation as evaluation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl" / "evaluation"

EXPECTED_PUBLIC_API = (
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
)


def test_evaluation_root_keeps_core_primitives() -> None:
    for name in ("__init__.py", "replay.py", "metrics.py", "evidence.py", "series.py"):
        assert (PACKAGE / name).is_file(), name
    assert (PACKAGE / "gates" / "__init__.py").is_file()


def test_evaluation_responsibility_packages_exist() -> None:
    required = (
        "comparison/__init__.py",
        "comparison/bootstrap.py",
        "comparison/paired.py",
        "comparison/seed_robustness.py",
        "comparison/strategies.py",
        "robustness/__init__.py",
        "robustness/capacity.py",
        "robustness/closed_trades.py",
        "robustness/fold_metrics.py",
        "robustness/perfect_information/__init__.py",
        "robustness/perfect_information/bound.py",
        "robustness/perfect_information/solver.py",
        "robustness/walk_forward/__init__.py",
        "robustness/walk_forward/capabilities.py",
        "robustness/walk_forward/folds.py",
        "robustness/walk_forward/sealed_test.py",
        "robustness/walk_forward/stitching.py",
        "runs/__init__.py",
        "runs/candidate.py",
        "runs/candidate_suite.py",
        "runs/config.py",
        "runs/execute.py",
        "runs/provenance.py",
        "runs/artifact.py",
        "experiments/__init__.py",
        "experiments/errors.py",
        "experiments/store.py",
        "experiments/evidence.py",
        "experiments/delta.py",
        "experiments/analysis.py",
        "experiments/workflow.py",
        "experiments/contracts/__init__.py",
        "experiments/contracts/_common.py",
        "experiments/contracts/run.py",
        "experiments/contracts/study.py",
        "experiments/contracts/experiment.py",
        "experiments/contracts/decision.py",
        "experiments/contracts/comparison.py",
    )
    missing = [path for path in required if not (PACKAGE / path).is_file()]
    assert missing == []


def test_retired_evaluation_private_paths_are_absent() -> None:
    retired = (
        "_perfect_information_lp.py",
        "bootstrap.py",
        "candidate_run.py",
        "candidate_suite.py",
        "capacity.py",
        "closed_trades.py",
        "comparisons.py",
        "fold_metrics.py",
        "perfect_information_bound.py",
        "seed_robustness.py",
        "strategy_comparison.py",
        "walk_forward",
    )
    surviving = [path for path in retired if (PACKAGE / path).exists()]
    assert surviving == []


def test_phase4a_one_shot_helpers_are_absent() -> None:
    retired_helpers = (
        ROOT / "scripts" / "phase4a_evaluation_migrate.py",
        ROOT / ".github" / "workflows" / "phase4a-evaluation-migrate.yml",
        ROOT
        / ".github"
        / "workflows"
        / "phase4a-evaluation-semantic-falsification.yml",
    )
    surviving = [
        str(path.relative_to(ROOT)) for path in retired_helpers if path.exists()
    ]
    assert surviving == []


def test_evaluation_package_public_api_is_exactly_preserved() -> None:
    assert tuple(evaluation.__all__) == EXPECTED_PUBLIC_API
    for name in EXPECTED_PUBLIC_API:
        assert hasattr(evaluation, name), name
