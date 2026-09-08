from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "trade_rl" / "evaluation"

MOVES = {
    "_perfect_information_lp.py": "robustness/perfect_information/solver.py",
    "bootstrap.py": "comparison/bootstrap.py",
    "candidate_run.py": "runs/candidate.py",
    "candidate_suite.py": "runs/candidate_suite.py",
    "capacity.py": "robustness/capacity.py",
    "closed_trades.py": "robustness/closed_trades.py",
    "comparisons.py": "comparison/paired.py",
    "fold_metrics.py": "robustness/fold_metrics.py",
    "perfect_information_bound.py": "robustness/perfect_information/bound.py",
    "seed_robustness.py": "comparison/seed_robustness.py",
    "strategy_comparison.py": "comparison/strategies.py",
    "walk_forward/__init__.py": "robustness/walk_forward/__init__.py",
    "walk_forward/capabilities.py": "robustness/walk_forward/capabilities.py",
    "walk_forward/folds.py": "robustness/walk_forward/folds.py",
    "walk_forward/sealed_test.py": "robustness/walk_forward/sealed_test.py",
    "walk_forward/stitching.py": "robustness/walk_forward/stitching.py",
}

MODULE_MAP = {
    "trade_rl.evaluation._perfect_information_lp": "trade_rl.evaluation.robustness.perfect_information.solver",
    "trade_rl.evaluation.bootstrap": "trade_rl.evaluation.comparison.bootstrap",
    "trade_rl.evaluation.candidate_run": "trade_rl.evaluation.runs.candidate",
    "trade_rl.evaluation.candidate_suite": "trade_rl.evaluation.runs.candidate_suite",
    "trade_rl.evaluation.capacity": "trade_rl.evaluation.robustness.capacity",
    "trade_rl.evaluation.closed_trades": "trade_rl.evaluation.robustness.closed_trades",
    "trade_rl.evaluation.comparisons": "trade_rl.evaluation.comparison.paired",
    "trade_rl.evaluation.fold_metrics": "trade_rl.evaluation.robustness.fold_metrics",
    "trade_rl.evaluation.perfect_information_bound": "trade_rl.evaluation.robustness.perfect_information.bound",
    "trade_rl.evaluation.seed_robustness": "trade_rl.evaluation.comparison.seed_robustness",
    "trade_rl.evaluation.strategy_comparison": "trade_rl.evaluation.comparison.strategies",
    "trade_rl.evaluation.walk_forward": "trade_rl.evaluation.robustness.walk_forward",
}

PACKAGE_INITS = {
    "comparison/__init__.py": '"""Evaluation comparison and statistical comparison utilities."""\n',
    "robustness/__init__.py": '"""Evaluation robustness and stress-test utilities."""\n',
    "robustness/perfect_information/__init__.py": '"""Perfect-information upper-bound evaluation."""\n',
    "runs/__init__.py": '"""Candidate evaluation run orchestration and CLI entry points."""\n',
}

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


def exported_names(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return tuple(value)
    raise RuntimeError(f"missing __all__: {path}")


def rewrite_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in sorted(MODULE_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        updated = updated.replace(old, new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def main() -> None:
    facade = EVALUATION / "__init__.py"
    if exported_names(facade) != EXPECTED_PUBLIC_API:
        raise RuntimeError("evaluation public API changed before migration")

    for old_rel, new_rel in MOVES.items():
        old = EVALUATION / old_rel
        new = EVALUATION / new_rel
        if not old.is_file():
            raise RuntimeError(f"expected source is missing: {old_rel}")
        if new.exists():
            raise RuntimeError(f"target already exists: {new_rel}")

    for init_rel in PACKAGE_INITS:
        target = EVALUATION / init_rel
        if target.exists():
            raise RuntimeError(f"new package init unexpectedly exists: {init_rel}")

    for old_rel, new_rel in MOVES.items():
        old = EVALUATION / old_rel
        new = EVALUATION / new_rel
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)

    old_walk_forward = EVALUATION / "walk_forward"
    if any(old_walk_forward.iterdir()):
        raise RuntimeError("old walk_forward directory is not empty after planned moves")
    old_walk_forward.rmdir()

    for init_rel, content in PACKAGE_INITS.items():
        target = EVALUATION / init_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    for root in (ROOT / "trade_rl", ROOT / "tests"):
        for path in root.rglob("*.py"):
            rewrite_text(path)

    for doc in (
        ROOT / "README.md",
        ROOT / "docs" / "trade_rl_lean_redesign_20260908.md",
    ):
        if doc.is_file():
            rewrite_text(doc)

    if exported_names(facade) != EXPECTED_PUBLIC_API:
        raise RuntimeError("evaluation public API changed during migration")

    retired_modules = set(MODULE_MAP)
    offenders: list[tuple[str, list[str]]] = []
    for root in (ROOT / "trade_rl", ROOT / "tests"):
        for path in root.rglob("*.py"):
            invalid = sorted(imported_modules(path) & retired_modules)
            if invalid:
                offenders.append((str(path.relative_to(ROOT)), invalid))
    if offenders:
        raise RuntimeError(f"retired evaluation imports remain: {offenders}")

    for old_rel in MOVES:
        if (EVALUATION / old_rel).exists():
            raise RuntimeError(f"retired evaluation path remains: {old_rel}")

    for new_rel in MOVES.values():
        if not (EVALUATION / new_rel).is_file():
            raise RuntimeError(f"moved evaluation target missing: {new_rel}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "python -m trade_rl.evaluation.candidate_run" in readme:
        raise RuntimeError("README still contains retired candidate CLI path")


if __name__ == "__main__":
    main()
