from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "trade_rl" / "evaluation"
ARCHITECTURE_TEST = ROOT / "tests" / "architecture" / "test_lean_evaluation_layout.py"

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

OWNER_MAP = {
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
    "trade_rl.evaluation.walk_forward.capabilities": "trade_rl.evaluation.robustness.walk_forward.capabilities",
    "trade_rl.evaluation.walk_forward.folds": "trade_rl.evaluation.robustness.walk_forward.folds",
    "trade_rl.evaluation.walk_forward.sealed_test": "trade_rl.evaluation.robustness.walk_forward.sealed_test",
    "trade_rl.evaluation.walk_forward.stitching": "trade_rl.evaluation.robustness.walk_forward.stitching",
}

EXPECTED_ROOT_FILES_BEFORE = {
    "__init__.py",
    "_perfect_information_lp.py",
    "bootstrap.py",
    "candidate_run.py",
    "candidate_suite.py",
    "capacity.py",
    "closed_trades.py",
    "comparisons.py",
    "evidence.py",
    "fold_metrics.py",
    "metrics.py",
    "perfect_information_bound.py",
    "replay.py",
    "seed_robustness.py",
    "series.py",
    "strategy_comparison.py",
}

EXPECTED_ROOT_FILES_AFTER = {
    "__init__.py",
    "evidence.py",
    "metrics.py",
    "replay.py",
    "series.py",
}

EXPECTED_DIRECTORIES_AFTER = {"comparison", "gates", "robustness", "runs"}

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

PACKAGE_FILES = {
    "comparison": {
        "__init__.py",
        "bootstrap.py",
        "paired.py",
        "seed_robustness.py",
        "strategies.py",
    },
    "runs": {"__init__.py", "candidate.py", "candidate_suite.py"},
    "robustness/perfect_information": {"__init__.py", "bound.py", "solver.py"},
    "robustness/walk_forward": {
        "__init__.py",
        "capabilities.py",
        "folds.py",
        "sealed_test.py",
        "stitching.py",
    },
}

CURRENT_CLI = "python -m trade_rl.evaluation.candidate_run"
NEW_CLI = "python -m trade_rl.evaluation.runs.candidate"
CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "trade_rl_lean_redesign_20260908.md",
)


def _exported_names(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError(f"missing __all__: {path}")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _assert_preconditions() -> None:
    observed = {path.name for path in EVALUATION.glob("*.py")}
    if observed != EXPECTED_ROOT_FILES_BEFORE:
        raise RuntimeError(
            "evaluation root inventory changed before Phase 4A: "
            f"observed={sorted(observed)} expected={sorted(EXPECTED_ROOT_FILES_BEFORE)}"
        )
    if len(MOVES) != 16 or len(OWNER_MAP) != 16:
        raise RuntimeError("Phase 4A requires exactly 16 MOVE owners")
    if _exported_names(EVALUATION / "__init__.py") != EXPECTED_PUBLIC_API:
        raise RuntimeError("evaluation public API changed before migration")
    for package in ("comparison", "robustness", "runs"):
        if (EVALUATION / package).exists():
            raise RuntimeError(f"new evaluation owner already exists: {package}")
    walk_forward = EVALUATION / "walk_forward"
    if {path.name for path in walk_forward.glob("*.py")} != {
        "__init__.py",
        "capabilities.py",
        "folds.py",
        "sealed_test.py",
        "stitching.py",
    }:
        raise RuntimeError("walk_forward inventory changed before migration")
    for path in CURRENT_DOCS:
        if not path.is_file():
            raise RuntimeError(f"current documentation missing: {path}")
    occurrences = sum(
        path.read_text(encoding="utf-8").count(CURRENT_CLI) for path in CURRENT_DOCS
    )
    if occurrences != 2:
        raise RuntimeError(
            f"expected exactly two current pre-move candidate CLI references, observed {occurrences}"
        )


def _move_files() -> None:
    for old, new in MOVES.items():
        old_path = EVALUATION / old
        new_path = EVALUATION / new
        if not old_path.is_file() or new_path.exists():
            raise RuntimeError(f"invalid move precondition: {old} -> {new}")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)
    old_walk_forward = EVALUATION / "walk_forward"
    try:
        old_walk_forward.rmdir()
    except OSError as exc:
        raise RuntimeError("old walk_forward directory is not empty after migration") from exc


def _rewrite_owner_references() -> None:
    placeholders = {
        old: f"__TRADE_RL_PHASE4A_OWNER_{index:02d}__"
        for index, old in enumerate(OWNER_MAP, start=1)
    }
    for root_name in ("trade_rl", "tests", "scripts"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.resolve() in {Path(__file__).resolve(), ARCHITECTURE_TEST.resolve()}:
                continue
            text = path.read_text(encoding="utf-8")
            updated = text
            for old, placeholder in placeholders.items():
                updated = updated.replace(old, placeholder)
            for old, placeholder in placeholders.items():
                updated = updated.replace(placeholder, OWNER_MAP[old])
            if updated != text:
                path.write_text(updated, encoding="utf-8")


def _write_package_markers() -> None:
    descriptions = {
        "comparison/__init__.py": "Comparison and statistical sampling ownership.",
        "robustness/__init__.py": "Evaluation robustness diagnostics ownership.",
        "robustness/perfect_information/__init__.py": "Perfect-information bound ownership.",
        "runs/__init__.py": "Immutable candidate run orchestration ownership.",
    }
    for relative, description in descriptions.items():
        path = EVALUATION / relative
        if path.exists():
            raise RuntimeError(f"unexpected package marker: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'"""{description}"""\n', encoding="utf-8")


def _rewrite_current_cli_docs() -> None:
    replacements = 0
    for path in CURRENT_DOCS:
        text = path.read_text(encoding="utf-8")
        count = text.count(CURRENT_CLI)
        updated = text.replace(CURRENT_CLI, NEW_CLI)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            replacements += count
    if replacements != 2:
        raise RuntimeError(f"expected exactly two current CLI replacements, observed {replacements}")


def _assert_final_tree() -> None:
    observed_root = {path.name for path in EVALUATION.glob("*.py")}
    if observed_root != EXPECTED_ROOT_FILES_AFTER:
        raise RuntimeError(
            f"unexpected final evaluation root: observed={sorted(observed_root)} "
            f"expected={sorted(EXPECTED_ROOT_FILES_AFTER)}"
        )
    observed_directories = {
        path.name
        for path in EVALUATION.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    if observed_directories != EXPECTED_DIRECTORIES_AFTER:
        raise RuntimeError(
            "unexpected final evaluation directories: "
            f"observed={sorted(observed_directories)} "
            f"expected={sorted(EXPECTED_DIRECTORIES_AFTER)}"
        )
    for package, expected_files in PACKAGE_FILES.items():
        package_root = EVALUATION / package
        observed = {path.name for path in package_root.glob("*.py")}
        if observed != expected_files:
            raise RuntimeError(
                f"unexpected {package} files: observed={sorted(observed)} "
                f"expected={sorted(expected_files)}"
            )
    robustness_root_files = {
        path.name for path in (EVALUATION / "robustness").glob("*.py")
    }
    if robustness_root_files != {
        "__init__.py",
        "capacity.py",
        "closed_trades.py",
        "fold_metrics.py",
    }:
        raise RuntimeError(
            f"unexpected robustness root files: {sorted(robustness_root_files)}"
        )
    if (EVALUATION / "experiments").exists():
        raise RuntimeError("evaluation/experiments is not part of Phase 4A")
    if _exported_names(EVALUATION / "__init__.py") != EXPECTED_PUBLIC_API:
        raise RuntimeError("evaluation public API changed during migration")

    retired = set(OWNER_MAP)
    offenders: list[tuple[str, list[str]]] = []
    for root_name in ("trade_rl", "tests"):
        for path in (ROOT / root_name).rglob("*.py"):
            modules = _imported_modules(path)
            invalid = sorted(modules & retired)
            if invalid:
                offenders.append((str(path.relative_to(ROOT)), invalid))
    if offenders:
        raise RuntimeError(f"retired evaluation imports survive: {offenders}")

    for path in CURRENT_DOCS:
        text = path.read_text(encoding="utf-8")
        if CURRENT_CLI in text or NEW_CLI not in text:
            raise RuntimeError(f"candidate CLI documentation not migrated: {path}")


def main() -> None:
    _assert_preconditions()
    _move_files()
    _rewrite_owner_references()
    _write_package_markers()
    _rewrite_current_cli_docs()
    _assert_final_tree()
    print("Phase 4A evaluation ownership migration created successfully")


if __name__ == "__main__":
    main()
