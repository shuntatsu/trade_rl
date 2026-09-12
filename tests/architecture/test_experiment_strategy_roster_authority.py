from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from tests.architecture.imports import ImportCollector, within_module
from trade_rl.evaluation.experiments.contracts import (
    CANDIDATE_STRATEGY_NAMES,
    CONTROL_STRATEGY_NAMES,
    StudyPlan,
)
from trade_rl.evaluation.experiments.delta import FACTOR_RULES

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
EXPERIMENTS = PACKAGE / "evaluation" / "experiments"
RUNS = PACKAGE / "evaluation" / "runs"
EXPECTED_STUDY_STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "ppo",
)
EXPECTED_PPO_SEED_INVARIANT_STRATEGIES = EXPECTED_STUDY_STRATEGIES[:-1]


def _module_level_literal_rosters(path: Path) -> tuple[tuple[str, ...], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rosters: list[tuple[str, ...]] = []
    for statement in tree.body:
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
        if not isinstance(value, (ast.Tuple, ast.List)):
            continue
        if not value.elts or any(
            not isinstance(element, ast.Constant)
            or not isinstance(element.value, str)
            for element in value.elts
        ):
            continue
        rosters.append(tuple(element.value for element in value.elts))
    return tuple(rosters)


def test_study_plan_owns_strategy_rosters_without_serializing_them() -> None:
    assert StudyPlan.STRATEGY_NAMES == (
        *CONTROL_STRATEGY_NAMES,
        *CANDIDATE_STRATEGY_NAMES,
    )
    assert StudyPlan.STRATEGY_NAMES == EXPECTED_STUDY_STRATEGIES
    assert (
        StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES
        == EXPECTED_PPO_SEED_INVARIANT_STRATEGIES
    )
    dataclass_fields = {field.name for field in fields(StudyPlan)}
    assert "STRATEGY_NAMES" not in dataclass_fields
    assert "PPO_SEED_INVARIANT_STRATEGY_NAMES" not in dataclass_fields


def test_experiment_consumers_do_not_redeclare_study_rosters() -> None:
    forbidden = {
        EXPECTED_STUDY_STRATEGIES,
        EXPECTED_PPO_SEED_INVARIANT_STRATEGIES,
    }
    for name in ("analysis.py", "evidence.py", "delta.py"):
        rosters = _module_level_literal_rosters(EXPERIMENTS / name)
        assert forbidden.isdisjoint(rosters)


def test_factor_rules_reference_only_study_strategies() -> None:
    allowed = frozenset(EXPECTED_STUDY_STRATEGIES)
    for rule in FACTOR_RULES.values():
        assert rule.unaffected_strategies <= allowed


def test_run_core_does_not_depend_on_controlled_experiments() -> None:
    collector = ImportCollector(PACKAGE)
    forbidden = "trade_rl.evaluation.experiments"
    offenders = {
        path.relative_to(ROOT).as_posix(): sorted(
            name for name in collector.collect_direct(path) if within_module(name, forbidden)
        )
        for path in sorted(RUNS.rglob("*.py"))
    }
    assert {path: imports for path, imports in offenders.items() if imports} == {}
