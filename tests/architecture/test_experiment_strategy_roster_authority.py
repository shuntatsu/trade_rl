from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from trade_rl.evaluation.experiments.contracts import (
    CANDIDATE_STRATEGY_NAMES,
    CONTROL_STRATEGY_NAMES,
    StudyPlan,
)
from trade_rl.evaluation.experiments.delta import FACTOR_RULES

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "trade_rl" / "evaluation" / "experiments"


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
    assert StudyPlan.STRATEGY_NAMES == (
        "cash",
        "constant_long",
        "constant_short",
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
        "ppo",
    )
    assert StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES == (
        "cash",
        "constant_long",
        "constant_short",
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
    )
    dataclass_fields = {field.name for field in fields(StudyPlan)}
    assert "STRATEGY_NAMES" not in dataclass_fields
    assert "PPO_SEED_INVARIANT_STRATEGY_NAMES" not in dataclass_fields


def test_experiment_consumers_do_not_redeclare_study_rosters() -> None:
    forbidden = {
        StudyPlan.STRATEGY_NAMES,
        StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES,
    }
    for name in ("analysis.py", "evidence.py", "delta.py"):
        rosters = _module_level_literal_rosters(EXPERIMENTS / name)
        assert forbidden.isdisjoint(rosters)


def test_factor_rules_reference_only_study_strategies() -> None:
    allowed = frozenset(StudyPlan.STRATEGY_NAMES)
    for rule in FACTOR_RULES.values():
        assert rule.unaffected_strategies <= allowed
