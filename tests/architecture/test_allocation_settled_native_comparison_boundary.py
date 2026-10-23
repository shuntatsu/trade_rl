"""Keep matched control comparison passive and its new collector bounded."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNERS = (
    "trade_rl/evaluation/allocation_settled_native_comparison.py",
    "trade_rl/evaluation/rl_allocation/settled_comparison_source.py",
)


def test_control_comparison_owners_never_execute_or_train_accounts():
    forbidden_modules = (
        "gymnasium",
        "stable_baselines3",
        "torch",
        ".settled_env",
        ".simulation",
        ".integrations",
        ".training",
    )
    forbidden_calls = {
        "reset",
        "step",
        "fit",
        "predict",
        "learn",
        "save",
        "load",
        "run",
        "SettledAllocationTradingEnv",
        "MarketExecutor",
        "Book",
    }
    for name in OWNERS:
        path = ROOT / name
        assert path.is_file(), "source-bound matched-control comparison is missing"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(
                    part in (node.module or "") for part in forbidden_modules
                )
            elif isinstance(node, ast.Import):
                assert not any(
                    part in alias.name
                    for alias in node.names
                    for part in forbidden_modules
                )
            elif isinstance(node, ast.Call):
                called = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else ""
                )
                assert called not in forbidden_calls


def test_control_comparison_revalidates_raw_native_captures():
    path = ROOT / OWNERS[0]
    assert path.is_file(), "source-bound matched-control comparison is missing"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "validate_settled_native_capture" in calls
    assert not {
        "build_settled_forecast_comparison",
        "run_global_nonrl_allocation",
        "run_global_cash_allocation",
    }.intersection(calls)


def test_control_collector_has_only_one_reset_and_step_site():
    path = ROOT / "tests/evaluation/test_allocation_settled_native_comparison.py"
    assert path.is_file(), "the dedicated matched-control collector is missing"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert names.count("reset") == names.count("step") == 1
    imports = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any(
        "test_allocation_settled_native_validation" in name or "learnability" in name
        for name in imports
    )
