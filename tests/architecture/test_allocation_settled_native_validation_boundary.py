"""A native readback owner must not execute accounts or fabricate learning."""

import ast
from pathlib import Path


def test_source_receipt_accepts_only_read_only_structural_context():
    path = (
        Path(__file__).resolve().parents[2]
        / "trade_rl/evaluation/rl_allocation/settled_training_source.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    protocols = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_SettledSourceContext"
    ]
    assert len(protocols) == 1, "passive source Protocol missing"
    functions = [
        node for node in protocols[0].body if isinstance(node, ast.FunctionDef)
    ]
    assert {node.name for node in functions} == {
        "dataset",
        "stream",
        "bound",
        "_estimates",
        "account_id",
        "start_index",
        "actor_stop_index",
        "settlement_stop_index",
        "symbol_index",
        "feature_indices",
        "expected_horizon_seconds",
        "recipe_digest",
    }
    assert all(
        len(node.decorator_list) == 1
        and ast.unparse(node.decorator_list[0]) == "property"
        for node in functions
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "settled_training_source_receipt"
    )
    assert ast.unparse(function.args.args[0].annotation) == "_SettledSourceContext"


def test_native_readback_has_no_execution_or_learning_owner():
    path = (
        Path(__file__).resolve().parents[2]
        / "trade_rl/evaluation/rl_allocation/settled_native_validation.py"
    )
    assert path.is_file(), "native-only readback owner missing"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {
        "MarketExecutor",
        "SettledAllocationTradingEnv",
        "validate_settled_transition_capture",
        "execute_stateful_orders",
        "execute_target_statefully",
        "execute_quantity_hold_statefully",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            module = node.module or "" if isinstance(node, ast.ImportFrom) else ""
            assert not any(
                n.startswith(("gymnasium", "torch", "stable_baselines3"))
                for n in [module, *names]
            )
            assert not forbidden.intersection(names)
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", "")
            )
            assert name not in forbidden | {
                "step",
                "reset",
                "learn",
                "fit",
                "predict",
                "save",
                "load",
            }


def test_fixed_roster_has_one_learner_free_collector():
    path = (
        Path(__file__).resolve().parents[2]
        / "tests/evaluation/test_allocation_settled_native_validation.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roster = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_ROSTER"
            for target in node.targets
        )
    )
    cases = ast.literal_eval(roster)
    assert [case[0] for case in cases] == [
        "cash_carry",
        "nonrl_residual_baseline",
        "direct_long",
        "direct_short",
        "partial_then_no_market",
        "reserve_drawdown_recovery",
        "signed_debt_end",
        "actor_post_return_capture_failure",
        "reserve_post_return_capture_failure",
    ]
    assert sum(case[1] + case[2] for case in cases) == 25
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"learn", "fit", "predict", "save", "load"}
    collector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_collect_case"
    )
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reset"
            for node in ast.walk(collector)
        )
        == 1
    )
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
            for node in ast.walk(collector)
        )
        == 1
    )
