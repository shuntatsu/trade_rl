"""The completed global adapter is a pure bounded evidence consumer."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "trade_rl/evaluation/allocation_global_comparison_evidence.py"


def test_global_comparison_owner_has_no_execution_or_research_authority():
    assert OWNER.exists(), "pure completed global evidence owner is missing"
    tree = ast.parse(OWNER.read_text("utf-8"))
    imports = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            calls.add(ast.unparse(node.func))
    forbidden = (
        "torch",
        "stable_baselines3",
        "trade_rl.evaluation.experiments",
        "trade_rl.evaluation.final_test",
        "trade_rl.evaluation.runs",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.evaluation.rl_allocation.env",
        "trade_rl.evaluation.rl_allocation.scheduled_training",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in forbidden
    )
    assert not any(
        name.split(".")[-1]
        in {
            "reset",
            "step",
            "predict",
            "action",
            "learn",
            "load_allocation_policy",
            "run_declared_global_allocation_execution",
            "select_allocation_candidate",
            "build_continuous_allocation_comparison_evidence",
            "build_allocation_cash_control_evidence",
        }
        for name in calls
    )


def test_global_comparison_exports_only_closed_record_and_two_builders():
    assert OWNER.exists(), "pure completed global evidence owner is missing"
    tree = ast.parse(OWNER.read_text("utf-8"))
    declaration = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    assert ast.literal_eval(declaration.value) == [
        "GlobalAllocationValidityRecord",
        "build_global_allocation_comparison_evidence",
        "build_global_allocation_cash_control_evidence",
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(
            "build_global_allocation_"
        ):
            assert "expected_assurance_scope" in [
                arg.arg for arg in node.args.kwonlyargs
            ]
            assert (
                node.args.kw_defaults[
                    [arg.arg for arg in node.args.kwonlyargs].index(
                        "expected_assurance_scope"
                    )
                ]
                is None
            )
