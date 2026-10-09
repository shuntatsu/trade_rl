"""The global observer adds bounded consistency, never study/model authority."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_global_context_and_facade_owners_keep_upper_provenance_separate():
    lower = ROOT / "trade_rl/evaluation/rl_allocation/global_execution_context.py"
    upper = ROOT / "trade_rl/evaluation/allocation_global_execution.py"
    assert lower.exists() and upper.exists(), "global receipt owners are missing"
    for path in (lower, upper):
        tree = ast.parse(path.read_text("utf-8"))
        imports = {
            n.module
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module
        }
        forbidden = (
            "torch",
            "stable_baselines3",
            "trade_rl.evaluation.experiments",
            "trade_rl.evaluation.final_test",
            "trade_rl.evaluation.rl_allocation.training",
            "trade_rl.evaluation.rl_allocation.scheduled_training",
        )
        assert not any(
            name == p or name.startswith(p + ".") for name in imports for p in forbidden
        )
        if path == lower:
            assert not any(
                name.startswith("trade_rl.evaluation.runs") for name in imports
            )
        else:
            assert {
                name for name in imports if name.startswith("trade_rl.evaluation.runs")
            } == {"trade_rl.evaluation.runs"}
            provenance_imports = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module == "trade_rl.evaluation.runs"
            ]
            assert len(provenance_imports) == 1
            assert [
                (alias.name, alias.asname) for alias in provenance_imports[0].names
            ] == [("build_candidate_run_provenance", None)]


def test_lower_collector_is_concrete_and_not_a_second_callback_protocol():
    lower = ROOT / "trade_rl/evaluation/rl_allocation/global_execution_context.py"
    tree = ast.parse(lower.read_text("utf-8"))
    collector = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "GlobalAllocationExecutionCollector"
    )
    assert not collector.bases and not collector.decorator_list
    assert "Protocol" not in lower.read_text(
        "utf-8"
    ) and "runtime_checkable" not in lower.read_text("utf-8")
    imports = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "trade_rl.evaluation.allocation_global_execution" not in imports
    assert "trade_rl.evaluation.rl_allocation.global_walk_forward" not in imports
    validate = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_global_collector"
    )
    assert "type(collector) is not GlobalAllocationExecutionCollector" in ast.unparse(
        validate
    )


def test_forecast_controls_preflight_stays_in_native_plan_owner_before_execution():
    upper = ROOT / "trade_rl/evaluation/allocation_global_execution.py"
    tree = ast.parse(upper.read_text("utf-8"))
    controls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GlobalAllocationForecastControls"
    )
    assert not controls.bases
    decorator = controls.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert {k.arg: ast.literal_eval(k.value) for k in decorator.keywords} == {
        "frozen": True,
        "slots": True,
    }
    assert {n.target.id for n in controls.body if isinstance(n, ast.AnnAssign)} == {
        "control_plan",
        "treatment_plan",
        "cash_plan",
        "expected_control_forecast_digest",
        "expected_treatment_forecast_digest",
    }
    run = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "run_declared_global_allocation_execution"
    )
    gate = next(
        i
        for i, n in enumerate(run.body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "forecast_controls is not None"
    )
    provenance = next(
        i
        for i, n in enumerate(run.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "_provenance"
    )
    assert gate < provenance
    calls = {ast.unparse(n.func) for n in ast.walk(controls) if isinstance(n, ast.Call)}
    assert not any(
        name in calls
        for name in (
            "_provenance",
            "GlobalAllocationExecutionCollector",
            "run_declared_global_allocation_execution",
        )
    )
