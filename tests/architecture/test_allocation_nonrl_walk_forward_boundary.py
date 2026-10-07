import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "trade_rl/evaluation/allocation_nonrl_walk_forward.py"


def test_fixed_nonrl_consumer_reuses_execution_without_learning_or_selection():
    tree = ast.parse(OWNER.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "trade_rl.evaluation.rl_allocation.continuous_walk_forward" in imports
    assert "trade_rl.evaluation.allocation_comparison_evidence" in imports
    forbidden = (
        "torch",
        "stable_baselines3",
        "trade_rl.evaluation.experiments",
        "trade_rl.evaluation.final_test",
        "trade_rl.evaluation.allocation_selection",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.evaluation.rl_allocation.scheduled_training",
        "trade_rl.strategies.forecasts.simple_prequential",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in forbidden
    )


def test_global_nonrl_boundary_accepts_only_native_folds_and_one_environment():
    tree = ast.parse(OWNER.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "run_global_nonrl_allocation" in functions
    run = functions["run_global_nonrl_allocation"]
    assert [a.arg for a in run.args.args] == ["folds", "env"]
    assert not run.args.defaults and run.args.vararg is None and run.args.kwarg is None
    assert [a.arg for a in run.args.kwonlyargs] == ["collector"]
    assert (
        ast.unparse(run.args.kwonlyargs[0].annotation)
        == "GlobalAllocationExecutionCollector | None"
    )
    assert (
        len(run.args.kw_defaults) == 1
        and isinstance(run.args.kw_defaults[0], ast.Constant)
        and run.args.kw_defaults[0].value is None
    )
    native_calls = [
        n
        for n in ast.walk(run)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "run_global_allocation_walk_forward"
    ]
    assert len(native_calls) == 1
    forwarding = {k.arg: k.value for k in native_calls[0].keywords}
    assert ast.unparse(forwarding["collector"]) == "collector"

    calls = {
        n.func.id
        for n in ast.walk(run)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "run_global_allocation_walk_forward" in calls
    assert "run_continuous_allocation_walk_forward" not in calls
