import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "trade_rl" / "evaluation" / "allocation_cash_control.py"


def imports():
    tree = ast.parse(OWNER.read_text(encoding="utf-8"), filename=str(OWNER))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_cash_control_does_not_train_select_or_open_final_data():
    actual = imports()
    forbidden = (
        "stable_baselines3",
        "torch",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.evaluation.rl_allocation.scheduled_training",
        "trade_rl.evaluation.experiments",
        "trade_rl.evaluation.gates",
        "trade_rl.evaluation.final_test",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in actual
        for prefix in forbidden
    )


def test_global_cash_boundary_accepts_only_native_folds_and_one_environment():
    tree = ast.parse(OWNER.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "run_global_allocation_cash_control" in functions
    run = functions["run_global_allocation_cash_control"]
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
    assert ast.unparse(forwarding["reset_seed"]) == "0"
    assert "run_global_allocation_walk_forward" in calls
    assert "run_continuous_allocation_walk_forward" not in calls
