import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = (
    ROOT / "trade_rl" / "evaluation" / "rl_allocation" / "continuous_walk_forward.py"
)


def imports() -> set[str]:
    tree = ast.parse(OWNER.read_text(encoding="utf-8"), filename=str(OWNER))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_continuous_runner_does_not_train_or_deserialize_policies():
    actual = imports()
    forbidden = (
        "stable_baselines3",
        "torch",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.evaluation.rl_allocation.scheduled_training",
        "trade_rl.strategies.rl.allocation_artifact",
        "trade_rl.strategies.rl.allocation_model",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in actual
        for prefix in forbidden
    )


def test_continuous_runner_import_needs_no_optional_learner_backend():
    code = f"""
import builtins
import importlib.util
import sys
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in {{"torch", "stable_baselines3"}}:
        raise RuntimeError(name)
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location("continuous_walk_forward_probe", {str(OWNER)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""
    subprocess.run([sys.executable, "-c", code], check=True)