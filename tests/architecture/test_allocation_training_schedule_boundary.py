import ast
import subprocess
import sys
from pathlib import Path

from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
    AllocationTrainingWindow,
)

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "trade_rl" / "strategies" / "rl" / "allocation_training_schedule.py"


def imports() -> set[str]:
    tree = ast.parse(OWNER.read_text(encoding="utf-8"), filename=str(OWNER))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_training_schedule_owner_stays_pure_and_below_runtime():
    actual = imports()
    forbidden = (
        "trade_rl.evaluation",
        "trade_rl.simulation",
        "trade_rl.risk",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "numpy",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in actual
        for prefix in forbidden
    )
    assert AllocationTrainingSchedule.__module__ == (
        "trade_rl.strategies.rl.allocation_training_schedule"
    )
    assert AllocationTrainingWindow.__module__ == (
        "trade_rl.strategies.rl.allocation_training_schedule"
    )


def test_training_schedule_owner_loads_without_optional_rl_backend():
    code = f"""
import builtins
import importlib.util
import sys
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in {{"torch", "stable_baselines3", "gymnasium"}}:
        raise RuntimeError(name)
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location("allocation_training_schedule_probe", {str(OWNER)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
