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
FIT_OWNER = ROOT / "trade_rl" / "evaluation" / "rl_allocation" / "scheduled_training.py"
RECEIPT_OWNER = (
    ROOT / "trade_rl" / "strategies" / "rl" / "allocation_schedule_receipt.py"
)


def direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_training_schedule_owner_stays_pure_and_below_runtime():
    actual = direct_imports(OWNER)
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


def test_scheduled_fit_stays_training_only_and_uses_existing_economic_env():
    actual = direct_imports(FIT_OWNER)
    forbidden = (
        "trade_rl.simulation",
        "trade_rl.risk",
        "trade_rl.evaluation.runs",
        "trade_rl.evaluation.experiments",
        "trade_rl.strategies.rl.allocation_artifact",
        "trade_rl.strategies.rl.allocation_manifest",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in actual
        for prefix in forbidden
    )
    assert "trade_rl.evaluation.rl_allocation.training_schedule" in actual
    assert "trade_rl.evaluation.rl_allocation.training_protocol" in actual


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


def test_scheduled_inference_validator_stays_lower_and_runs_without_learner():
    forbidden = (
        "trade_rl.evaluation",
        "trade_rl.simulation",
        "trade_rl.risk",
        "torch",
        "stable_baselines3",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in direct_imports(RECEIPT_OWNER)
        for prefix in forbidden
    )
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'stable_baselines3'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from tests.strategies.test_allocation_schedule_receipt import manifest_v5
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
assert validate_allocation_manifest(manifest_v5())['schema'] == 'allocation_ppo_inference_bundle_v5'
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "test_allocation_schedule_ppo_runtime.py" in (
        ROOT / ".github/workflows/ci.yml"
    ).read_text(encoding="utf-8")
    assert "Scheduled allocation inference software assurance" in (
        ROOT / "docs/architecture/research-assurance.md"
    ).read_text(encoding="utf-8")
