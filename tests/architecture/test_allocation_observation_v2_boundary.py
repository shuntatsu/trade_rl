"""The standalone v2 declaration owns no encoder or runtime dependency."""

import subprocess
import sys
from importlib import import_module
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_v2_schema_has_no_direct_account_runtime_or_numeric_imports():
    path = PACKAGE / "strategies/rl/allocation_observation_v2.py"
    assert path.is_file(), (
        "the separately versioned lower v2 observation owner is missing"
    )
    imports = ImportCollector(PACKAGE).collect(path)
    assert not any(
        within_module(name, prefix)
        for name in imports
        for prefix in (
            "trade_rl.evaluation",
            "trade_rl.simulation",
            "trade_rl.risk",
            "gymnasium",
            "stable_baselines3",
            "torch",
            "numpy",
            "trade_rl.strategies.allocation_action",
            "trade_rl.strategies.allocation_snapshot",
        )
    )


def test_v2_stage_exports_only_a_schema_and_no_numeric_encoder():
    module = import_module("trade_rl.strategies.rl.allocation_observation_v2")
    assert module.__all__ == ["AllocationObservationSchema"]
    assert not hasattr(module, "encode_allocation_observation_v2")


def test_v2_import_requires_no_optional_torch():
    script = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'stable_baselines3'):
            raise AssertionError('optional learner imported: ' + fullname)
sys.meta_path.insert(0, BlockTorch())
import trade_rl.strategies.rl.allocation_observation_v2
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
