"""Configuration declaration adds no training/runtime consumer."""

import subprocess
import sys
from importlib import import_module
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
MODULE = "trade_rl.strategies.rl.allocation_training_protocol"


def test_protocol_imports_only_stdlib_and_canonical_artifact_helpers():
    path = PACKAGE / "strategies/rl/allocation_training_protocol.py"
    assert path.is_file(), "the pure lower PPO protocol owner is missing"
    imports = ImportCollector(PACKAGE).collect(path)
    assert not any(
        within_module(name, prefix)
        for name in imports
        for prefix in (
            "trade_rl.evaluation",
            "trade_rl.simulation",
            "trade_rl.risk",
            "numpy",
            "gymnasium",
            "torch",
            "stable_baselines3",
        )
    )
    assert all(
        not name.startswith("trade_rl.") or name.startswith("trade_rl.artifacts.")
        for name in imports
    )
    assert import_module(MODULE).__all__ == ["AllocationPPOTrainingProtocol"]


def test_protocol_module_requires_no_optional_learner():
    script = f"""
import importlib.abc, sys
class BlockLearners(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'stable_baselines3'):
            raise AssertionError('optional learner imported: ' + fullname)
sys.meta_path.insert(0, BlockLearners())
import {MODULE}
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
