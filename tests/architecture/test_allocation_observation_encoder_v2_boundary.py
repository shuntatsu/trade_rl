"""Pure v2 numeric projection keeps source admission above this owner."""

import subprocess
import sys
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_encoder_has_no_direct_runtime_or_optional_learner_imports():
    path = PACKAGE / "strategies/rl/allocation_observation_encoder_v2.py"
    assert path.is_file(), "the standalone v2 numeric encoder is missing"
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
        )
    )


def test_encoder_import_requires_no_optional_torch():
    script = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'stable_baselines3'):
            raise AssertionError('optional learner imported: ' + fullname)
sys.meta_path.insert(0, BlockTorch())
from trade_rl.strategies.rl.allocation_observation_encoder_v2 import encode_allocation_observation_v2
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
