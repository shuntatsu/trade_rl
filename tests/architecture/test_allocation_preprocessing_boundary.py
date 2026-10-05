"""No learner consumer, upper account ownership or optional backend imports."""

import subprocess
import sys
from importlib import import_module
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    "trade_rl.strategies.rl.allocation_preprocessing",
    "trade_rl.evaluation.rl_allocation.preprocessing",
)


def test_preprocessing_owners_avoid_account_and_optional_learner_imports():
    collector = ImportCollector(ROOT / "trade_rl")
    for module in MODULES:
        imports = collector.collect(ROOT / (module.replace(".", "/") + ".py"))
        assert not any(
            within_module(name, prefix)
            for name in imports
            for prefix in "trade_rl.simulation trade_rl.risk torch stable_baselines3".split()
        )
    assert import_module(MODULES[0]).__all__ == ["AllocationFeaturePreprocessing"]
    assert import_module(MODULES[1]).__all__ == ["fit_allocation_feature_preprocessing"]
    assert not any(
        within_module(name, MODULES[0])
        for name in collector.collect(ROOT / "trade_rl/evaluation/rl_allocation/env.py")
    )


def test_preprocessing_imports_need_no_optional_learner():
    script = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','stable_baselines3'): raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import trade_rl.evaluation.rl_allocation.preprocessing
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
