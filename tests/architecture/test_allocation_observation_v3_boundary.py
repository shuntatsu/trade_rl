"""Pure v3 owners stay below explicit runtime admission and receipt consumers."""

import subprocess
import sys
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
MODULES = ("allocation_observation_v3", "allocation_recipe_v3")


def test_v3_lower_owners_have_no_direct_runtime_or_optional_learner_dependency():
    collector = ImportCollector(ROOT / "trade_rl")
    for module in MODULES:
        imports = collector.collect(ROOT / f"trade_rl/strategies/rl/{module}.py")
        assert not any(
            within_module(name, prefix)
            for name in imports
            for prefix in "trade_rl.evaluation trade_rl.simulation trade_rl.risk torch stable_baselines3 gymnasium".split()
        )
    env_imports = collector.collect(ROOT / "trade_rl/evaluation/rl_allocation/env.py")
    assert all(f"trade_rl.strategies.rl.{module}" in env_imports for module in MODULES)
    for path in ("training.py", "input_receipt.py"):
        imports = collector.collect(ROOT / f"trade_rl/evaluation/rl_allocation/{path}")
        assert not any(
            within_module(name, f"trade_rl.strategies.rl.{module}")
            for name in imports
            for module in MODULES
        )
    lower = collector.collect(
        ROOT / "trade_rl/strategies/rl/allocation_preprocessing_receipt.py"
    )
    assert not any(within_module(name, "trade_rl.evaluation") for name in lower)


def test_pure_v3_import_does_not_require_optional_learners():
    script = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','stable_baselines3'): raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from trade_rl.strategies.rl.allocation_observation_v3 import allocation_observation_payload_v3,encode_allocation_observation_v3
from trade_rl.strategies.rl.allocation_recipe_v3 import allocation_recipe_payload_v3
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
