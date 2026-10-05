"""Actual-model readers and receipts keep optional/runtime imports above them."""

import subprocess
import sys
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]


def test_protocol_receipt_and_model_have_no_reverse_or_optional_static_imports():
    for name in ("allocation_protocol_receipt", "allocation_model"):
        imports = ImportCollector(ROOT / "trade_rl").collect(
            ROOT / "trade_rl/strategies/rl" / f"{name}.py"
        )
        assert not any(
            within_module(module, prefix)
            for module in imports
            for prefix in (
                "trade_rl.evaluation",
                "trade_rl.simulation",
                "trade_rl.risk",
                "torch",
                "stable_baselines3",
            )
        )
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','stable_baselines3'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import trade_rl.strategies.rl.allocation_protocol_receipt
import trade_rl.strategies.rl.allocation_model
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
