"""Scheduled diagnostics reuse closed admission without a new learner or facade."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
OWNER = PACKAGE / "evaluation" / "rl_allocation" / "scheduled_credit_diagnostics.py"


def test_credit_diagnostics_reuses_closed_validator_without_runtime_execution():
    imports = ImportCollector(PACKAGE).collect(OWNER)
    assert (
        "trade_rl.evaluation.rl_allocation.scheduled_transition_validation" in imports
    )
    assert not any(
        within_module(module, prefix)
        for module in imports
        for prefix in (
            "trade_rl.simulation",
            "trade_rl.risk",
            "trade_rl.integrations",
            "trade_rl.evaluation.runs",
            "trade_rl.evaluation.experiments",
            "trade_rl.evaluation.rl_allocation.env",
            "trade_rl.evaluation.rl_allocation.scheduled_training",
            "torch",
            "stable_baselines3",
            "gymnasium",
        )
    )
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(ast.parse(OWNER.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert calls.isdisjoint(
        {
            "fit",
            "predict",
            "predict_values",
            "load",
            "execute_interval",
            "apply_fill",
            "get",
        }
    )


def test_credit_diagnostics_stays_module_local_and_imports_without_backend():
    module = importlib.import_module(
        "trade_rl.evaluation.rl_allocation.scheduled_credit_diagnostics"
    )
    assert module.__all__ == [
        "ScheduledAllocationCreditDiagnostics",
        "diagnose_scheduled_allocation_credit",
    ]
    from trade_rl import evaluation, strategies
    from trade_rl.evaluation import rl_allocation

    for facade in (evaluation, strategies, rl_allocation):
        assert not hasattr(facade, "diagnose_scheduled_allocation_credit")
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'stable_baselines3'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from trade_rl.evaluation.rl_allocation.scheduled_credit_diagnostics import diagnose_scheduled_allocation_credit
assert callable(diagnose_scheduled_allocation_credit)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
