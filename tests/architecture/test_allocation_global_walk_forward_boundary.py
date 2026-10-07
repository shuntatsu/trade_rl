"""Global consumer stays optional, mechanical and outside common evidence."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "trade_rl/evaluation/rl_allocation/global_walk_forward.py"


def test_global_consumer_reuses_native_owners_without_learning_or_authority():
    tree = ast.parse(OWNER.read_text(encoding="utf-8"))
    imports = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "trade_rl.strategies.rl.allocation_artifact" in imports
    assert "trade_rl.evaluation.rl_allocation.continuation" in imports
    forbidden = (
        "torch",
        "stable_baselines3",
        "trade_rl.evaluation.experiments",
        "trade_rl.evaluation.final_test",
        "trade_rl.evaluation.allocation_comparison_evidence",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.evaluation.rl_allocation.scheduled_training",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in forbidden
    )
    code = """
import builtins
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'stable_baselines3'}:
        raise RuntimeError(name)
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
import trade_rl.evaluation.rl_allocation.global_walk_forward
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_global_result_cannot_enter_historical_common_comparison_adapter():
    from tests.evaluation.test_allocation_comparison_evidence import contract_for
    from tests.evaluation.test_allocation_continuous_walk_forward import fold
    from tests.evaluation.test_allocation_global_walk_forward import declared, v2_env
    from trade_rl.evaluation.allocation_comparison_evidence import (
        canonical_continuous_allocation_metrics,
    )
    from trade_rl.evaluation.rl_allocation.global_walk_forward import (
        run_global_allocation_walk_forward,
    )

    env = v2_env()
    folds = (fold(0, 6, 10),)
    contract = contract_for(env, folds)
    result = run_global_allocation_walk_forward(
        folds, env, (declared(env, lambda *_: 0),)
    )
    with pytest.raises(ValueError, match="continuous allocation result"):
        canonical_continuous_allocation_metrics(contract, (env,), result)
