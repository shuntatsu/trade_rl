"""Fee views remain below native admission and outside evidence authority."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOWER = ROOT / "trade_rl/strategies/rl/allocation_fee_stress.py"
UPPER = ROOT / "trade_rl/evaluation/rl_allocation/fee_stress_admission.py"


def imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    } | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}


def test_fee_views_have_no_learner_or_upper_authority_dependency():
    for path, forbidden in (
        (
            LOWER,
            (
                "trade_rl.evaluation",
                "trade_rl.simulation",
                "trade_rl.risk",
                "torch",
                "stable_baselines3",
            ),
        ),
        (
            UPPER,
            (
                "torch",
                "stable_baselines3",
                "trade_rl.evaluation.experiments",
                "trade_rl.evaluation.final_test",
                "trade_rl.evaluation.rl_allocation.training",
                "trade_rl.evaluation.rl_allocation.scheduled_training",
            ),
        ),
    ):
        assert not any(
            name == p or name.startswith(p + ".")
            for name in imports(path)
            for p in forbidden
        )
    assert "build_continuous_allocation_comparison_evidence" not in UPPER.read_text(
        encoding="utf-8"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
real = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.', 1)[0] in ('torch', 'stable_baselines3'):
        raise RuntimeError(name)
    return real(name, *args, **kwargs)
builtins.__import__ = guard
import trade_rl.strategies.rl.allocation_fee_stress
import trade_rl.evaluation.rl_allocation.fee_stress_admission
""",
        ],
        cwd=ROOT,
        check=True,
    )


def test_fee_global_result_cannot_enter_historical_common_metrics(
    tmp_path, monkeypatch
):
    import trade_rl.strategies.rl.allocation_artifact as archive
    from tests.evaluation.allocation_fee_stress_fixture import native_pair, publish_fake
    from tests.evaluation.test_allocation_fee_stress import declaration, run
    from trade_rl.evaluation.allocation_comparison_evidence import (
        canonical_continuous_allocation_metrics,
    )

    base, stress = native_pair()
    root, digest, model = publish_fake(base, tmp_path, monkeypatch)
    monkeypatch.setattr(archive, "_load_policy", lambda _: model)
    result = run(base, stress, root, digest)
    _, contract, _ = declaration(base, stress, root, digest)
    for value in (result, result.walk_forward):
        with pytest.raises(ValueError, match="continuous allocation result"):
            canonical_continuous_allocation_metrics(contract, (stress,), value)
    assert "test_allocation_fee_stress_runtime.py" in (
        ROOT / ".github/workflows/ci.yml"
    ).read_text(encoding="utf-8")
