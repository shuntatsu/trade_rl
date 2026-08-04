from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_without_torch(source: str) -> subprocess.CompletedProcess[str]:
    guarded = f"""
import builtins
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        error = ModuleNotFoundError("No module named 'torch'")
        error.name = "torch"
        raise error
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
{textwrap.dedent(source)}
"""
    return subprocess.run(
        [sys.executable, "-c", guarded],
        check=False,
        capture_output=True,
        text=True,
    )


def test_learning_contracts_import_without_torch() -> None:
    result = _run_without_torch(
        """
from trade_rl.learning.oracle_bellman_contracts import OracleSolverConfig
assert OracleSolverConfig(selection="numpy").selection == "numpy"
"""
    )

    assert result.returncode == 0, result.stderr


def test_cuda_backend_reports_typed_failure_without_torch() -> None:
    result = _run_without_torch(
        """
from trade_rl.learning.oracle_bellman_contracts import OracleBackendFailure
from trade_rl.learning.oracle_solver import solve_torch_cuda_oracle_batch

try:
    solve_torch_cuda_oracle_batch(
        tape=None,
        states=None,
        episode_inputs=None,
        parameters=None,
        solver_config=None,
    )
except OracleBackendFailure as error:
    assert error.backend == "torch_cuda"
    assert error.reason == "torch_unavailable"
else:
    raise AssertionError("missing Torch did not raise OracleBackendFailure")
"""
    )

    assert result.returncode == 0, result.stderr
