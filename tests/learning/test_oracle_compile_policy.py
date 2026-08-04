from __future__ import annotations

import pytest

from trade_rl.learning.oracle_bellman_contracts import OracleSolverConfig


def test_unvalidated_compiled_cuda_mode_cannot_enter_solver_contract() -> None:
    with pytest.raises(ValueError, match="compile_mode_unvalidated"):
        OracleSolverConfig(
            selection="cuda",
            compile_mode="reduce_overhead",
        )
