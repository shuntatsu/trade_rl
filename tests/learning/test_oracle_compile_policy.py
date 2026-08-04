from __future__ import annotations

import pytest

from trade_rl.learning.oracle_bellman_contracts import (
    OracleBackendFailure,
    OracleSolverConfig,
)
from trade_rl.learning.oracle_solver import solve_torch_cuda_oracle_batch


def test_unvalidated_compiled_cuda_mode_fails_closed_before_backend_import() -> None:
    with pytest.raises(OracleBackendFailure, match="compile_mode_unvalidated"):
        solve_torch_cuda_oracle_batch(
            tape=None,
            states=None,
            episode_inputs=None,
            parameters=None,
            solver_config=OracleSolverConfig(
                selection="cuda",
                compile_mode="reduce_overhead",
            ),
        )
