from __future__ import annotations

import numpy as np
import pytest

from trade_rl.integrations.oracle_solver import solve_torch_cuda_oracle_batch
from trade_rl.learning.oracle_bellman_contracts import (
    OracleBackendFailure,
    OracleSolverConfig,
)


def test_unvalidated_compiled_cuda_mode_fails_at_public_adapter_boundary() -> None:
    config = OracleSolverConfig(
        selection="cuda",
        compile_mode="reduce_overhead",
    )

    with pytest.raises(OracleBackendFailure, match="compile_mode_unvalidated"):
        solve_torch_cuda_oracle_batch(
            tape=None,
            states=np.zeros((1, 1), dtype=np.float64),
            episode_inputs=None,
            parameters=None,
            solver_config=config,
        )
