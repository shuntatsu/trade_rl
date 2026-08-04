"""Optional Oracle accelerator backends owned by the integration layer."""

from __future__ import annotations

import numpy as np

from trade_rl.learning.oracle_bellman_contracts import (
    OracleBackendFailure,
    OracleBellmanParameters,
    OracleEpisodeInputs,
    OracleSolverConfig,
    OracleSolveResult,
)
from trade_rl.learning.oracle_market_tape import OracleMarketTape


def solve_torch_cuda_oracle_batch(
    *,
    tape: OracleMarketTape,
    states: np.ndarray,
    episode_inputs: OracleEpisodeInputs,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
) -> OracleSolveResult:
    """Load the optional Torch implementation only for an explicit CUDA solve."""

    if solver_config.compile_mode != "disabled":
        raise OracleBackendFailure("torch_cuda", "compile_mode_unvalidated")
    try:
        from trade_rl.integrations.oracle_bellman_torch import (
            solve_torch_cuda_oracle_batch as implementation,
        )
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing == "torch" or missing.startswith("torch."):
            raise OracleBackendFailure("torch_cuda", "torch_unavailable") from error
        raise
    return implementation(
        tape=tape,
        states=states,
        episode_inputs=episode_inputs,
        parameters=parameters,
        solver_config=solver_config,
    )


__all__ = ["solve_torch_cuda_oracle_batch"]
