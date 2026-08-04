from pathlib import Path

path = Path("trade_rl/learning/oracle_solver.py")
text = path.read_text(encoding="utf-8")
old_imports = '''from trade_rl.learning.oracle_bellman_numpy import solve_numpy_oracle_batch
from trade_rl.learning.oracle_bellman_torch import solve_torch_cuda_oracle_batch
from trade_rl.learning.oracle_market_tape import build_oracle_market_tape
'''
new_imports = '''from trade_rl.learning.oracle_bellman_numpy import solve_numpy_oracle_batch
from trade_rl.learning.oracle_market_tape import (
    OracleMarketTape,
    build_oracle_market_tape,
)
'''
if text.count(old_imports) != 1:
    raise SystemExit("Oracle solver import anchor changed")
text = text.replace(old_imports, new_imports)
anchor = '''def _episode_subset(
'''
wrapper = '''def solve_torch_cuda_oracle_batch(
    *,
    tape: OracleMarketTape,
    states: np.ndarray,
    episode_inputs: OracleEpisodeInputs,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
) -> OracleSolveResult:
    """Load the optional Torch backend only when CUDA execution is requested."""

    try:
        from trade_rl.learning.oracle_bellman_torch import (
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


'''
if text.count(anchor) != 1:
    raise SystemExit("Oracle solver function anchor changed")
path.write_text(text.replace(anchor, wrapper + anchor), encoding="utf-8")
