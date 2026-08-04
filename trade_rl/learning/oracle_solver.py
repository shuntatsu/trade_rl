"""Backend-neutral orchestration for batched Oracle teacher solves."""

from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.learning.oracle_bellman_contracts import (
    OracleBackendFailure,
    OracleBellmanParameters,
    OracleEpisodeInputs,
    OracleSolverConfig,
    OracleSolveResult,
)
from trade_rl.learning.oracle_bellman_numpy import solve_numpy_oracle_batch
from trade_rl.learning.oracle_market_tape import build_oracle_market_tape


def _episode_subset(
    inputs: OracleEpisodeInputs,
    positions: np.ndarray,
) -> OracleEpisodeInputs:
    return OracleEpisodeInputs(
        episode_indices=inputs.episode_indices[positions],
        starts=inputs.starts[positions],
        stops=inputs.stops[positions],
        initial_weights=inputs.initial_weights[positions],
    )


def solve_oracle_episodes(
    dataset: MarketDataset,
    *,
    states: np.ndarray,
    episode_inputs: OracleEpisodeInputs,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
) -> OracleSolveResult:
    """Solve independent episodes while preserving their input ordering."""

    if not isinstance(dataset, MarketDataset):
        raise ValueError("dataset must be MarketDataset")
    if not isinstance(episode_inputs, OracleEpisodeInputs):
        raise ValueError("episode_inputs must be OracleEpisodeInputs")
    if not isinstance(parameters, OracleBellmanParameters):
        raise ValueError("parameters must be OracleBellmanParameters")
    if not isinstance(solver_config, OracleSolverConfig):
        raise ValueError("solver_config must be OracleSolverConfig")
    if solver_config.selection != "numpy":
        raise OracleBackendFailure("torch_cuda", "CUDA backend is not available")

    tape = build_oracle_market_tape(
        dataset,
        (int(episode_inputs.starts.min()), int(episode_inputs.stops.max())),
        parameters,
    )
    targets: list[np.ndarray | None] = [None] * episode_inputs.episode_count
    scores = np.empty(episode_inputs.episode_count, dtype=np.float64)
    provenance = None
    horizons = episode_inputs.stops - episode_inputs.starts - 1
    for horizon in sorted(set(horizons.tolist())):
        horizon_positions = np.flatnonzero(horizons == horizon)
        for offset in range(
            0, horizon_positions.size, solver_config.episode_batch_size
        ):
            positions = horizon_positions[
                offset : offset + solver_config.episode_batch_size
            ]
            result = solve_numpy_oracle_batch(
                tape=tape,
                states=states,
                episode_inputs=_episode_subset(episode_inputs, positions),
                parameters=parameters,
                solver_config=solver_config,
            )
            if provenance is None:
                provenance = result.provenance
            elif provenance.digest != result.provenance.digest:
                raise RuntimeError("Oracle backend provenance changed within one solve")
            for local_index, position in enumerate(positions):
                targets[int(position)] = result.targets[local_index]
                scores[int(position)] = result.final_scores[local_index]
    if provenance is None or any(target is None for target in targets):
        raise RuntimeError("Oracle solve did not produce every episode")
    return OracleSolveResult(
        targets=tuple(target for target in targets if target is not None),
        final_scores=scores,
        provenance=provenance,
    )


__all__ = ["solve_oracle_episodes"]
