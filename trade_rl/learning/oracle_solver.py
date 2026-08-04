"""Backend-neutral orchestration for batched Oracle teacher solves."""

from __future__ import annotations

from dataclasses import replace

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
from trade_rl.learning.oracle_market_tape import (
    OracleMarketTape,
    build_oracle_market_tape,
)


def solve_torch_cuda_oracle_batch(
    *,
    tape: OracleMarketTape,
    states: np.ndarray,
    episode_inputs: OracleEpisodeInputs,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
) -> OracleSolveResult:
    """Load the optional Torch integration only when CUDA execution is requested."""

    if (
        isinstance(solver_config, OracleSolverConfig)
        and solver_config.compile_mode != "disabled"
    ):
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
    if solver_config.selection == "cuda_or_numpy":
        try:
            return solve_oracle_episodes(
                dataset,
                states=states,
                episode_inputs=episode_inputs,
                parameters=parameters,
                solver_config=replace(solver_config, selection="cuda"),
            )
        except OracleBackendFailure as error:
            fallback = solve_oracle_episodes(
                dataset,
                states=states,
                episode_inputs=episode_inputs,
                parameters=parameters,
                solver_config=replace(
                    solver_config,
                    selection="numpy",
                    compile_mode="disabled",
                ),
            )
            provenance = replace(
                fallback.provenance,
                fallback_reason=f"{error.backend}:{error.reason}",
                digest="",
            )
            return replace(fallback, provenance=provenance, digest="")
    tape = build_oracle_market_tape(
        dataset,
        (int(episode_inputs.starts.min()), int(episode_inputs.stops.max())),
        parameters,
    )
    targets: list[np.ndarray | None] = [None] * episode_inputs.episode_count
    scores = np.empty(episode_inputs.episode_count, dtype=np.float64)
    provenances = []
    horizons = episode_inputs.stops - episode_inputs.starts - 1
    for horizon in sorted(set(horizons.tolist())):
        horizon_positions = np.flatnonzero(horizons == horizon)
        for offset in range(
            0, horizon_positions.size, solver_config.episode_batch_size
        ):
            positions = horizon_positions[
                offset : offset + solver_config.episode_batch_size
            ]
            backend = (
                solve_numpy_oracle_batch
                if solver_config.selection == "numpy"
                else solve_torch_cuda_oracle_batch
            )
            result = backend(
                tape=tape,
                states=states,
                episode_inputs=_episode_subset(episode_inputs, positions),
                parameters=parameters,
                solver_config=solver_config,
            )
            provenances.append(result.provenance)
            for local_index, position in enumerate(positions):
                targets[int(position)] = result.targets[local_index]
                scores[int(position)] = result.final_scores[local_index]
    if not provenances or any(target is None for target in targets):
        raise RuntimeError("Oracle solve did not produce every episode")
    first = provenances[0]
    stable_fields = (
        "backend",
        "solver_config_digest",
        "market_tape_digest",
        "numeric_dtype",
        "tie_tolerance",
        "episode_batch_size",
        "compile_chunk_size",
        "solver_contract",
        "tie_break_contract",
        "torch_version",
        "cuda_version",
        "device_name",
        "compute_capability",
    )
    for provenance in provenances[1:]:
        if any(
            getattr(provenance, field) != getattr(first, field)
            for field in stable_fields
        ):
            raise RuntimeError("Oracle backend provenance changed within one solve")
    wall_times = [
        value.solver_wall_time_seconds
        for value in provenances
        if value.solver_wall_time_seconds is not None
    ]
    host_peaks = [
        value.peak_host_memory_bytes
        for value in provenances
        if value.peak_host_memory_bytes is not None
    ]
    device_peaks = [
        value.peak_device_memory_bytes
        for value in provenances
        if value.peak_device_memory_bytes is not None
    ]
    effective_blocks = [
        value.target_state_block_size
        for value in provenances
        if value.target_state_block_size is not None
    ]
    compile_modes = {value.compile_mode for value in provenances}
    fallback_reasons = sorted(
        {
            value.fallback_reason
            for value in provenances
            if value.fallback_reason is not None
        }
    )
    aggregate = replace(
        first,
        target_state_block_size=(min(effective_blocks) if effective_blocks else None),
        compile_mode=(
            "reduce_overhead" if compile_modes == {"reduce_overhead"} else "disabled"
        ),
        fallback_reason=(";".join(fallback_reasons) if fallback_reasons else None),
        oom_retry_performed=any(value.oom_retry_performed for value in provenances),
        solver_wall_time_seconds=sum(wall_times) if wall_times else None,
        peak_host_memory_bytes=max(host_peaks) if host_peaks else None,
        peak_device_memory_bytes=max(device_peaks) if device_peaks else None,
        digest="",
    )
    return OracleSolveResult(
        targets=tuple(target for target in targets if target is not None),
        final_scores=scores,
        provenance=aggregate,
    )


__all__ = ["solve_oracle_episodes"]
