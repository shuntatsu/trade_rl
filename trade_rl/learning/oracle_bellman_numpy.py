"""Rolling-buffer batched NumPy Bellman solver for Oracle teachers."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Final

import numpy as np

from trade_rl.learning.oracle_bellman_contracts import (
    OracleBellmanParameters,
    OracleEpisodeInputs,
    OracleSolverConfig,
    OracleSolveResult,
    OracleSolverProvenance,
)
from trade_rl.learning.oracle_market_tape import OracleMarketTape
from trade_rl.learning.oracle_transition_numpy import numpy_transition_step

_EPSILON: Final = 1e-12
_INT16_STATE_LIMIT: Final = 32_767


def _pointer_dtype(state_count: int) -> np.dtype[np.signedinteger]:
    if isinstance(state_count, bool) or not isinstance(state_count, int):
        raise ValueError("state_count must be an integer")
    if state_count <= 0:
        raise ValueError("state_count must be positive")
    return np.dtype(np.int16 if state_count <= _INT16_STATE_LIMIT else np.int32)


def _candidate_scores(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 3:
        raise ValueError("candidate_scores must be three-dimensional")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError("candidate_scores must be numeric")
    scores = np.asarray(raw, dtype=np.float64)
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("candidate_scores contain unsupported non-finite values")
    if scores.shape[0] == 0 or scores.shape[1] == 0 or scores.shape[2] == 0:
        raise ValueError("candidate_scores dimensions must be non-empty")
    return scores


def reduce_candidates_numpy(
    candidate_scores: np.ndarray,
    *,
    tie_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce prior-state candidates with deterministic tolerance-aware ties."""

    scores = _candidate_scores(candidate_scores)
    if not math.isfinite(tie_tolerance) or tie_tolerance <= 0.0:
        raise ValueError("tie_tolerance must be finite and positive")
    best = np.max(scores, axis=1)
    finite = np.isfinite(best)
    eligible = (
        np.isfinite(scores)
        & finite[:, None, :]
        & (scores >= best[:, None, :] - tie_tolerance)
    )
    pointers = np.argmax(eligible, axis=1).astype(
        _pointer_dtype(scores.shape[1]),
        copy=False,
    )
    pointers = np.where(finite, pointers, -1).astype(pointers.dtype, copy=False)
    return best, pointers


def reconstruct_state_paths_numpy(
    pointers: np.ndarray,
    final_states: np.ndarray,
) -> np.ndarray:
    """Reconstruct one target-state path per episode from backpointers."""

    raw_pointers = np.asarray(pointers)
    if raw_pointers.ndim != 3 or not np.issubdtype(
        raw_pointers.dtype, np.signedinteger
    ):
        raise ValueError("pointers must be a three-dimensional signed integer array")
    batch_size, steps, state_count = raw_pointers.shape
    raw_final = np.asarray(final_states)
    if raw_final.shape != (batch_size,) or not np.issubdtype(
        raw_final.dtype, np.integer
    ):
        raise ValueError("final_states must match the episode batch")
    states = np.asarray(raw_final, dtype=np.int64)
    if np.any(states < 0) or np.any(states >= state_count):
        raise ValueError("final_states contain an invalid state index")
    paths = np.empty((batch_size, steps), dtype=np.int64)
    paths[:, -1] = states
    batch_indices = np.arange(batch_size)
    for step in range(steps - 1, 0, -1):
        prior = raw_pointers[batch_indices, step, paths[:, step]].astype(
            np.int64,
            copy=False,
        )
        if np.any(prior < 0) or np.any(prior >= state_count):
            raise RuntimeError("oracle portfolio backpointer is missing")
        paths[:, step - 1] = prior
    return paths


def _validated_states(
    states: object,
    *,
    symbol_count: int,
    parameters: OracleBellmanParameters,
) -> tuple[np.ndarray, int]:
    raw = np.asarray(states)
    if raw.ndim != 2 or raw.shape[1] != symbol_count or raw.shape[0] == 0:
        raise ValueError("states must match the market tape symbols")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError("states must be numeric")
    values = np.asarray(raw, dtype=np.float64).copy(order="C")
    if not np.isfinite(values).all():
        raise ValueError("states must be finite")
    if values.shape[0] > parameters.maximum_states:
        raise ValueError("state count exceeds the maintained bound")
    if np.unique(values, axis=0).shape[0] != values.shape[0]:
        raise ValueError("states must be unique")
    if np.any(np.abs(values) > parameters.max_abs_weight + _EPSILON):
        raise ValueError("states exceed max_abs_weight")
    if np.any(np.abs(values).sum(axis=1) > parameters.max_gross + _EPSILON):
        raise ValueError("states exceed max_gross")
    if not parameters.execution_cost.allow_short and np.any(values < -_EPSILON):
        raise ValueError("states contain a disallowed short position")
    cash = np.flatnonzero(np.all(np.isclose(values, 0.0), axis=1))
    if cash.size != 1:
        raise ValueError("states must contain exactly one cash state")
    return values, int(cash[0])


def _validated_episode_batch(
    *,
    tape: OracleMarketTape,
    episode_inputs: OracleEpisodeInputs,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
) -> tuple[np.ndarray, int]:
    if not isinstance(episode_inputs, OracleEpisodeInputs):
        raise ValueError("episode_inputs must be OracleEpisodeInputs")
    if episode_inputs.initial_weights.shape[1] != tape.symbol_count:
        raise ValueError("episode initial weights do not match the market tape")
    if episode_inputs.episode_count > solver_config.episode_batch_size:
        raise ValueError("episode batch exceeds episode_batch_size")
    horizons = episode_inputs.stops - episode_inputs.starts - 1
    if np.any(horizons != horizons[0]):
        raise ValueError("episode inputs must have an equal horizon")
    if np.any(episode_inputs.starts < tape.start) or np.any(
        episode_inputs.stops > tape.stop
    ):
        raise ValueError("episode bounds are outside the market tape")
    initial = episode_inputs.initial_weights
    if np.any(np.abs(initial) > parameters.max_abs_weight + _EPSILON):
        raise ValueError("episode initial weights exceed max_abs_weight")
    if np.any(np.abs(initial).sum(axis=1) > parameters.max_gross + _EPSILON):
        raise ValueError("episode initial weights exceed max_gross")
    if not parameters.execution_cost.allow_short and np.any(initial < -_EPSILON):
        raise ValueError("episode initial weights contain a disallowed short position")
    step_offsets = episode_inputs.starts - tape.start
    return np.asarray(step_offsets, dtype=np.int64), int(horizons[0])


def _target_blocks(state_count: int, block_size: int | None) -> Iterator[slice]:
    resolved = state_count if block_size is None else min(block_size, state_count)
    for start in range(0, state_count, resolved):
        yield slice(start, min(start + resolved, state_count))


def _selected_candidate_values(
    values: np.ndarray,
    pointers: np.ndarray,
) -> np.ndarray:
    batch_size, _, target_count = values.shape[:3]
    if pointers.shape != (batch_size, target_count):
        raise ValueError("candidate pointers do not align with values")
    safe = np.maximum(np.asarray(pointers, dtype=np.int64), 0)
    batch = np.arange(batch_size)[:, None]
    target = np.arange(target_count)[None, :]
    selected = values[batch, safe, target]
    valid = pointers >= 0
    while valid.ndim < selected.ndim:
        valid = valid[..., None]
    return np.where(valid, selected, 0.0)


def _immediate_step(
    *,
    tape: OracleMarketTape,
    step_indices: np.ndarray,
    previous_scores: np.ndarray,
    previous_weights: np.ndarray,
    states: np.ndarray,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
    step_pointers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = previous_scores.shape[0]
    state_count, symbol_count = states.shape
    next_scores = np.full((batch_size, state_count), -np.inf, dtype=np.float64)
    next_weights = np.zeros(
        (batch_size, state_count, symbol_count),
        dtype=np.float64,
    )
    for block in _target_blocks(state_count, solver_config.target_state_block_size):
        target_states = states[block]
        transition = numpy_transition_step(
            tape=tape,
            step=step_indices,
            prior_scores=previous_scores,
            prior_close_weights=previous_weights,
            targets=target_states,
            parameters=parameters,
        )
        candidate_scores = (
            previous_scores[:, :, None]
            + np.log(
                np.where(
                    transition.valid_prior,
                    transition.gap_factor,
                    1.0,
                )
            )[:, :, None]
            + np.log(np.where(transition.valid, transition.close_factor, 1.0))
        )
        projection = np.abs(
            target_states[None, None, :, :] - transition.effective_targets
        ).sum(axis=3)
        candidate_scores -= parameters.control_tie_break_penalty * projection
        candidate_scores = np.where(transition.valid, candidate_scores, -np.inf)
        best, pointers = reduce_candidates_numpy(
            candidate_scores,
            tie_tolerance=solver_config.tie_tolerance,
        )
        next_scores[:, block] = best
        next_weights[:, block] = _selected_candidate_values(
            transition.close_weights,
            pointers,
        )
        step_pointers[:, block] = pointers
    return next_scores, next_weights


def _delayed_initial_step(
    *,
    tape: OracleMarketTape,
    step_indices: np.ndarray,
    initial_weights: np.ndarray,
    state_count: int,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
    step_pointers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, symbol_count = initial_weights.shape
    previous_scores = np.zeros((batch_size, 1), dtype=np.float64)
    previous_weights = initial_weights[:, None, :]
    transition = numpy_transition_step(
        tape=tape,
        step=step_indices,
        prior_scores=previous_scores,
        prior_close_weights=previous_weights,
        targets=initial_weights[:, None, :],
        parameters=parameters,
    )
    candidate_scores = (
        previous_scores[:, :, None]
        + np.log(
            np.where(
                transition.valid_prior,
                transition.gap_factor,
                1.0,
            )
        )[:, :, None]
        + np.log(np.where(transition.valid, transition.close_factor, 1.0))
    )
    candidate_scores = np.where(transition.valid, candidate_scores, -np.inf)
    best, pointers = reduce_candidates_numpy(
        candidate_scores,
        tie_tolerance=solver_config.tie_tolerance,
    )
    selected = _selected_candidate_values(transition.close_weights, pointers)[:, 0]
    step_pointers[:] = pointers[:, :1]
    scores = np.broadcast_to(best[:, :1], (batch_size, state_count)).copy()
    weights = np.broadcast_to(
        selected[:, None, :],
        (batch_size, state_count, symbol_count),
    ).copy()
    weights[~np.isfinite(scores)] = 0.0
    return scores, weights


def _delayed_later_step(
    *,
    tape: OracleMarketTape,
    step_indices: np.ndarray,
    previous_scores: np.ndarray,
    previous_weights: np.ndarray,
    states: np.ndarray,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
    step_pointers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, state_count = previous_scores.shape
    symbol_count = states.shape[1]
    diagonal_scores = np.full_like(previous_scores, -np.inf)
    diagonal_weights = np.zeros_like(previous_weights)
    batch = np.arange(batch_size)[:, None]
    for block in _target_blocks(state_count, solver_config.target_state_block_size):
        target_states = states[block]
        global_targets = np.arange(block.start, block.stop, dtype=np.int64)
        local_targets = np.arange(global_targets.size, dtype=np.int64)
        transition = numpy_transition_step(
            tape=tape,
            step=step_indices,
            prior_scores=previous_scores,
            prior_close_weights=previous_weights,
            targets=target_states,
            parameters=parameters,
        )
        valid = transition.valid[
            batch,
            global_targets[None, :],
            local_targets[None, :],
        ]
        gap = transition.gap_factor[batch, global_targets[None, :]]
        close_factor = transition.close_factor[
            batch,
            global_targets[None, :],
            local_targets[None, :],
        ]
        scores = (
            previous_scores[:, global_targets]
            + np.log(np.where(valid, gap, 1.0))
            + np.log(np.where(valid, close_factor, 1.0))
        )
        diagonal_scores[:, block] = np.where(valid, scores, -np.inf)
        diagonal_weights[:, block] = transition.close_weights[
            batch,
            global_targets[None, :],
            local_targets[None, :],
        ]
    best, pointers = reduce_candidates_numpy(
        diagonal_scores[:, :, None],
        tie_tolerance=solver_config.tie_tolerance,
    )
    safe = np.maximum(pointers[:, 0].astype(np.int64), 0)
    selected = diagonal_weights[np.arange(batch_size), safe]
    valid = pointers[:, 0] >= 0
    selected = np.where(valid[:, None], selected, 0.0)
    step_pointers[:] = pointers[:, :1]
    next_scores = np.broadcast_to(best[:, :1], previous_scores.shape).copy()
    next_weights = np.broadcast_to(
        selected[:, None, :],
        (batch_size, state_count, symbol_count),
    ).copy()
    next_weights[~np.isfinite(next_scores)] = 0.0
    return next_scores, next_weights


def solve_numpy_oracle_batch(
    *,
    tape: OracleMarketTape,
    states: np.ndarray,
    episode_inputs: OracleEpisodeInputs,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
) -> OracleSolveResult:
    """Solve equal-horizon Oracle episodes with rolling NumPy buffers."""

    if not isinstance(tape, OracleMarketTape):
        raise ValueError("tape must be OracleMarketTape")
    if not isinstance(parameters, OracleBellmanParameters):
        raise ValueError("parameters must be OracleBellmanParameters")
    if not isinstance(solver_config, OracleSolverConfig):
        raise ValueError("solver_config must be OracleSolverConfig")
    if solver_config.selection == "cuda":
        raise ValueError("NumPy solver cannot satisfy a CUDA-only selection")
    state_values, cash_index = _validated_states(
        states,
        symbol_count=tape.symbol_count,
        parameters=parameters,
    )
    step_offsets, steps = _validated_episode_batch(
        tape=tape,
        episode_inputs=episode_inputs,
        parameters=parameters,
        solver_config=solver_config,
    )
    batch_size = episode_inputs.episode_count
    state_count = state_values.shape[0]
    pointer_dtype = _pointer_dtype(state_count)
    pointers = np.full(
        (batch_size, steps, state_count),
        -1,
        dtype=pointer_dtype,
    )

    previous_scores = np.zeros((batch_size, 1), dtype=np.float64)
    previous_weights = episode_inputs.initial_weights[:, None, :].copy()
    for step in range(steps):
        step_indices = step_offsets + step
        if parameters.signal_delay_decisions == 0:
            previous_scores, previous_weights = _immediate_step(
                tape=tape,
                step_indices=step_indices,
                previous_scores=previous_scores,
                previous_weights=previous_weights,
                states=state_values,
                parameters=parameters,
                solver_config=solver_config,
                step_pointers=pointers[:, step],
            )
        elif step == 0:
            previous_scores, previous_weights = _delayed_initial_step(
                tape=tape,
                step_indices=step_indices,
                initial_weights=episode_inputs.initial_weights,
                state_count=state_count,
                parameters=parameters,
                solver_config=solver_config,
                step_pointers=pointers[:, step],
            )
        else:
            previous_scores, previous_weights = _delayed_later_step(
                tape=tape,
                step_indices=step_indices,
                previous_scores=previous_scores,
                previous_weights=previous_weights,
                states=state_values,
                parameters=parameters,
                solver_config=solver_config,
                step_pointers=pointers[:, step],
            )

    if parameters.signal_delay_decisions == 1:
        final_states = np.full(batch_size, cash_index, dtype=np.int64)
        final_scores = previous_scores[:, cash_index]
    else:
        final_scores_matrix, final_pointers = reduce_candidates_numpy(
            previous_scores[:, :, None],
            tie_tolerance=solver_config.tie_tolerance,
        )
        final_states = final_pointers[:, 0].astype(np.int64, copy=False)
        final_scores = final_scores_matrix[:, 0]
    if np.any(final_states < 0) or not np.isfinite(final_scores).all():
        raise RuntimeError("oracle found no executable portfolio path")
    state_paths = reconstruct_state_paths_numpy(pointers, final_states)
    target_paths = tuple(
        np.asarray(state_values[path], dtype=np.float32) for path in state_paths
    )
    return OracleSolveResult(
        targets=target_paths,
        final_scores=final_scores,
        provenance=OracleSolverProvenance.numpy_reference(
            config=solver_config,
            market_tape_digest=tape.digest,
        ),
    )


__all__ = [
    "reconstruct_state_paths_numpy",
    "reduce_candidates_numpy",
    "solve_numpy_oracle_batch",
]
