"""Float64 batched Torch Bellman solver for Oracle teachers."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, TypeVar

import numpy as np
import torch

from trade_rl.learning.oracle_bellman_contracts import (
    CompileMode,
    OracleBackendFailure,
    OracleBellmanParameters,
    OracleEpisodeInputs,
    OracleSolverConfig,
    OracleSolveResult,
    OracleSolverProvenance,
)
from trade_rl.learning.oracle_market_tape import OracleMarketTape
from trade_rl.learning.oracle_transition_torch import torch_transition_step

_EPSILON: Final = 1e-12
_INT16_STATE_LIMIT: Final = 32_767
_COMPILE_CHUNK_SIZES: Final = frozenset({8, 16, 32, 64})
_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class TorchMarketTape:
    """One Oracle market tape copied once onto a Torch device."""

    raw_position_factor: torch.Tensor
    equity_position_factor: torch.Tensor
    mark_open_ratio: torch.Tensor
    active: torch.Tensor
    tradable: torch.Tensor
    buy_allowed: torch.Tensor
    sell_allowed: torch.Tensor
    borrow_available: torch.Tensor
    market_notional: torch.Tensor
    participation_capacity: torch.Tensor
    minimum_notional: torch.Tensor
    base_unit_cost: torch.Tensor
    funding_due_rate: torch.Tensor
    borrow_rate: torch.Tensor
    dividend_open_ratio: torch.Tensor
    cash_rate: torch.Tensor
    elapsed_year_fraction: torch.Tensor
    start: int
    stop: int
    digest: str

    @property
    def steps(self) -> int:
        return self.stop - self.start - 1

    @property
    def symbol_count(self) -> int:
        return int(self.raw_position_factor.shape[1])

    @property
    def device(self) -> torch.device:
        return self.raw_position_factor.device


@dataclass(frozen=True, slots=True)
class TorchBellmanResult:
    """Device-resident Bellman result before final host publication."""

    target_paths: torch.Tensor
    final_scores: torch.Tensor


def transfer_market_tape_to_torch(
    tape: OracleMarketTape,
    *,
    device: torch.device,
) -> TorchMarketTape:
    """Copy an immutable NumPy tape once to the requested device."""

    if not isinstance(tape, OracleMarketTape):
        raise ValueError("tape must be OracleMarketTape")
    if not isinstance(device, torch.device):
        raise ValueError("device must be torch.device")

    def numeric(name: str) -> torch.Tensor:
        return torch.tensor(
            getattr(tape, name),
            dtype=torch.float64,
            device=device,
        )

    def boolean(name: str) -> torch.Tensor:
        return torch.tensor(
            getattr(tape, name),
            dtype=torch.bool,
            device=device,
        )

    return TorchMarketTape(
        raw_position_factor=numeric("raw_position_factor"),
        equity_position_factor=numeric("equity_position_factor"),
        mark_open_ratio=numeric("mark_open_ratio"),
        active=boolean("active"),
        tradable=boolean("tradable"),
        buy_allowed=boolean("buy_allowed"),
        sell_allowed=boolean("sell_allowed"),
        borrow_available=boolean("borrow_available"),
        market_notional=numeric("market_notional"),
        participation_capacity=numeric("participation_capacity"),
        minimum_notional=numeric("minimum_notional"),
        base_unit_cost=numeric("base_unit_cost"),
        funding_due_rate=numeric("funding_due_rate"),
        borrow_rate=numeric("borrow_rate"),
        dividend_open_ratio=numeric("dividend_open_ratio"),
        cash_rate=numeric("cash_rate"),
        elapsed_year_fraction=numeric("elapsed_year_fraction"),
        start=tape.start,
        stop=tape.stop,
        digest=tape.digest,
    )


def _pointer_dtype(state_count: int) -> torch.dtype:
    if isinstance(state_count, bool) or not isinstance(state_count, int):
        raise ValueError("state_count must be an integer")
    if state_count <= 0:
        raise ValueError("state_count must be positive")
    return torch.int16 if state_count <= _INT16_STATE_LIMIT else torch.int32


def _reduce_candidates_torch_unchecked(
    candidate_scores: torch.Tensor,
    *,
    tie_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    best = candidate_scores.amax(dim=1)
    finite = torch.isfinite(best)
    eligible = (
        torch.isfinite(candidate_scores)
        & finite.unsqueeze(1)
        & (candidate_scores >= best.unsqueeze(1) - tie_tolerance)
    )
    pointers = (
        eligible.to(torch.int8)
        .argmax(dim=1)
        .to(_pointer_dtype(candidate_scores.shape[1]))
    )
    pointers = torch.where(finite, pointers, torch.full_like(pointers, -1))
    return best, pointers


def reduce_candidates_torch(
    candidate_scores: torch.Tensor,
    *,
    tie_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce prior candidates with the NumPy backend's deterministic ties."""

    if candidate_scores.ndim != 3 or candidate_scores.dtype != torch.float64:
        raise ValueError("candidate_scores must be float64 [B,P,K]")
    if candidate_scores.numel() == 0:
        raise ValueError("candidate_scores dimensions must be non-empty")
    if bool(torch.isnan(candidate_scores).any()) or bool(
        torch.isposinf(candidate_scores).any()
    ):
        raise ValueError("candidate_scores contain unsupported non-finite values")
    if not math.isfinite(tie_tolerance) or tie_tolerance <= 0.0:
        raise ValueError("tie_tolerance must be finite and positive")
    return _reduce_candidates_torch_unchecked(
        candidate_scores, tie_tolerance=tie_tolerance
    )


def _validated_states(
    states: torch.Tensor,
    *,
    symbol_count: int,
    parameters: OracleBellmanParameters,
) -> tuple[torch.Tensor, int]:
    if states.ndim != 2 or states.dtype != torch.float64:
        raise ValueError("states must be float64 [S,N]")
    if states.shape[0] == 0 or states.shape[1] != symbol_count:
        raise ValueError("states must match the market tape symbols")
    if not torch.isfinite(states).all():
        raise ValueError("states must be finite")
    if states.shape[0] > parameters.maximum_states:
        raise ValueError("state count exceeds the maintained bound")
    values = states.detach().cpu().numpy()
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
    return states, int(cash[0])


def _validated_episode_batch(
    *,
    tape: TorchMarketTape,
    episode_inputs: OracleEpisodeInputs,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
) -> tuple[torch.Tensor, int]:
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
    offsets = torch.tensor(
        episode_inputs.starts - tape.start,
        dtype=torch.int64,
        device=tape.device,
    )
    return offsets, int(horizons[0])


def _target_blocks(state_count: int, block_size: int | None) -> tuple[slice, ...]:
    resolved = state_count if block_size is None else min(block_size, state_count)
    return tuple(
        slice(start, min(start + resolved, state_count))
        for start in range(0, state_count, resolved)
    )


def _selected_candidate_values(
    values: torch.Tensor,
    pointers: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, target_count = values.shape[:3]
    if pointers.shape != (batch_size, target_count):
        raise ValueError("candidate pointers do not align with values")
    safe = pointers.to(torch.int64).clamp_min(0)
    batch = torch.arange(batch_size, device=values.device).unsqueeze(1)
    target = torch.arange(target_count, device=values.device).unsqueeze(0)
    selected = values[batch, safe, target]
    valid = pointers >= 0
    while valid.ndim < selected.ndim:
        valid = valid.unsqueeze(-1)
    return torch.where(valid, selected, torch.zeros_like(selected))


def _immediate_step(
    *,
    tape: TorchMarketTape,
    step_indices: torch.Tensor,
    previous_scores: torch.Tensor,
    previous_weights: torch.Tensor,
    states: torch.Tensor,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
    step_pointers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = previous_scores.shape[0]
    state_count, symbol_count = states.shape
    next_scores = torch.full(
        (batch_size, state_count),
        -torch.inf,
        dtype=torch.float64,
        device=states.device,
    )
    next_weights = torch.zeros(
        (batch_size, state_count, symbol_count),
        dtype=torch.float64,
        device=states.device,
    )
    for block in _target_blocks(state_count, solver_config.target_state_block_size):
        target_states = states[block]
        transition = torch_transition_step(
            tape=tape,
            step=step_indices,
            prior_scores=previous_scores,
            prior_close_weights=previous_weights,
            targets=target_states,
            parameters=parameters,
        )
        candidate_scores = (
            previous_scores.unsqueeze(2)
            + torch.log(
                torch.where(
                    transition.valid_prior,
                    transition.gap_factor,
                    torch.ones_like(transition.gap_factor),
                )
            ).unsqueeze(2)
            + torch.log(
                torch.where(
                    transition.valid,
                    transition.close_factor,
                    torch.ones_like(transition.close_factor),
                )
            )
        )
        projection = (
            (target_states.unsqueeze(0).unsqueeze(0) - transition.effective_targets)
            .abs()
            .sum(dim=3)
        )
        candidate_scores = candidate_scores - (
            parameters.control_tie_break_penalty * projection
        )
        candidate_scores = torch.where(
            transition.valid,
            candidate_scores,
            torch.full_like(candidate_scores, -torch.inf),
        )
        best, pointers = _reduce_candidates_torch_unchecked(
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
    tape: TorchMarketTape,
    step_indices: torch.Tensor,
    initial_weights: torch.Tensor,
    state_count: int,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
    step_pointers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, symbol_count = initial_weights.shape
    previous_scores = torch.zeros(
        (batch_size, 1), dtype=torch.float64, device=initial_weights.device
    )
    previous_weights = initial_weights.unsqueeze(1)
    transition = torch_transition_step(
        tape=tape,
        step=step_indices,
        prior_scores=previous_scores,
        prior_close_weights=previous_weights,
        targets=initial_weights.unsqueeze(1),
        parameters=parameters,
    )
    candidate_scores = (
        previous_scores.unsqueeze(2)
        + torch.log(
            torch.where(
                transition.valid_prior,
                transition.gap_factor,
                torch.ones_like(transition.gap_factor),
            )
        ).unsqueeze(2)
        + torch.log(
            torch.where(
                transition.valid,
                transition.close_factor,
                torch.ones_like(transition.close_factor),
            )
        )
    )
    candidate_scores = torch.where(
        transition.valid,
        candidate_scores,
        torch.full_like(candidate_scores, -torch.inf),
    )
    best, pointers = _reduce_candidates_torch_unchecked(
        candidate_scores,
        tie_tolerance=solver_config.tie_tolerance,
    )
    selected = _selected_candidate_values(transition.close_weights, pointers)[:, 0]
    step_pointers[:] = pointers[:, :1]
    scores = best[:, :1].expand(batch_size, state_count).clone()
    weights = (
        selected.unsqueeze(1).expand(batch_size, state_count, symbol_count).clone()
    )
    weights = torch.where(
        torch.isfinite(scores).unsqueeze(2), weights, torch.zeros_like(weights)
    )
    return scores, weights


def _delayed_later_step(
    *,
    tape: TorchMarketTape,
    step_indices: torch.Tensor,
    previous_scores: torch.Tensor,
    previous_weights: torch.Tensor,
    states: torch.Tensor,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
    step_pointers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, state_count = previous_scores.shape
    diagonal_scores = torch.full_like(previous_scores, -torch.inf)
    diagonal_weights = torch.zeros_like(previous_weights)
    batch = torch.arange(batch_size, device=states.device).unsqueeze(1)
    for block in _target_blocks(state_count, solver_config.target_state_block_size):
        target_states = states[block]
        global_targets = torch.arange(
            block.start, block.stop, dtype=torch.int64, device=states.device
        )
        local_targets = torch.arange(
            global_targets.numel(), dtype=torch.int64, device=states.device
        )
        transition = torch_transition_step(
            tape=tape,
            step=step_indices,
            prior_scores=previous_scores,
            prior_close_weights=previous_weights,
            targets=target_states,
            parameters=parameters,
        )
        valid = transition.valid[
            batch,
            global_targets.unsqueeze(0),
            local_targets.unsqueeze(0),
        ]
        gap = transition.gap_factor[batch, global_targets.unsqueeze(0)]
        close_factor = transition.close_factor[
            batch,
            global_targets.unsqueeze(0),
            local_targets.unsqueeze(0),
        ]
        scores = (
            previous_scores[:, global_targets]
            + torch.log(torch.where(valid, gap, torch.ones_like(gap)))
            + torch.log(torch.where(valid, close_factor, torch.ones_like(close_factor)))
        )
        diagonal_scores[:, block] = torch.where(
            valid, scores, torch.full_like(scores, -torch.inf)
        )
        diagonal_weights[:, block] = transition.close_weights[
            batch,
            global_targets.unsqueeze(0),
            local_targets.unsqueeze(0),
        ]
    best, pointers = _reduce_candidates_torch_unchecked(
        diagonal_scores.unsqueeze(2),
        tie_tolerance=solver_config.tie_tolerance,
    )
    safe = pointers[:, 0].to(torch.int64).clamp_min(0)
    selected = diagonal_weights[torch.arange(batch_size, device=states.device), safe]
    valid = pointers[:, 0] >= 0
    selected = torch.where(valid.unsqueeze(1), selected, torch.zeros_like(selected))
    step_pointers[:] = pointers[:, :1]
    next_scores = best[:, :1].expand_as(previous_scores).clone()
    next_weights = selected.unsqueeze(1).expand_as(previous_weights).clone()
    next_weights = torch.where(
        torch.isfinite(next_scores).unsqueeze(2),
        next_weights,
        torch.zeros_like(next_weights),
    )
    return next_scores, next_weights


def _reconstruct_state_paths(
    pointers: torch.Tensor,
    final_states: torch.Tensor,
) -> torch.Tensor:
    batch_size, steps, state_count = pointers.shape
    if final_states.shape != (batch_size,):
        raise ValueError("final_states must match the episode batch")
    paths = torch.empty((batch_size, steps), dtype=torch.int64, device=pointers.device)
    paths[:, -1] = final_states
    batch = torch.arange(batch_size, device=pointers.device)
    invalid = torch.zeros((), dtype=torch.bool, device=pointers.device)
    for step in range(steps - 1, 0, -1):
        prior = pointers[batch, step, paths[:, step]].to(torch.int64)
        invalid = invalid | ((prior < 0) | (prior >= state_count)).any()
        paths[:, step - 1] = prior.clamp(0, state_count - 1)
    if bool(invalid):
        raise RuntimeError("oracle portfolio backpointer is missing")
    return paths


def _solve_torch_oracle_batch_core(
    *,
    tape: TorchMarketTape,
    states: torch.Tensor,
    episode_inputs: OracleEpisodeInputs,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
) -> TorchBellmanResult:
    """Solve one equal-horizon batch while keeping forward state on-device."""

    if not isinstance(tape, TorchMarketTape):
        raise ValueError("tape must be TorchMarketTape")
    if not isinstance(parameters, OracleBellmanParameters):
        raise ValueError("parameters must be OracleBellmanParameters")
    if not isinstance(solver_config, OracleSolverConfig):
        raise ValueError("solver_config must be OracleSolverConfig")
    if states.device != tape.device:
        raise ValueError("states and tape must share a device")
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
    pointers = torch.full(
        (batch_size, steps, state_count),
        -1,
        dtype=_pointer_dtype(state_count),
        device=tape.device,
    )
    previous_scores = torch.zeros(
        (batch_size, 1), dtype=torch.float64, device=tape.device
    )
    previous_weights = torch.tensor(
        episode_inputs.initial_weights,
        dtype=torch.float64,
        device=tape.device,
    ).unsqueeze(1)

    chunk_size = (
        _validated_compile_chunk_size(solver_config)
        if solver_config.compile_mode == "reduce_overhead"
        else steps
    )
    for chunk_start in range(0, steps, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, steps)
        for step in range(chunk_start, chunk_stop):
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
                    initial_weights=previous_weights[:, 0],
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
        final_states = torch.full(
            (batch_size,), cash_index, dtype=torch.int64, device=tape.device
        )
        final_scores = previous_scores[:, cash_index]
    else:
        final_score_matrix, final_pointers = _reduce_candidates_torch_unchecked(
            previous_scores.unsqueeze(2),
            tie_tolerance=solver_config.tie_tolerance,
        )
        final_states = final_pointers[:, 0].to(torch.int64)
        final_scores = final_score_matrix[:, 0]
    if bool((final_states < 0).any()) or not bool(torch.isfinite(final_scores).all()):
        raise RuntimeError("oracle found no executable portfolio path")
    state_paths = _reconstruct_state_paths(pointers, final_states)
    return TorchBellmanResult(
        target_paths=state_values[state_paths].to(torch.float32),
        final_scores=final_scores,
    )


def _validated_compile_chunk_size(solver_config: OracleSolverConfig) -> int:
    if solver_config.compile_mode != "reduce_overhead":
        return solver_config.compile_chunk_size
    if solver_config.compile_chunk_size not in _COMPILE_CHUNK_SIZES:
        raise ValueError("compile_chunk_size must be one of 8, 16, 32, or 64")
    return solver_config.compile_chunk_size


def _is_compile_failure(error: Exception) -> bool:
    module = type(error).__module__
    return (
        module.startswith("torch._dynamo")
        or module.startswith("torch._inductor")
        or "compile" in str(error).lower()
    )


def _prepare_compiled_core(
    solver_config: OracleSolverConfig,
) -> tuple[Callable[..., TorchBellmanResult] | None, str | None]:
    if solver_config.compile_mode != "reduce_overhead":
        return None, None
    try:
        return (
            torch.compile(
                _solve_torch_oracle_batch_core,
                mode="reduce-overhead",
                fullgraph=False,
            ),
            None,
        )
    except Exception as error:
        if not _is_compile_failure(error):
            raise
        return None, f"compile_setup_failed:{type(error).__name__}"


def _resolve_compile_fallback_reason(
    *,
    setup_reason: str | None,
    execution_reason: str | None,
) -> str | None:
    """Preserve the most specific truthful reason for eager execution."""

    return execution_reason or setup_reason


def _run_compiled_or_eager(
    *,
    compiled: Callable[[], _ResultT],
    eager: Callable[[], _ResultT],
) -> tuple[_ResultT, CompileMode, str | None]:
    try:
        return compiled(), "reduce_overhead", None
    except Exception as error:
        if not _is_compile_failure(error):
            raise
        return eager(), "disabled", f"compile_failed:{type(error).__name__}"


def _run_with_oom_retry(
    *,
    solve: Callable[[int], _ResultT],
    initial_block_size: int,
    cleanup: Callable[[], None],
) -> tuple[_ResultT, int, bool]:
    if initial_block_size <= 0:
        raise ValueError("initial_block_size must be positive")
    block_size = initial_block_size
    for attempt in range(2):
        try:
            return solve(block_size), block_size, attempt == 1
        except torch.OutOfMemoryError as error:
            if attempt == 1:
                raise OracleBackendFailure("torch_cuda", "cuda_oom") from error
            cleanup()
            block_size = max(1, block_size // 2)
    raise AssertionError("unreachable")


def _derived_target_block_size(
    *,
    state_count: int,
    symbol_count: int,
    episode_count: int,
    cuda_memory_fraction: float,
) -> int:
    free_bytes, _ = torch.cuda.mem_get_info()
    budget = max(1, int(free_bytes * cuda_memory_fraction))
    # Transition candidates retain two symbol tensors plus several scalar tensors.
    bytes_per_candidate = max(1, 8 * (2 * symbol_count + 8) + 16)
    denominator = max(1, episode_count * state_count * bytes_per_candidate)
    return max(1, min(state_count, budget // denominator))


def solve_torch_cuda_oracle_batch(
    *,
    tape: OracleMarketTape,
    states: np.ndarray,
    episode_inputs: OracleEpisodeInputs,
    parameters: OracleBellmanParameters,
    solver_config: OracleSolverConfig,
) -> OracleSolveResult:
    """Solve one batch on CUDA with compilation fallback and one OOM retry."""

    if not isinstance(solver_config, OracleSolverConfig):
        raise ValueError("solver_config must be OracleSolverConfig")
    if solver_config.selection not in {"cuda", "cuda_or_numpy"}:
        raise ValueError("Torch CUDA solver requires a CUDA selection")
    _validated_compile_chunk_size(solver_config)
    if not torch.cuda.is_available():
        raise OracleBackendFailure("torch_cuda", "CUDA is unavailable")

    device = torch.device("cuda", torch.cuda.current_device())
    torch_tape = transfer_market_tape_to_torch(tape, device=device)
    state_tensor = torch.tensor(states, dtype=torch.float64, device=device)
    requested_block = solver_config.target_state_block_size
    initial_block = requested_block or _derived_target_block_size(
        state_count=int(state_tensor.shape[0]),
        symbol_count=int(state_tensor.shape[1]),
        episode_count=episode_inputs.episode_count,
        cuda_memory_fraction=solver_config.cuda_memory_fraction,
    )
    initial_block = max(1, min(initial_block, int(state_tensor.shape[0])))

    compiled_core, compile_setup_reason = _prepare_compiled_core(solver_config)

    def cleanup() -> None:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    def solve_block(
        block_size: int,
    ) -> tuple[TorchBellmanResult, CompileMode, str | None]:
        run_config = replace(
            solver_config,
            target_state_block_size=block_size,
        )
        eager_config = replace(run_config, compile_mode="disabled")

        def eager() -> TorchBellmanResult:
            return _solve_torch_oracle_batch_core(
                tape=torch_tape,
                states=state_tensor,
                episode_inputs=episode_inputs,
                parameters=parameters,
                solver_config=eager_config,
            )

        if compiled_core is None:
            return eager(), "disabled", None

        def compiled() -> TorchBellmanResult:
            return compiled_core(
                tape=torch_tape,
                states=state_tensor,
                episode_inputs=episode_inputs,
                parameters=parameters,
                solver_config=run_config,
            )

        return _run_compiled_or_eager(compiled=compiled, eager=eager)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    solve_output, block_size, retry_performed = _run_with_oom_retry(
        solve=solve_block,
        initial_block_size=initial_block,
        cleanup=cleanup,
    )
    result, actual_compile_mode, execution_fallback_reason = solve_output
    fallback_reason = _resolve_compile_fallback_reason(
        setup_reason=compile_setup_reason,
        execution_reason=execution_fallback_reason,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_device = int(torch.cuda.max_memory_allocated(device))
    properties = torch.cuda.get_device_properties(device)
    target_paths = tuple(
        np.asarray(path, dtype=np.float32)
        for path in result.target_paths.detach().cpu().numpy()
    )
    final_scores = np.asarray(
        result.final_scores.detach().cpu().numpy(), dtype=np.float64
    )
    return OracleSolveResult(
        targets=target_paths,
        final_scores=final_scores,
        provenance=OracleSolverProvenance(
            backend="torch_cuda",
            solver_config_digest=solver_config.digest,
            market_tape_digest=tape.digest,
            numeric_dtype="float64",
            tie_tolerance=solver_config.tie_tolerance,
            episode_batch_size=solver_config.episode_batch_size,
            target_state_block_size=block_size,
            compile_mode=actual_compile_mode,
            compile_chunk_size=solver_config.compile_chunk_size,
            fallback_reason=fallback_reason,
            oom_retry_performed=retry_performed,
            solver_wall_time_seconds=elapsed,
            peak_device_memory_bytes=peak_device,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            device_name=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
        ),
    )


__all__ = [
    "TorchBellmanResult",
    "TorchMarketTape",
    "reduce_candidates_torch",
    "solve_torch_cuda_oracle_batch",
    "transfer_market_tape_to_torch",
]
