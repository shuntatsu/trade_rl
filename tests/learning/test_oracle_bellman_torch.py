from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from trade_rl.data.market import MarketDataset
from trade_rl.integrations.oracle_bellman_torch import (
    _solve_torch_oracle_batch_core,
    reduce_candidates_torch,
    solve_torch_cuda_oracle_batch,
    transfer_market_tape_to_torch,
)
from trade_rl.integrations.oracle_transition_torch import torch_transition_step
from trade_rl.learning.oracle_bellman_contracts import (
    OracleBackendFailure,
    OracleEpisodeInputs,
    OracleSolverConfig,
)
from trade_rl.learning.oracle_bellman_numpy import (
    reduce_candidates_numpy,
    solve_numpy_oracle_batch,
)
from trade_rl.learning.oracle_market_tape import build_oracle_market_tape
from trade_rl.learning.oracle_teacher import OracleTeacherConfig, portfolio_states
from trade_rl.learning.oracle_transition_numpy import numpy_transition_step
from trade_rl.simulation.execution import ExecutionCostConfig


def _market(close_values: np.ndarray) -> MarketDataset:
    close = np.asarray(close_values, dtype=np.float64)
    if close.ndim == 1:
        close = close[:, None]
    n_bars, n_symbols = close.shape
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="8" * 64,
        symbols=tuple(f"S{index}" for index in range(n_symbols)),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _episodes(
    *,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    initial_weights: np.ndarray,
) -> OracleEpisodeInputs:
    return OracleEpisodeInputs(
        episode_indices=np.arange(len(starts), dtype=np.int64),
        starts=np.asarray(starts, dtype=np.int64),
        stops=np.asarray(stops, dtype=np.int64),
        initial_weights=np.asarray(initial_weights, dtype=np.float64),
    )


def test_reduce_candidates_torch_matches_numpy_tie_contract() -> None:
    scores = np.array(
        [[[1.0, -np.inf], [1.0 + 5e-13, -np.inf], [0.0, -np.inf]]],
        dtype=np.float64,
    )
    expected_best, expected_pointers = reduce_candidates_numpy(
        scores,
        tie_tolerance=1e-12,
    )

    best, pointers = reduce_candidates_torch(
        torch.as_tensor(scores, dtype=torch.float64),
        tie_tolerance=1e-12,
    )

    np.testing.assert_allclose(best.numpy(), expected_best, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(pointers.numpy(), expected_pointers)
    assert pointers.dtype == torch.int16


def test_torch_transition_matches_numpy_on_cpu() -> None:
    close = np.column_stack(
        [
            100.0 * np.exp(np.arange(7) * 0.02),
            100.0 * np.exp(np.arange(7) * -0.01),
        ]
    )
    market = _market(close)
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig(
            fee_rate=0.001,
            spread_rate=0.0005,
            impact_rate=0.0002,
            max_participation_rate=0.5,
        )
    )
    tape = build_oracle_market_tape(market, (0, 7), config.bellman_parameters)
    torch_tape = transfer_market_tape_to_torch(tape, device=torch.device("cpu"))
    scores = np.array([[0.0, -0.1]], dtype=np.float64)
    weights = np.array([[[0.0, 0.0], [0.2, -0.1]]], dtype=np.float64)
    targets = np.array([[0.0, 0.0], [0.3, -0.2]], dtype=np.float64)

    expected = numpy_transition_step(
        tape=tape,
        step=0,
        prior_scores=scores,
        prior_close_weights=weights,
        targets=targets,
        parameters=config.bellman_parameters,
    )
    actual = torch_transition_step(
        tape=torch_tape,
        step=torch.tensor(0),
        prior_scores=torch.as_tensor(scores, dtype=torch.float64),
        prior_close_weights=torch.as_tensor(weights, dtype=torch.float64),
        targets=torch.as_tensor(targets, dtype=torch.float64),
        parameters=config.bellman_parameters,
    )

    np.testing.assert_array_equal(actual.valid.numpy(), expected.valid)
    np.testing.assert_array_equal(
        actual.fill_classification.numpy(), expected.fill_classification
    )
    np.testing.assert_allclose(
        actual.close_factor.numpy(), expected.close_factor, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.close_weights.numpy(), expected.close_weights, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.effective_targets.numpy(),
        expected.effective_targets,
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("signal_delay", [0, 1])
@pytest.mark.parametrize("target_block_size", [None, 1, 2])
def test_torch_solver_core_matches_numpy(
    signal_delay: int,
    target_block_size: int | None,
) -> None:
    close = np.column_stack(
        [
            100.0 * np.exp(np.arange(8) * 0.025),
            100.0 * np.exp(np.arange(8) * -0.015),
        ]
    )
    market = _market(close)
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        signal_delay_decisions=signal_delay,
    )
    states = portfolio_states(market, config)
    episodes = _episodes(
        starts=(0, 1),
        stops=(7, 8),
        initial_weights=np.array([[0.0, 0.0], [0.2, -0.1]]),
    )
    tape = build_oracle_market_tape(market, (0, 8), config.bellman_parameters)
    numpy_config = OracleSolverConfig(
        selection="numpy",
        episode_batch_size=2,
        target_state_block_size=target_block_size,
    )
    torch_config = replace(numpy_config, selection="cuda")
    expected = solve_numpy_oracle_batch(
        tape=tape,
        states=states,
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=numpy_config,
    )

    actual = _solve_torch_oracle_batch_core(
        tape=transfer_market_tape_to_torch(tape, device=torch.device("cpu")),
        states=torch.as_tensor(states, dtype=torch.float64),
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=torch_config,
    )

    np.testing.assert_allclose(
        actual.final_scores.numpy(), expected.final_scores, rtol=1e-10, atol=1e-12
    )
    assert actual.target_paths.dtype == torch.float32
    for batch_index, expected_path in enumerate(expected.targets):
        np.testing.assert_array_equal(
            actual.target_paths[batch_index].numpy(), expected_path
        )


def test_torch_solver_preserves_minimum_notional_noop_path() -> None:
    market = _market(100.0 * np.exp(np.arange(7) * 0.02))
    market = replace(
        market,
        minimum_notional=np.full_like(market.close, 1_000_000.0),
    )
    config = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    states = portfolio_states(market, config)
    episodes = _episodes(
        starts=(0,),
        stops=(7,),
        initial_weights=np.zeros((1, 1)),
    )
    tape = build_oracle_market_tape(market, (0, 7), config.bellman_parameters)
    numpy_config = OracleSolverConfig(selection="numpy", target_state_block_size=1)
    expected = solve_numpy_oracle_batch(
        tape=tape,
        states=states,
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=numpy_config,
    )

    actual = _solve_torch_oracle_batch_core(
        tape=transfer_market_tape_to_torch(tape, device=torch.device("cpu")),
        states=torch.as_tensor(states, dtype=torch.float64),
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=replace(numpy_config, selection="cuda"),
    )

    np.testing.assert_array_equal(actual.target_paths[0].numpy(), expected.targets[0])
    np.testing.assert_allclose(actual.final_scores.numpy(), expected.final_scores)


def test_cuda_wrapper_fails_closed_when_cuda_is_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU-only contract test")
    market = _market(np.linspace(100.0, 106.0, 7))
    config = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    states = portfolio_states(market, config)
    episodes = _episodes(
        starts=(0,),
        stops=(7,),
        initial_weights=np.zeros((1, 1)),
    )
    tape = build_oracle_market_tape(market, (0, 7), config.bellman_parameters)

    with pytest.raises(OracleBackendFailure, match="CUDA is unavailable"):
        solve_torch_cuda_oracle_batch(
            tape=tape,
            states=states,
            episode_inputs=episodes,
            parameters=config.bellman_parameters,
            solver_config=OracleSolverConfig(selection="cuda"),
        )


def test_oom_retry_halves_target_block_once() -> None:
    from trade_rl.integrations.oracle_bellman_torch import _run_with_oom_retry

    calls: list[int] = []

    def solve(block_size: int) -> str:
        calls.append(block_size)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("synthetic")
        return "ok"

    result, block_size, retried = _run_with_oom_retry(
        solve=solve,
        initial_block_size=8,
        cleanup=lambda: None,
    )

    assert result == "ok"
    assert block_size == 4
    assert retried
    assert calls == [8, 4]


def test_oom_retry_fails_after_second_out_of_memory() -> None:
    from trade_rl.integrations.oracle_bellman_torch import _run_with_oom_retry

    calls: list[int] = []

    def solve(block_size: int) -> str:
        calls.append(block_size)
        raise torch.OutOfMemoryError("synthetic")

    with pytest.raises(OracleBackendFailure, match="cuda_oom"):
        _run_with_oom_retry(
            solve=solve,
            initial_block_size=8,
            cleanup=lambda: None,
        )

    assert calls == [8, 4]


def test_forward_solver_core_contains_no_explicit_host_transfer() -> None:
    import inspect

    source = inspect.getsource(_solve_torch_oracle_batch_core)
    assert ".cpu(" not in source
    assert ".numpy(" not in source
    assert ".item(" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_eager_solver_matches_numpy_reference() -> None:
    close = np.column_stack(
        [
            100.0 * np.exp(np.arange(8) * 0.02),
            100.0 * np.exp(np.arange(8) * -0.01),
        ]
    )
    market = _market(close)
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        signal_delay_decisions=1,
    )
    states = portfolio_states(market, config)
    episodes = _episodes(
        starts=(0, 1),
        stops=(7, 8),
        initial_weights=np.array([[0.0, 0.0], [0.2, -0.1]], dtype=np.float64),
    )
    tape = build_oracle_market_tape(market, (0, 8), config.bellman_parameters)
    numpy_config = OracleSolverConfig(
        selection="numpy",
        episode_batch_size=2,
        target_state_block_size=1,
    )
    expected = solve_numpy_oracle_batch(
        tape=tape,
        states=states,
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=numpy_config,
    )

    actual = solve_torch_cuda_oracle_batch(
        tape=tape,
        states=states,
        episode_inputs=episodes,
        parameters=config.bellman_parameters,
        solver_config=OracleSolverConfig(
            selection="cuda",
            episode_batch_size=2,
            target_state_block_size=1,
        ),
    )

    np.testing.assert_allclose(
        actual.final_scores,
        expected.final_scores,
        rtol=1e-10,
        atol=1e-12,
    )
    for actual_path, expected_path in zip(
        actual.targets, expected.targets, strict=True
    ):
        np.testing.assert_array_equal(actual_path, expected_path)
    assert actual.provenance.backend == "torch_cuda"
    assert actual.provenance.target_state_block_size == 1
    assert actual.provenance.compile_mode == "disabled"
    assert actual.provenance.fallback_reason is None
