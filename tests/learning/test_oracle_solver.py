from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.learning.oracle_bellman_contracts import (
    OracleBackendFailure,
    OracleEpisodeInputs,
    OracleSolverConfig,
)
from trade_rl.learning.oracle_bellman_numpy import solve_numpy_oracle_batch
from trade_rl.learning.oracle_market_tape import build_oracle_market_tape
from trade_rl.learning.oracle_solver import solve_oracle_episodes
from trade_rl.learning.oracle_teacher import (
    OracleTeacherConfig,
    _portfolio_states,
    oracle_target_path,
)
from trade_rl.simulation.execution import ExecutionCostConfig


def _market() -> MarketDataset:
    phase = np.arange(10, dtype=np.float64)
    close = np.column_stack(
        [100.0 * np.exp(phase * 0.02), 80.0 * np.exp(phase * -0.01)]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="e" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(10) * np.timedelta64(15, "m"),
        features=np.zeros((10, 2, 1), dtype=np.float32),
        global_features=np.zeros((10, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((10, 2, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _inputs() -> OracleEpisodeInputs:
    return OracleEpisodeInputs(
        episode_indices=np.array([20, 10, 30], dtype=np.int64),
        starts=np.array([0, 2, 1], dtype=np.int64),
        stops=np.array([6, 8, 8], dtype=np.int64),
        initial_weights=np.array(
            [[0.0, 0.0], [0.2, -0.1], [0.0, 0.0]],
            dtype=np.float64,
        ),
    )


def test_orchestrator_groups_horizons_and_restores_input_order() -> None:
    market = _market()
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    states = _portfolio_states(market, teacher)
    inputs = _inputs()
    config = OracleSolverConfig(
        selection="numpy",
        episode_batch_size=1,
        target_state_block_size=2,
    )

    result = solve_oracle_episodes(
        market,
        states=states,
        episode_inputs=inputs,
        parameters=teacher.bellman_parameters,
        solver_config=config,
    )

    expected_targets: list[np.ndarray | None] = [None] * inputs.episode_count
    expected_scores = np.empty(inputs.episode_count, dtype=np.float64)
    tape = build_oracle_market_tape(
        market,
        (int(inputs.starts.min()), int(inputs.stops.max())),
        teacher.bellman_parameters,
    )
    horizons = inputs.stops - inputs.starts - 1
    for horizon in sorted(set(horizons.tolist())):
        positions = np.flatnonzero(horizons == horizon)
        for position in positions:
            subgroup = OracleEpisodeInputs(
                episode_indices=inputs.episode_indices[position : position + 1],
                starts=inputs.starts[position : position + 1],
                stops=inputs.stops[position : position + 1],
                initial_weights=inputs.initial_weights[position : position + 1],
            )
            expected = solve_numpy_oracle_batch(
                tape=tape,
                states=states,
                episode_inputs=subgroup,
                parameters=teacher.bellman_parameters,
                solver_config=config,
            )
            expected_targets[int(position)] = expected.targets[0]
            expected_scores[int(position)] = expected.final_scores[0]

    assert result.provenance.backend == "numpy"
    assert len(result.targets) == inputs.episode_count
    for actual, expected in zip(result.targets, expected_targets, strict=True):
        assert expected is not None
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(result.final_scores, expected_scores)


def test_orchestrator_digest_is_deterministic() -> None:
    market = _market()
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    states = _portfolio_states(market, teacher)
    inputs = _inputs()
    config = OracleSolverConfig(episode_batch_size=2, target_state_block_size=1)

    first = solve_oracle_episodes(
        market,
        states=states,
        episode_inputs=inputs,
        parameters=teacher.bellman_parameters,
        solver_config=config,
    )
    second = solve_oracle_episodes(
        market,
        states=states,
        episode_inputs=inputs,
        parameters=teacher.bellman_parameters,
        solver_config=config,
    )

    assert second.digest == first.digest


def test_orchestrator_rejects_cuda_until_backend_is_available() -> None:
    market = _market()
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())

    with pytest.raises(OracleBackendFailure, match="torch_cuda"):
        solve_oracle_episodes(
            market,
            states=_portfolio_states(market, teacher),
            episode_inputs=_inputs(),
            parameters=teacher.bellman_parameters,
            solver_config=OracleSolverConfig(selection="cuda"),
        )


def test_public_oracle_path_accepts_explicit_numpy_solver_config() -> None:
    market = _market()
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())

    default = oracle_target_path(market, (0, 8), teacher)
    explicit = oracle_target_path(
        market,
        (0, 8),
        teacher,
        solver_config=OracleSolverConfig(
            selection="numpy",
            episode_batch_size=1,
            target_state_block_size=1,
        ),
    )

    np.testing.assert_array_equal(explicit, default)


def test_orchestrator_routes_cuda_batches_to_torch_backend(monkeypatch) -> None:
    from dataclasses import replace

    import trade_rl.learning.oracle_solver as oracle_solver_module
    from trade_rl.learning.oracle_bellman_contracts import (
        OracleSolveResult,
        OracleSolverProvenance,
    )

    market = _market()
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    states = _portfolio_states(market, teacher)
    inputs = _inputs()
    calls: list[int] = []

    def fake_torch_backend(*, tape, states, episode_inputs, parameters, solver_config):
        calls.append(episode_inputs.episode_count)
        numpy_result = solve_numpy_oracle_batch(
            tape=tape,
            states=states,
            episode_inputs=episode_inputs,
            parameters=parameters,
            solver_config=replace(solver_config, selection="numpy"),
        )
        return OracleSolveResult(
            targets=numpy_result.targets,
            final_scores=numpy_result.final_scores,
            provenance=OracleSolverProvenance(
                backend="torch_cuda",
                solver_config_digest=solver_config.digest,
                market_tape_digest=tape.digest,
                numeric_dtype="float64",
                tie_tolerance=solver_config.tie_tolerance,
                episode_batch_size=solver_config.episode_batch_size,
                target_state_block_size=solver_config.target_state_block_size,
                compile_mode=solver_config.compile_mode,
                compile_chunk_size=solver_config.compile_chunk_size,
                torch_version="test",
                cuda_version="test",
                device_name="test",
                compute_capability="0.0",
            ),
        )

    monkeypatch.setattr(
        oracle_solver_module,
        "solve_torch_cuda_oracle_batch",
        fake_torch_backend,
    )

    result = solve_oracle_episodes(
        market,
        states=states,
        episode_inputs=inputs,
        parameters=teacher.bellman_parameters,
        solver_config=OracleSolverConfig(
            selection="cuda",
            episode_batch_size=2,
            target_state_block_size=1,
        ),
    )

    assert calls == [2, 1]
    assert result.provenance.backend == "torch_cuda"
