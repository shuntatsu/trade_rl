from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.learning.oracle_bellman_contracts import (
    OracleBackendFailure,
    OracleEpisodeInputs,
    OracleSolveResult,
    OracleSolverConfig,
    OracleSolverProvenance,
)
from trade_rl.learning.oracle_teacher import OracleTeacherConfig
from trade_rl.simulation.execution import ExecutionCostConfig


def test_solver_config_digest_includes_tie_tolerance() -> None:
    first = OracleSolverConfig(selection="numpy", tie_tolerance=1e-12)
    second = OracleSolverConfig(selection="numpy", tie_tolerance=1e-11)
    assert first.digest != second.digest


@pytest.mark.parametrize("selection", ["numpy", "cuda", "cuda_or_numpy"])
def test_solver_config_accepts_maintained_selection(selection: str) -> None:
    assert OracleSolverConfig(selection=selection).selection == selection


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("numeric_dtype", "float32"),
        ("tie_tolerance", 0.0),
        ("episode_batch_size", 0),
        ("target_state_block_size", 0),
        ("cuda_memory_fraction", 0.0),
        ("cuda_memory_fraction", 1.1),
        ("compile_mode", "other"),
        ("compile_chunk_size", 0),
    ],
)
def test_solver_config_rejects_unsupported_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        OracleSolverConfig(**{field: value})


def test_teacher_config_exports_backend_neutral_parameters() -> None:
    teacher = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        positions=(-1.0, 0.0, 1.0),
        signal_delay_decisions=1,
    )

    parameters = teacher.bellman_parameters

    assert parameters.execution_cost == teacher.execution_cost
    assert parameters.portfolio_risk == teacher.portfolio_risk
    assert parameters.positions == teacher.positions
    assert parameters.signal_delay_decisions == 1
    assert parameters.digest == teacher.bellman_parameters.digest


def test_episode_inputs_copy_and_freeze_arrays() -> None:
    starts = np.array([1, 2], dtype=np.int64)
    stops = np.array([5, 6], dtype=np.int64)
    initial = np.zeros((2, 3), dtype=np.float64)

    inputs = OracleEpisodeInputs(
        episode_indices=np.array([0, 1], dtype=np.int64),
        starts=starts,
        stops=stops,
        initial_weights=initial,
    )
    starts[0] = 99
    initial[0, 0] = 1.0

    assert inputs.starts[0] == 1
    assert inputs.initial_weights[0, 0] == 0.0
    assert not inputs.starts.flags.writeable
    assert not inputs.initial_weights.flags.writeable


def test_solve_result_requires_one_target_per_episode() -> None:
    provenance = OracleSolverProvenance.numpy_reference(
        config=OracleSolverConfig(),
        market_tape_digest="a" * 64,
    )
    with pytest.raises(ValueError, match="episode"):
        OracleSolveResult(
            targets=(np.zeros((2, 1), dtype=np.float32),),
            final_scores=np.zeros(2, dtype=np.float64),
            provenance=provenance,
        )


def test_backend_failure_preserves_backend_and_reason() -> None:
    error = OracleBackendFailure("torch_cuda", "out_of_memory")
    assert error.backend == "torch_cuda"
    assert error.reason == "out_of_memory"
    assert "torch_cuda" in str(error)


def test_solver_provenance_digest_changes_with_backend() -> None:
    config = OracleSolverConfig()
    numpy_provenance = OracleSolverProvenance.numpy_reference(
        config=config,
        market_tape_digest="a" * 64,
    )
    cuda_provenance = replace(numpy_provenance, backend="torch_cuda", digest="")
    assert numpy_provenance.digest != cuda_provenance.digest
