"""Reproducible Oracle Bellman backend benchmark and evidence writer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.learning.oracle_bellman_contracts import (
    OracleEpisodeInputs,
    OracleSolverConfig,
    OracleSolveResult,
    OracleSolverProvenance,
)
from trade_rl.learning.oracle_solver import solve_oracle_episodes
from trade_rl.learning.oracle_teacher import OracleTeacherConfig, _portfolio_states
from trade_rl.simulation.execution import ExecutionCostConfig

ORACLE_BENCHMARK_SCHEMA: Final = "oracle_teacher_benchmark_v2"
_SUPPORTED_BACKENDS: Final = (
    "legacy_numpy",
    "numpy_batched",
    "torch_cuda_eager",
    "torch_cuda_compiled",
)


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class OracleBenchmarkCase:
    """One immutable benchmark workload description."""

    episode_count: int
    episode_bars: int
    state_count: int
    symbol_count: int
    repetitions: int
    episode_batch_size: int = 8
    target_state_block_size: int | None = None
    compile_chunk_size: int = 16
    schema_version: str = ORACLE_BENCHMARK_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "episode_count",
            "episode_bars",
            "state_count",
            "symbol_count",
            "repetitions",
            "episode_batch_size",
            "compile_chunk_size",
        ):
            _positive_integer(getattr(self, field), field=field)
        if self.target_state_block_size is not None:
            _positive_integer(
                self.target_state_block_size,
                field="target_state_block_size",
            )
        if self.compile_chunk_size not in {8, 16, 32, 64}:
            raise ValueError("compile_chunk_size must be one of 8, 16, 32, or 64")
        if self.schema_version != ORACLE_BENCHMARK_SCHEMA:
            raise ValueError("unsupported Oracle benchmark schema")


@dataclass(frozen=True, slots=True)
class OracleBenchmarkResult:
    """Measured correctness, wall-clock, and memory evidence for one backend."""

    backend: str
    cold_seconds: float
    steady_seconds: tuple[float, ...]
    cold_solver_seconds: float | None
    steady_solver_seconds: tuple[float | None, ...]
    peak_host_bytes: int | None
    peak_device_allocated_bytes: int | None
    peak_device_reserved_bytes: int | None
    output_digest: str
    metadata: dict[str, object]
    schema_version: str = ORACLE_BENCHMARK_SCHEMA

    def __post_init__(self) -> None:
        if self.backend not in _SUPPORTED_BACKENDS:
            raise ValueError("unsupported Oracle benchmark backend")
        if not math.isfinite(self.cold_seconds) or self.cold_seconds < 0.0:
            raise ValueError("cold_seconds must be finite and non-negative")
        steady = tuple(float(value) for value in self.steady_seconds)
        if not steady or any(
            not math.isfinite(value) or value < 0.0 for value in steady
        ):
            raise ValueError("steady_seconds must contain finite non-negative values")
        solver = tuple(self.steady_solver_seconds)
        if len(solver) != len(steady):
            raise ValueError("steady solver timings must align with wall timings")
        for value in (self.cold_solver_seconds, *solver):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError("solver timings must be finite and non-negative")
        for field, value in (
            ("peak_host_bytes", self.peak_host_bytes),
            ("peak_device_allocated_bytes", self.peak_device_allocated_bytes),
            ("peak_device_reserved_bytes", self.peak_device_reserved_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field} must be a non-negative integer or None")
        if len(self.output_digest) != 64:
            raise ValueError("output_digest must be a SHA-256 digest")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if self.schema_version != ORACLE_BENCHMARK_SCHEMA:
            raise ValueError("unsupported Oracle benchmark schema")
        object.__setattr__(self, "steady_seconds", steady)
        object.__setattr__(self, "steady_solver_seconds", solver)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class _OperationEvidence:
    output_digest: str
    solver_seconds: float | None
    peak_device_allocated_bytes: int | None
    peak_device_reserved_bytes: int | None
    provenance: OracleSolverProvenance | None


def _episode_inputs(
    contracts: tuple[OracleEpisodeContract, ...],
) -> OracleEpisodeInputs:
    return OracleEpisodeInputs(
        episode_indices=np.asarray(
            [contract.episode_index for contract in contracts], dtype=np.int64
        ),
        starts=np.asarray([contract.start for contract in contracts], dtype=np.int64),
        stops=np.asarray([contract.stop for contract in contracts], dtype=np.int64),
        initial_weights=np.stack(
            [contract.initial_weights for contract in contracts], axis=0
        ),
    )


def _target_payload_digest(
    targets: tuple[np.ndarray, ...],
    scores: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for target in targets:
        array = np.ascontiguousarray(target)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    score_array = np.ascontiguousarray(scores, dtype=np.float64)
    digest.update(score_array.tobytes(order="C"))
    return digest.hexdigest()


def _synchronize_cuda_if_needed(backend: str) -> None:
    if not backend.startswith("torch_cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark backend requires available CUDA")
    torch.cuda.synchronize()


def _reset_cuda_peak_if_needed(backend: str) -> None:
    if not backend.startswith("torch_cuda"):
        return
    import torch

    torch.cuda.reset_peak_memory_stats()


def _solver_config(
    backend: str,
    *,
    episode_batch_size: int,
    target_state_block_size: int | None,
    compile_chunk_size: int,
) -> OracleSolverConfig:
    if backend in {"legacy_numpy", "numpy_batched"}:
        return OracleSolverConfig(
            selection="numpy",
            episode_batch_size=(1 if backend == "legacy_numpy" else episode_batch_size),
            target_state_block_size=target_state_block_size,
        )
    return OracleSolverConfig(
        selection="cuda",
        episode_batch_size=episode_batch_size,
        target_state_block_size=target_state_block_size,
        compile_mode=(
            "reduce_overhead" if backend == "torch_cuda_compiled" else "disabled"
        ),
        compile_chunk_size=compile_chunk_size,
    )


def _run_solver(
    dataset: MarketDataset,
    contracts: tuple[OracleEpisodeContract, ...],
    teacher_config: OracleTeacherConfig,
    *,
    backend: str,
    solver_config: OracleSolverConfig,
) -> _OperationEvidence:
    states = _portfolio_states(dataset, teacher_config)
    if backend == "legacy_numpy":
        targets: list[np.ndarray] = []
        scores: list[float] = []
        provenances: list[OracleSolverProvenance] = []
        for contract in contracts:
            serial_result = solve_oracle_episodes(
                dataset,
                states=states,
                episode_inputs=_episode_inputs((contract,)),
                parameters=teacher_config.bellman_parameters,
                solver_config=solver_config,
            )
            targets.extend(serial_result.targets)
            scores.extend(serial_result.final_scores.tolist())
            provenances.append(serial_result.provenance)
        solver_values = [
            value.solver_wall_time_seconds
            for value in provenances
            if value.solver_wall_time_seconds is not None
        ]
        peaks = [
            value.peak_device_memory_bytes
            for value in provenances
            if value.peak_device_memory_bytes is not None
        ]
        return _OperationEvidence(
            output_digest=_target_payload_digest(
                tuple(targets), np.asarray(scores, dtype=np.float64)
            ),
            solver_seconds=sum(solver_values) if solver_values else None,
            peak_device_allocated_bytes=max(peaks) if peaks else None,
            peak_device_reserved_bytes=None,
            provenance=provenances[0] if provenances else None,
        )
    batched_result: OracleSolveResult = solve_oracle_episodes(
        dataset,
        states=states,
        episode_inputs=_episode_inputs(contracts),
        parameters=teacher_config.bellman_parameters,
        solver_config=solver_config,
    )
    return _OperationEvidence(
        output_digest=_target_payload_digest(
            batched_result.targets, batched_result.final_scores
        ),
        solver_seconds=batched_result.provenance.solver_wall_time_seconds,
        peak_device_allocated_bytes=(
            batched_result.provenance.peak_device_memory_bytes
        ),
        peak_device_reserved_bytes=None,
        provenance=batched_result.provenance,
    )


def _measure_operation(
    operation: Callable[[], _OperationEvidence],
    *,
    backend: str,
) -> tuple[float, _OperationEvidence]:
    _reset_cuda_peak_if_needed(backend)
    _synchronize_cuda_if_needed(backend)
    started = time.perf_counter()
    evidence = operation()
    _synchronize_cuda_if_needed(backend)
    elapsed = time.perf_counter() - started
    if not backend.startswith("torch_cuda"):
        return elapsed, evidence
    import torch

    return elapsed, replace(
        evidence,
        peak_device_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_device_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )


def run_oracle_teacher_benchmark(
    dataset: MarketDataset,
    contracts: tuple[OracleEpisodeContract, ...],
    teacher_config: OracleTeacherConfig,
    *,
    backend: str,
    repetitions: int,
    episode_batch_size: int = 8,
    target_state_block_size: int | None = None,
    compile_chunk_size: int = 16,
) -> OracleBenchmarkResult:
    """Measure one backend and fail if repeated outputs drift."""

    repeat_count = _positive_integer(repetitions, field="repetitions")
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported Oracle benchmark backend: {backend}")
    resolved_contracts = tuple(contracts)
    if not resolved_contracts:
        raise ValueError("Oracle benchmark contracts must be non-empty")
    if any(
        contract.dataset_id != dataset.dataset_id for contract in resolved_contracts
    ):
        raise ValueError("Oracle benchmark contract dataset identity mismatch")
    episode_lengths = tuple(
        contract.stop - contract.start - 1 for contract in resolved_contracts
    )
    if any(length <= 0 for length in episode_lengths):
        raise ValueError("Oracle benchmark contracts must contain decisions")
    config = _solver_config(
        backend,
        episode_batch_size=_positive_integer(
            episode_batch_size, field="episode_batch_size"
        ),
        target_state_block_size=target_state_block_size,
        compile_chunk_size=_positive_integer(
            compile_chunk_size, field="compile_chunk_size"
        ),
    )

    def operation() -> _OperationEvidence:
        return _run_solver(
            dataset,
            resolved_contracts,
            teacher_config,
            backend=backend,
            solver_config=config,
        )

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        if was_tracing:
            tracemalloc.reset_peak()
        cold_seconds, cold = _measure_operation(operation, backend=backend)
        steady_pairs = tuple(
            _measure_operation(operation, backend=backend) for _ in range(repeat_count)
        )
        _, peak_host_bytes = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    if any(value.output_digest != cold.output_digest for _, value in steady_pairs):
        raise RuntimeError("Oracle benchmark output changed across repetitions")
    provenance = cold.provenance
    return OracleBenchmarkResult(
        backend=backend,
        cold_seconds=cold_seconds,
        steady_seconds=tuple(value for value, _ in steady_pairs),
        cold_solver_seconds=cold.solver_seconds,
        steady_solver_seconds=tuple(
            evidence.solver_seconds for _, evidence in steady_pairs
        ),
        peak_host_bytes=int(peak_host_bytes),
        peak_device_allocated_bytes=max(
            [
                value
                for value in (
                    cold.peak_device_allocated_bytes,
                    *(
                        evidence.peak_device_allocated_bytes
                        for _, evidence in steady_pairs
                    ),
                )
                if value is not None
            ],
            default=None,
        ),
        peak_device_reserved_bytes=max(
            [
                value
                for value in (
                    cold.peak_device_reserved_bytes,
                    *(
                        evidence.peak_device_reserved_bytes
                        for _, evidence in steady_pairs
                    ),
                )
                if value is not None
            ],
            default=None,
        ),
        output_digest=cold.output_digest,
        metadata={
            "dataset_id": dataset.dataset_id,
            "episode_bars": episode_lengths[0]
            if len(set(episode_lengths)) == 1
            else episode_lengths,
            "episode_count": len(resolved_contracts),
            "episode_batch_size": config.episode_batch_size,
            "requested_target_state_block_size": config.target_state_block_size,
            "actual_target_state_block_size": (
                None if provenance is None else provenance.target_state_block_size
            ),
            "requested_compile_mode": config.compile_mode,
            "actual_compile_mode": (
                None if provenance is None else provenance.compile_mode
            ),
            "compile_chunk_size": config.compile_chunk_size,
            "fallback_reason": (
                None if provenance is None else provenance.fallback_reason
            ),
            "oom_retry_performed": (
                False if provenance is None else provenance.oom_retry_performed
            ),
            "repetitions": repeat_count,
            "state_count": int(_portfolio_states(dataset, teacher_config).shape[0]),
            "symbol_count": dataset.n_symbols,
            "actual_backend": None if provenance is None else provenance.backend,
            "torch_version": None if provenance is None else provenance.torch_version,
            "cuda_version": None if provenance is None else provenance.cuda_version,
            "device_name": None if provenance is None else provenance.device_name,
            "compute_capability": (
                None if provenance is None else provenance.compute_capability
            ),
            "total_wall_scope": (
                "validation, market-tape construction, backend dispatch, transfers, "
                "solve, backtracking, host materialization, and result construction"
            ),
            "solver_wall_scope": (
                "backend provenance timer; CUDA excludes tape/state host-to-device "
                "transfer and output device-to-host materialization, includes the "
                "synchronized solve, first-call compilation when enabled, and "
                "backtracking"
            ),
            "compatibility_note": (
                "serial calls through the maintained NumPy solver; the removed "
                "pre-refactor implementation is not retained"
                if backend == "legacy_numpy"
                else None
            ),
        },
    )


def _synthetic_market(*, episode_bars: int, episode_count: int) -> MarketDataset:
    bars = _positive_integer(episode_bars, field="episode_bars")
    count = _positive_integer(episode_count, field="episode_count")
    n_bars = bars + count
    close = (100.0 * np.exp(np.arange(n_bars, dtype=np.float64) * 0.001))[:, None]
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _synthetic_contracts(
    dataset: MarketDataset,
    *,
    episode_count: int,
    episode_bars: int,
) -> tuple[OracleEpisodeContract, ...]:
    return tuple(
        OracleEpisodeContract(
            dataset_id=dataset.dataset_id,
            episode_index=index,
            start=index,
            stop=index + episode_bars + 1,
            initial_state_mode="cash",
            initial_weights=np.zeros(dataset.n_symbols, dtype=np.float64),
        )
        for index in range(episode_count)
    )


def _timing_summary(values: Sequence[float | None]) -> dict[str, float] | None:
    resolved = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    if resolved.size == 0:
        return None
    return {
        "maximum_seconds": float(np.max(resolved)),
        "median_seconds": float(np.median(resolved)),
        "minimum_seconds": float(np.min(resolved)),
    }


def _result_payload(
    result: OracleBenchmarkResult,
    *,
    case: OracleBenchmarkCase,
) -> dict[str, object]:
    return {
        "backend": result.backend,
        "case": asdict(case),
        "cold_seconds": result.cold_seconds,
        "cold_solver_seconds": result.cold_solver_seconds,
        "metadata": result.metadata,
        "output_digest": result.output_digest,
        "peak_device_allocated_bytes": result.peak_device_allocated_bytes,
        "peak_device_reserved_bytes": result.peak_device_reserved_bytes,
        "peak_host_bytes": result.peak_host_bytes,
        "schema_version": result.schema_version,
        "steady_seconds": result.steady_seconds,
        "steady_solver_seconds": result.steady_solver_seconds,
        "steady_summary": _timing_summary(result.steady_seconds),
        "steady_solver_summary": _timing_summary(result.steady_solver_seconds),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=(*_SUPPORTED_BACKENDS, "all"), required=True
    )
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--episode-bars", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--episode-batch-size", type=int, default=8)
    parser.add_argument("--target-state-block-size", type=int)
    parser.add_argument("--compile-chunk-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic synthetic benchmark and write canonical JSON."""

    arguments = _parser().parse_args(argv)
    dataset = _synthetic_market(
        episode_bars=arguments.episode_bars,
        episode_count=arguments.episode_count,
    )
    contracts = _synthetic_contracts(
        dataset,
        episode_count=arguments.episode_count,
        episode_bars=arguments.episode_bars,
    )
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    backends = (
        _SUPPORTED_BACKENDS if arguments.backend == "all" else (arguments.backend,)
    )
    results = tuple(
        run_oracle_teacher_benchmark(
            dataset,
            contracts,
            teacher,
            backend=backend,
            repetitions=arguments.repetitions,
            episode_batch_size=arguments.episode_batch_size,
            target_state_block_size=arguments.target_state_block_size,
            compile_chunk_size=arguments.compile_chunk_size,
        )
        for backend in backends
    )
    case = OracleBenchmarkCase(
        episode_count=arguments.episode_count,
        episode_bars=arguments.episode_bars,
        state_count=int(_portfolio_states(dataset, teacher).shape[0]),
        symbol_count=dataset.n_symbols,
        repetitions=arguments.repetitions,
        episode_batch_size=arguments.episode_batch_size,
        target_state_block_size=arguments.target_state_block_size,
        compile_chunk_size=arguments.compile_chunk_size,
    )
    payload: dict[str, object]
    if len(results) == 1:
        payload = _result_payload(results[0], case=case)
    else:
        output_digests = {result.output_digest for result in results}
        if len(output_digests) != 1:
            raise RuntimeError("Oracle benchmark backends produced different outputs")
        payload = {
            "case": asdict(case),
            "output_digest": next(iter(output_digests)),
            "results": tuple(_result_payload(result, case=case) for result in results),
            "schema_version": ORACLE_BENCHMARK_SCHEMA,
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ORACLE_BENCHMARK_SCHEMA",
    "OracleBenchmarkCase",
    "OracleBenchmarkResult",
    "main",
    "run_oracle_teacher_benchmark",
]
