"""Reproducible Oracle teacher benchmark contracts and command-line runner."""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.learning.episode_oracle_teacher import (
    OracleEpisodeContract,
    episode_oracle_target_path,
)
from trade_rl.learning.oracle_teacher import OracleTeacherConfig
from trade_rl.simulation.execution import ExecutionCostConfig

ORACLE_BENCHMARK_SCHEMA: Final = "oracle_teacher_benchmark_v1"
_SUPPORTED_BACKENDS: Final = frozenset({"legacy_numpy"})


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
    schema_version: str = ORACLE_BENCHMARK_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "episode_count",
            "episode_bars",
            "state_count",
            "symbol_count",
            "repetitions",
        ):
            _positive_integer(getattr(self, field), field=field)
        if self.schema_version != ORACLE_BENCHMARK_SCHEMA:
            raise ValueError("unsupported Oracle benchmark schema")


@dataclass(frozen=True, slots=True)
class OracleBenchmarkResult:
    """Measured wall-clock and memory evidence for one Oracle backend."""

    backend: str
    cold_seconds: float
    steady_seconds: tuple[float, ...]
    peak_host_bytes: int | None
    peak_device_bytes: int | None
    metadata: dict[str, object]
    schema_version: str = ORACLE_BENCHMARK_SCHEMA

    def __post_init__(self) -> None:
        if self.backend not in _SUPPORTED_BACKENDS:
            raise ValueError("unsupported Oracle benchmark backend")
        if not math.isfinite(self.cold_seconds) or self.cold_seconds < 0.0:
            raise ValueError("cold_seconds must be finite and non-negative")
        steady = tuple(float(value) for value in self.steady_seconds)
        if not steady or any(not math.isfinite(value) or value < 0.0 for value in steady):
            raise ValueError("steady_seconds must contain finite non-negative values")
        for field, value in (
            ("peak_host_bytes", self.peak_host_bytes),
            ("peak_device_bytes", self.peak_device_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field} must be a non-negative integer or None")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if self.schema_version != ORACLE_BENCHMARK_SCHEMA:
            raise ValueError("unsupported Oracle benchmark schema")
        object.__setattr__(self, "steady_seconds", steady)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _run_legacy_numpy(
    dataset: MarketDataset,
    contracts: tuple[OracleEpisodeContract, ...],
    teacher_config: OracleTeacherConfig,
) -> None:
    for contract in contracts:
        episode_oracle_target_path(
            dataset,
            (contract.start, contract.stop),
            teacher_config,
            initial_weights=contract.initial_weights,
        )


def _measure_seconds(operation: object) -> float:
    if not callable(operation):
        raise TypeError("benchmark operation must be callable")
    started = time.perf_counter()
    operation()
    return time.perf_counter() - started


def run_oracle_teacher_benchmark(
    dataset: MarketDataset,
    contracts: tuple[OracleEpisodeContract, ...],
    teacher_config: OracleTeacherConfig,
    *,
    backend: str,
    repetitions: int,
) -> OracleBenchmarkResult:
    """Measure one Oracle implementation without changing teacher semantics."""

    repeat_count = _positive_integer(repetitions, field="repetitions")
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported Oracle benchmark backend: {backend}")
    resolved_contracts = tuple(contracts)
    if not resolved_contracts:
        raise ValueError("Oracle benchmark contracts must be non-empty")
    if any(contract.dataset_id != dataset.dataset_id for contract in resolved_contracts):
        raise ValueError("Oracle benchmark contract dataset identity mismatch")
    episode_lengths = tuple(contract.stop - contract.start - 1 for contract in resolved_contracts)
    if any(length <= 0 for length in episode_lengths):
        raise ValueError("Oracle benchmark contracts must contain decisions")

    operation = lambda: _run_legacy_numpy(  # noqa: E731 - named benchmark closure
        dataset,
        resolved_contracts,
        teacher_config,
    )
    tracemalloc.start()
    try:
        cold_seconds = _measure_seconds(operation)
        steady_seconds = tuple(_measure_seconds(operation) for _ in range(repeat_count))
        _, peak_host_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return OracleBenchmarkResult(
        backend=backend,
        cold_seconds=cold_seconds,
        steady_seconds=steady_seconds,
        peak_host_bytes=int(peak_host_bytes),
        peak_device_bytes=None,
        metadata={
            "dataset_id": dataset.dataset_id,
            "episode_bars": episode_lengths[0]
            if len(set(episode_lengths)) == 1
            else episode_lengths,
            "episode_count": len(resolved_contracts),
            "repetitions": repeat_count,
            "state_count": None,
            "symbol_count": dataset.n_symbols,
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


def _result_payload(
    result: OracleBenchmarkResult,
    *,
    case: OracleBenchmarkCase,
) -> dict[str, object]:
    return {
        "backend": result.backend,
        "case": asdict(case),
        "cold_seconds": result.cold_seconds,
        "metadata": result.metadata,
        "peak_device_bytes": result.peak_device_bytes,
        "peak_host_bytes": result.peak_host_bytes,
        "schema_version": result.schema_version,
        "steady_seconds": result.steady_seconds,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(_SUPPORTED_BACKENDS), required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--episode-bars", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
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
    teacher_config = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    result = run_oracle_teacher_benchmark(
        dataset,
        contracts,
        teacher_config,
        backend=arguments.backend,
        repetitions=arguments.repetitions,
    )
    case = OracleBenchmarkCase(
        episode_count=arguments.episode_count,
        episode_bars=arguments.episode_bars,
        state_count=3,
        symbol_count=dataset.n_symbols,
        repetitions=arguments.repetitions,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            _result_payload(result, case=case),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "ORACLE_BENCHMARK_SCHEMA",
    "OracleBenchmarkCase",
    "OracleBenchmarkResult",
    "main",
    "run_oracle_teacher_benchmark",
]
