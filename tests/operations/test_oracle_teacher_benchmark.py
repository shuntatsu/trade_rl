from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.learning.oracle_teacher import OracleTeacherConfig
from trade_rl.operations.oracle_teacher_benchmark import (
    OracleBenchmarkCase,
    OracleBenchmarkResult,
    main,
    run_oracle_teacher_benchmark,
)
from trade_rl.simulation.execution import ExecutionCostConfig


def _market(n_bars: int = 8) -> MarketDataset:
    close = (100.0 * np.exp(np.arange(n_bars, dtype=np.float64) * 0.01))[:, None]
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
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


def _contract(market: MarketDataset, index: int = 0) -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=market.dataset_id,
        episode_index=index,
        start=index,
        stop=market.n_bars,
        initial_state_mode="cash",
        initial_weights=np.zeros(market.n_symbols, dtype=np.float64),
    )


def test_benchmark_case_rejects_non_positive_values() -> None:
    for field in (
        "episode_count",
        "episode_bars",
        "state_count",
        "symbol_count",
        "repetitions",
        "episode_batch_size",
    ):
        values = {
            "episode_count": 1,
            "episode_bars": 1,
            "state_count": 1,
            "symbol_count": 1,
            "repetitions": 1,
            "episode_batch_size": 1,
        }
        values[field] = 0
        try:
            OracleBenchmarkCase(**values)
        except ValueError as error:
            assert field in str(error)
        else:
            raise AssertionError(f"{field} accepted zero")


def test_serial_and_batched_numpy_return_equal_output_digests() -> None:
    market = _market(10)
    contracts = (_contract(market, 0),)
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())

    serial = run_oracle_teacher_benchmark(
        market,
        contracts,
        teacher,
        backend="serial_numpy",
        repetitions=1,
    )
    batched = run_oracle_teacher_benchmark(
        market,
        contracts,
        teacher,
        backend="numpy_batched",
        repetitions=1,
    )

    assert isinstance(serial, OracleBenchmarkResult)
    assert serial.output_digest == batched.output_digest
    assert serial.metadata["baseline_note"] == (
        "maintained NumPy solver invoked one episode at a time; "
        "not the removed pre-refactor implementation"
    )
    assert batched.metadata["actual_backend"] == "numpy"
    assert batched.metadata["requested_compile_mode"] == "disabled"
    assert batched.metadata["actual_compile_mode"] == "disabled"
    assert batched.metadata["fallback_reason"] is None
    assert batched.metadata["oom_retry_performed"] is False
    assert "market-tape construction" in str(batched.metadata["total_wall_scope"])
    assert "host-to-device" in str(batched.metadata["solver_wall_scope"])
    assert serial.peak_device_allocated_bytes is None
    assert serial.peak_device_reserved_bytes is None
    assert len(serial.steady_solver_seconds) == 1
    assert np.isfinite(serial.steady_seconds).all()


def test_removed_legacy_label_is_rejected() -> None:
    market = _market(10)
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())

    with pytest.raises(ValueError, match="unsupported"):
        run_oracle_teacher_benchmark(
            market,
            (_contract(market, 0),),
            teacher,
            backend="legacy_numpy",
            repetitions=1,
        )


def test_cli_writes_canonical_json(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"

    exit_code = main(
        [
            "--backend",
            "numpy_batched",
            "--episode-count",
            "1",
            "--episode-bars",
            "8",
            "--repetitions",
            "1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    raw = output.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    payload = json.loads(raw)
    assert payload["backend"] == "numpy_batched"
    assert payload["case"]["episode_count"] == 1
    assert payload["case"]["episode_bars"] == 8
    assert len(payload["output_digest"]) == 64
    assert len(payload["steady_solver_seconds"]) == 1
    assert payload["steady_summary"]["minimum_seconds"] >= 0.0
    assert payload["peak_device_allocated_bytes"] is None
    assert payload["peak_device_reserved_bytes"] is None
