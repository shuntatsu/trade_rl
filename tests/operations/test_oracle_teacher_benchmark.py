from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


def _contract(market: MarketDataset) -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=market.dataset_id,
        episode_index=0,
        start=0,
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
    ):
        values = {
            "episode_count": 1,
            "episode_bars": 1,
            "state_count": 1,
            "symbol_count": 1,
            "repetitions": 1,
        }
        values[field] = 0
        try:
            OracleBenchmarkCase(**values)
        except ValueError as error:
            assert field in str(error)
        else:  # pragma: no cover - assertion branch
            raise AssertionError(f"{field} accepted zero")


def test_legacy_benchmark_returns_finite_non_negative_timings() -> None:
    market = _market()
    result = run_oracle_teacher_benchmark(
        market,
        (_contract(market),),
        OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero()),
        backend="legacy_numpy",
        repetitions=2,
    )

    assert isinstance(result, OracleBenchmarkResult)
    assert result.backend == "legacy_numpy"
    assert np.isfinite(result.cold_seconds)
    assert result.cold_seconds >= 0.0
    assert len(result.steady_seconds) == 2
    assert np.isfinite(result.steady_seconds).all()
    assert all(value >= 0.0 for value in result.steady_seconds)
    assert result.metadata["episode_count"] == 1
    assert result.metadata["episode_bars"] == market.n_bars - 1


def test_cli_writes_canonical_json(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"

    exit_code = main(
        [
            "--backend",
            "legacy_numpy",
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
    assert payload["backend"] == "legacy_numpy"
    assert payload["case"]["episode_count"] == 1
    assert payload["case"]["episode_bars"] == 8
