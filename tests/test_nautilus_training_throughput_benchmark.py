from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import PublishedDatasetArtifact, publish_market_dataset_artifact
from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from tools.nautilus_training_throughput_benchmark import (
    _DEFAULT_TIMESTEPS,
    _benchmark_source_digest,
    _normalize_timesteps,
    _resolve_benchmark_dataset_source,
)


def _publish_benchmark_dataset(
    tmp_path: Path,
    *,
    symbol: str = "BTCUSDT",
    n_bars: int = 96,
) -> tuple[Path, PublishedDatasetArtifact]:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(n_bars, dtype=np.float64) * 0.01
    open_price = np.concatenate([close[:1], close[:-1]])
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=open_price,
        high=np.maximum(open_price, close) + 0.1,
        low=np.minimum(open_price, close) - 0.1,
        close=close,
        volume=np.full(n_bars, 1_000_000.0),
        funding_rate=np.zeros(n_bars),
        tradable=np.ones(n_bars, dtype=np.bool_),
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
        )
    ).build(
        InMemoryMarketDataSource({symbol: raw}),
        (
            InstrumentContract(
                symbol=symbol,
                listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ),
    )
    root = tmp_path / f"{symbol.lower()}-{n_bars}"
    return root, publish_market_dataset_artifact(root, dataset)


def test_default_timesteps_cover_broader_performance_workloads() -> None:
    assert _DEFAULT_TIMESTEPS == (8, 32, 128)


def test_normalize_timesteps_accepts_scalar_and_canonicalizes_sequence() -> None:
    assert _normalize_timesteps(8) == (8,)
    assert _normalize_timesteps([32, 8, 32]) == (8, 32)


def test_normalize_timesteps_rejects_bool_values_explicitly() -> None:
    with pytest.raises(TypeError, match="timesteps must contain integers"):
        _normalize_timesteps(True)
    with pytest.raises(TypeError, match="timesteps must contain integers"):
        _normalize_timesteps([8, True])


def test_benchmark_source_digest_binds_persisted_dataset_identity() -> None:
    first = _benchmark_source_digest((8, 32, 128), dataset_source_digest="a" * 64)
    second = _benchmark_source_digest((8, 32, 128), dataset_source_digest="b" * 64)

    assert first != second


def test_benchmark_source_digest_rejects_invalid_persisted_dataset_identity() -> None:
    with pytest.raises(
        ValueError, match="dataset_source_digest must be a SHA-256 digest"
    ):
        _benchmark_source_digest((8,), dataset_source_digest="not-a-digest")


def test_resolve_benchmark_dataset_source_preserves_synthetic_default() -> None:
    source = _resolve_benchmark_dataset_source(None, workloads=(8, 32))

    assert source.dataset_kind == "deterministic_synthetic_btcusdt"
    assert source.artifact_root is None
    assert source.dataset_source_digest is None


def test_resolve_benchmark_dataset_source_binds_canonical_artifact(
    tmp_path: Path,
) -> None:
    root, published = _publish_benchmark_dataset(tmp_path)

    source = _resolve_benchmark_dataset_source(root, workloads=(8, 32))

    assert source.dataset_kind == "persisted_market_dataset_artifact"
    assert source.artifact_root == root.resolve()
    assert source.dataset_source_digest == published.artifact_digest


def test_resolve_benchmark_dataset_source_rejects_wrong_symbol(tmp_path: Path) -> None:
    root, _ = _publish_benchmark_dataset(tmp_path, symbol="ETHUSDT")

    with pytest.raises(ValueError, match="exactly BTCUSDT"):
        _resolve_benchmark_dataset_source(root, workloads=(8, 32))


def test_resolve_benchmark_dataset_source_rejects_short_artifact(
    tmp_path: Path,
) -> None:
    root, _ = _publish_benchmark_dataset(tmp_path, n_bars=64)

    with pytest.raises(ValueError, match="at least 80 bars"):
        _resolve_benchmark_dataset_source(root, workloads=(8, 32))
