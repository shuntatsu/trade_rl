from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import scripts.nautilus_training_throughput_benchmark as benchmark
from scripts.nautilus_training_throughput_benchmark import (
    _DEFAULT_TIMESTEPS,
    _benchmark_dataset_source_contract,
    _benchmark_source_digest,
    _load_worker_benchmark_dataset,
    _normalize_timesteps,
    _parse_args,
    _resolve_benchmark_dataset_source,
    _worker_command,
)
from trade_rl.data import PublishedDatasetArtifact, publish_market_dataset_artifact
from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.simulation.runtime_performance import RuntimePerformanceMeasurement


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


def _measurement(timesteps: int) -> RuntimePerformanceMeasurement:
    return RuntimePerformanceMeasurement(
        timesteps=timesteps,
        elapsed_seconds=1.0,
        steps_per_second=float(timesteps),
        peak_self_rss_bytes=100,
        peak_children_rss_bytes=0,
        peak_process_tree_rss_bytes=100,
        peak_process_count=1,
    )


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


def test_benchmark_dataset_source_contract_preserves_synthetic_fixture() -> None:
    dataset_kind, dataset_contract = _benchmark_dataset_source_contract(None)

    assert dataset_kind == "deterministic_synthetic_btcusdt"
    assert dataset_contract["dataset_id"] == "7" * 64
    assert dataset_contract["symbol"] == "BTCUSDT"


def test_benchmark_dataset_source_contract_uses_persisted_artifact_identity() -> None:
    digest = "a" * 64

    dataset_kind, dataset_contract = _benchmark_dataset_source_contract(digest)

    assert dataset_kind == "persisted_market_dataset_artifact"
    assert dataset_contract == {
        "artifact_digest": digest,
        "symbol": "BTCUSDT",
    }


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


def test_worker_command_preserves_synthetic_default(tmp_path: Path) -> None:
    command = _worker_command(
        mode="legacy",
        timesteps=8,
        measurement_path=tmp_path / "measurement.json",
        dataset_artifact=None,
    )

    assert "--worker-dataset-artifact" not in command


def test_worker_command_propagates_persisted_artifact(tmp_path: Path) -> None:
    artifact_root = (tmp_path / "artifact").resolve()
    command = _worker_command(
        mode="streaming",
        timesteps=8,
        measurement_path=tmp_path / "measurement.json",
        dataset_artifact=artifact_root,
    )

    flag_index = command.index("--worker-dataset-artifact")
    assert command[flag_index + 1] == str(artifact_root)


def test_worker_dataset_loader_preserves_synthetic_fixture() -> None:
    dataset = _load_worker_benchmark_dataset(None, timesteps=8)

    assert dataset.dataset_id == "7" * 64
    assert dataset.symbols == ("BTCUSDT",)
    assert dataset.n_bars == 80


def test_worker_dataset_loader_revalidates_persisted_artifact(tmp_path: Path) -> None:
    root, _ = _publish_benchmark_dataset(tmp_path)

    dataset = _load_worker_benchmark_dataset(root, timesteps=8)

    assert dataset.symbols == ("BTCUSDT",)
    assert dataset.n_bars == 96
    assert dataset.identity_verified is True


def test_process_tree_rss_support_rejects_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="requires Linux /proc"):
        benchmark._require_process_tree_rss_support()


def test_run_benchmark_binds_persisted_source_to_workers_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, published = _publish_benchmark_dataset(tmp_path)
    calls: list[tuple[str, Path | None]] = []

    def fake_worker(
        *,
        mode: str,
        timesteps: int,
        root: Path,
        dataset_artifact: Path | None,
    ) -> RuntimePerformanceMeasurement:
        del root
        calls.append((mode, dataset_artifact))
        return _measurement(timesteps)

    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda _: "1.230.0")
    monkeypatch.setattr(benchmark, "_run_worker_subprocess", fake_worker)
    monkeypatch.setattr(benchmark, "_require_process_tree_rss_support", lambda: None)

    evidence = benchmark.run_benchmark(timesteps=(8,), dataset_artifact=root)

    assert evidence["dataset_kind"] == "persisted_market_dataset_artifact"
    assert evidence["source_digest"] == _benchmark_source_digest(
        (8,),
        dataset_source_digest=published.artifact_digest,
    )
    assert evidence["performance_approved"] is False
    assert "persisted-dataset evidence" in evidence["approval_note"]
    assert "CI evidence" not in evidence["approval_note"]
    assert calls == [
        ("legacy", root.resolve()),
        ("streaming", root.resolve()),
    ]


def test_parse_args_preserves_synthetic_default_and_accepts_dataset_artifact(
    tmp_path: Path,
) -> None:
    default_args = _parse_args([])
    artifact_root = tmp_path / "artifact"
    persisted_args = _parse_args(["--dataset-artifact", str(artifact_root)])

    assert default_args.dataset_artifact is None
    assert persisted_args.dataset_artifact == artifact_root
