from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact, publish_market_dataset_artifact
from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def _dataset():
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        24
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(24, dtype=np.float64)
    open_price = np.concatenate([close[:1], close[:-1]])
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full(24, 1_000.0),
        funding_rate=np.zeros(24),
        tradable=np.ones(24, dtype=np.bool_),
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
    )
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (contract,),
    )


def test_market_dataset_recomputes_identity_payload() -> None:
    dataset = _dataset()

    with pytest.raises(ValueError, match="dataset_id"):
        replace(dataset, dataset_id="f" * 64)


def test_artifact_loader_rejects_tampered_dataset_id_with_valid_outer_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    publish_market_dataset_artifact(root, _dataset())
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_id"] = "f" * 64
    digest_payload = dict(manifest)
    digest_payload.pop("artifact_digest")
    manifest["artifact_digest"] = content_digest(digest_payload)
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="dataset_id"):
        load_market_dataset_artifact(root)


def test_atomic_publish_rejects_existing_destination_without_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    sentinel = root / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        publish_market_dataset_artifact(root, _dataset())

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert tuple(root.iterdir()) == (sentinel,)


def test_failed_staging_is_removed_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.data import artifact as artifact_module

    root = tmp_path / "artifact"

    def fail_write(path: Path, dataset) -> None:
        del path, dataset
        raise RuntimeError("write failed")

    monkeypatch.setattr(artifact_module, "_write_arrays", fail_write)

    with pytest.raises(RuntimeError, match="write failed"):
        publish_market_dataset_artifact(root, _dataset())

    assert not root.exists()
    assert not tuple(tmp_path.glob(".artifact.staging-*"))


def test_publish_market_dataset_artifact_returns_typed_result(tmp_path: Path) -> None:
    from trade_rl.data import publish_market_dataset_artifact

    root = tmp_path / "artifact"
    result = publish_market_dataset_artifact(root, _dataset())

    assert result.root == root
    assert result.manifest_path == root / "manifest.json"
    assert result.arrays_path == root / "arrays.npz"
    assert len(result.artifact_digest) == 64


def test_legacy_atomic_writer_warns_and_preserves_digest_return(tmp_path: Path) -> None:
    from trade_rl.data import artifact as artifact_module

    with pytest.warns(DeprecationWarning, match="publish_market_dataset_artifact"):
        result = artifact_module.write_market_dataset_artifact(
            tmp_path / "artifact", _dataset()
        )

    assert isinstance(result, str)
    assert len(result) == 64
