from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    load_market_dataset_artifact,
    load_published_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.build import MarketDatasetBuilder
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


def test_market_dataset_publisher_never_replaces_racing_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trade_rl.data.artifacts import publication

    root = tmp_path / "artifact"
    original_write = publication._write_arrays

    def stage_then_publish_competing_directory(path: Path, dataset):
        files = original_write(path, dataset)
        root.mkdir()
        return files

    monkeypatch.setattr(
        publication, "_write_arrays", stage_then_publish_competing_directory
    )
    with pytest.raises(FileExistsError):
        publish_market_dataset_artifact(root, _dataset())

    assert root.is_dir() and not tuple(root.iterdir())
    assert not tuple(tmp_path.glob(".artifact.staging-*"))


def test_failed_staging_is_removed_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.data.artifacts import publication as artifact_module

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
    root = tmp_path / "artifact"
    result = publish_market_dataset_artifact(root, _dataset())

    assert result.root == root
    assert result.manifest_path == root / "manifest.json"
    assert result.arrays_path == root / "arrays.npz"
    assert len(result.artifact_digest) == 64


def test_inspection_does_not_pair_dataset_with_a_second_manifest_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.data import inspect_published_market_dataset_artifact

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_dataset = _dataset()
    second_dataset = replace(
        first_dataset,
        dataset_id="0" * 64,
        identity_payload_json=None,
        features=first_dataset.features + np.float32(0.125),
    ).with_content_identity()
    first_artifact = publish_market_dataset_artifact(first_root, first_dataset)
    second_artifact = publish_market_dataset_artifact(second_root, second_dataset)
    second_manifest = second_artifact.manifest_path.read_text(encoding="utf-8")

    original_read_text = Path.read_text
    manifest_reads = 0

    def replace_manifest_on_second_read(
        path: Path, *args: object, **kwargs: object
    ) -> str:
        nonlocal manifest_reads
        if path == first_artifact.manifest_path:
            manifest_reads += 1
            if manifest_reads == 2:
                return second_manifest
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", replace_manifest_on_second_read)

    inspected = inspect_published_market_dataset_artifact(first_root)

    assert inspected == first_artifact
    assert manifest_reads == 1


def test_published_dataset_loader_returns_one_verified_identity_data_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.data.artifacts import codec

    dataset = _dataset()
    published = publish_market_dataset_artifact(tmp_path / "artifact", dataset)
    manifest_path = published.manifest_path
    arrays_path = published.arrays_path
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    original_np_load = codec.np.load
    manifest_reads = 0
    array_reads = 0
    array_decodes = 0

    def count_manifest_reads(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal manifest_reads
        if path == manifest_path:
            manifest_reads += 1
        return original_read_text(path, *args, **kwargs)

    def count_array_reads(path: Path) -> bytes:
        nonlocal array_reads
        if path == arrays_path:
            array_reads += 1
        return original_read_bytes(path)

    def count_array_decodes(*args: object, **kwargs: object):
        nonlocal array_decodes
        array_decodes += 1
        return original_np_load(*args, **kwargs)

    monkeypatch.setattr(Path, "read_text", count_manifest_reads)
    monkeypatch.setattr(Path, "read_bytes", count_array_reads)
    monkeypatch.setattr(codec.np, "load", count_array_decodes)

    identity, restored = load_published_market_dataset_artifact(published.root)

    assert identity == published
    assert restored.dataset_id == dataset.dataset_id
    np.testing.assert_array_equal(restored.features, dataset.features)
    np.testing.assert_array_equal(restored.close, dataset.close)
    assert manifest_reads == 1
    assert array_reads == 1
    assert array_decodes == 1


def test_deprecated_direct_dataset_writer_is_removed() -> None:
    import trade_rl.data as data

    assert not hasattr(data, "write_market_dataset_artifact")


def test_loader_reads_frozen_market_build_v2_artifact(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    fixture = fixtures / "market_build_v2_artifact"
    root = tmp_path / "historical-v2"
    shutil.copytree(fixture, root)

    provenance = json.loads(
        (fixtures / "market_build_v2_artifact_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance == {
        "artifact_digest": "5c7ff5aa27c682a3320e14130b07a95d2adac8b63b855db1b78c4d6103736ed0",
        "dataset_id": "e79ca62deb65ec3bea476869e5a642665e68ca92865c688e452415c8186db96e",
        "feature_config_digest": "e0f167d7d258b0493f9d8181b84f5b0823081cf60e62d642f75c6dfc20187539",
        "historical_git_sha": "045dc84fb40ada3b5d782028448b938d9a65dfea",
        "market_build_schema": "market_build_v2",
        "normalization_digest": "330a9134ccb9b35ecd4769ff524e738ee8a1816b43b5696a6048f23adcb87aa7",
        "schema_version": "issue494_historical_market_build_v2_fixture_v1",
    }

    dataset = load_market_dataset_artifact(root)
    artifact_identity, paired_dataset = load_published_market_dataset_artifact(root)
    assert dataset.dataset_id == provenance["dataset_id"]
    assert paired_dataset.dataset_id == provenance["dataset_id"]
    assert artifact_identity.artifact_digest == provenance["artifact_digest"]
    assert dataset.feature_config_digest == provenance["feature_config_digest"]
    assert dataset.normalization_digest == provenance["normalization_digest"]
    identity = json.loads(dataset.identity_payload_json or "{}")
    config = identity.get("config")
    assert isinstance(config, dict)
    assert config.get("schema_version") == "market_build_v2"
    assert "feature_numerics_schema" not in config
