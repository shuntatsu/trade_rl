from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    load_market_dataset_artifact,
    load_published_market_dataset_artifact,
    write_market_dataset_files,
)
from trade_rl.data.artifacts import load_dataset_files
from trade_rl.data.contracts import FeatureKind, FeatureSpec, MarketBuildConfig
from trade_rl.data.market import MarketDataset
from trade_rl.data.view import MarketDatasetView


def _dataset() -> MarketDataset:
    n_bars = 12
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.linspace(100.0, 111.0, n_bars, dtype=np.float64)[:, None]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.linspace(0.0, 1.0, n_bars, dtype=np.float32)[:, None, None],
        global_features=np.linspace(1.0, 2.0, n_bars, dtype=np.float32)[:, None],
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 10_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("momentum",),
        global_feature_names=("market",),
        periods_per_year=8_760,
        fee_rate=np.full((n_bars, 1), 0.0005),
        borrow_available=np.ones((n_bars, 1), dtype=np.bool_),
        cash_rate=np.linspace(0.0, 0.001, n_bars),
    ).with_content_identity()


def _dataset_v7() -> MarketDataset:
    dataset = _dataset()
    funding_price_rate = np.zeros((dataset.n_bars, dataset.n_symbols), dtype=np.float64)
    funding_price_rate[3, 0] = 0.075
    return MarketDataset(
        **{
            field: getattr(dataset, field)
            for field in (
                "dataset_id",
                "symbols",
                "timestamps",
                "features",
                "global_features",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "funding_rate",
                "tradable",
                "feature_available",
                "feature_names",
                "global_feature_names",
                "periods_per_year",
                "fee_rate",
                "borrow_available",
                "cash_rate",
            )
        },
        funding_price_rate=funding_price_rate,
    ).with_content_identity({"source": "synthetic-v7-test"})


def test_market_dataset_identity_versions_match_independent_golden_ids() -> None:
    legacy = _dataset()
    funding_price = _dataset_v7()

    assert json.loads(legacy.identity_payload_json or "{}")["schema"] == (
        "market_dataset_identity_v6"
    )
    assert legacy.dataset_id == (
        "06972ed5faf2bc233e1fb361f414f5b53baef1ee21d6ca84432acc0979bfce57"
    )

    assert json.loads(funding_price.identity_payload_json or "{}")["schema"] == (
        "market_dataset_identity_v7"
    )
    # Independently derived using the historical v6 field order, with
    # funding_price_rate inserted immediately after funding_rate.
    assert funding_price.dataset_id == (
        "b01ae0172016b374266b004799f92400350b615c927783921658ed1fca7dfa90"
    )


def _rewrite_artifact_arrays(root: Path, arrays: dict[str, np.ndarray]) -> None:
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    array_bytes = payload.getvalue()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("artifact_digest")
    manifest["arrays"] = {
        name: {"dtype": value.dtype.str, "shape": list(value.shape)}
        for name, value in sorted(arrays.items())
    }
    manifest["arrays_digest"] = hashlib.sha256(array_bytes).hexdigest()
    manifest["artifact_digest"] = content_digest(manifest)
    (root / "arrays.npz").write_bytes(array_bytes)
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def test_market_dataset_artifact_round_trip_preserves_resolved_arrays(
    tmp_path: Path,
) -> None:
    original = _dataset()

    manifest_path = write_market_dataset_files(tmp_path, original).manifest_path
    restored = load_market_dataset_artifact(tmp_path)
    direct_restored = load_dataset_files(tmp_path)

    assert isinstance(restored, MarketDataset)
    assert isinstance(direct_restored, MarketDataset)
    assert direct_restored.dataset_id == original.dataset_id
    assert direct_restored.funding_price_rate is None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "market_dataset_artifact_v3"
    assert "funding_price_rate" not in manifest["arrays"]
    assert manifest_path == tmp_path / "manifest.json"
    assert restored.dataset_id == original.dataset_id
    assert restored.symbols == original.symbols
    assert restored.feature_names == original.feature_names
    assert restored.global_feature_names == original.global_feature_names
    for field_name in (
        "timestamps",
        "features",
        "global_features",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "tradable",
        "feature_available",
        "fee_rate",
        "borrow_available",
        "cash_rate",
    ):
        np.testing.assert_array_equal(
            getattr(restored, field_name),
            getattr(original, field_name),
        )


def test_market_dataset_artifact_rejects_tampered_npz(tmp_path: Path) -> None:
    write_market_dataset_files(tmp_path, _dataset())
    arrays_path = tmp_path / "arrays.npz"
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="digest"):
        load_market_dataset_artifact(tmp_path)
    with pytest.raises(ValueError, match="digest"):
        load_published_market_dataset_artifact(tmp_path)


def test_market_dataset_view_rejects_escape_and_materializes_range() -> None:
    dataset = _dataset()
    view = MarketDatasetView(dataset=dataset, start=2, stop=10)

    child = view.subview(3, 8)
    materialized = child.materialize()

    assert child.start == 3
    assert child.stop == 8
    assert materialized.n_bars == 5
    np.testing.assert_array_equal(materialized.close[:, 0], dataset.close[3:8, 0])
    assert materialized.dataset_id != dataset.dataset_id

    with pytest.raises(ValueError, match="outside"):
        view.subview(1, 8)
    with pytest.raises(ValueError, match="outside"):
        view.subview(3, 11)


def test_market_dataset_view_preserves_v7_funding_identity_array() -> None:
    dataset = _dataset_v7()
    view = MarketDatasetView(dataset=dataset, start=2, stop=10)

    materialized = view.materialize().with_content_identity(
        {"source": "synthetic-view-test"}
    )

    assert materialized.identity_verified
    assert '"schema":"market_dataset_identity_v7"' in (
        materialized.identity_payload_json or ""
    )
    np.testing.assert_array_equal(
        materialized.funding_price_rate,
        dataset.funding_price_rate[2:10],
    )


def test_write_market_dataset_files_returns_typed_result(tmp_path: Path) -> None:
    result = write_market_dataset_files(tmp_path, _dataset())

    assert result.manifest_path == tmp_path / "manifest.json"
    assert result.arrays_path == tmp_path / "arrays.npz"
    assert len(result.artifact_digest) == 64


def test_v4_artifact_round_trip_preserves_v7_funding_identity(
    tmp_path: Path,
) -> None:
    original = _dataset_v7()

    artifact = write_market_dataset_files(tmp_path, original)
    restored_artifact, restored = load_published_market_dataset_artifact(tmp_path)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert artifact.schema_version == "market_dataset_artifact_v4"
    assert restored_artifact.schema_version == "market_dataset_artifact_v4"
    assert manifest["schema_version"] == "market_dataset_artifact_v4"
    assert manifest["dataset"]["identity_payload_json"]
    assert (
        '"schema":"market_dataset_identity_v7"'
        in manifest["dataset"]["identity_payload_json"]
    )
    assert restored.dataset_id == original.dataset_id
    assert restored.identity_payload_json == original.identity_payload_json
    assert restored.identity_verified
    np.testing.assert_array_equal(
        restored.funding_price_rate,
        original.funding_price_rate,
    )


def test_v4_artifact_round_trip_preserves_market_build_v3_metadata(
    tmp_path: Path,
) -> None:
    build_config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="momentum", kind=FeatureKind.LOG_RETURN),),
    )
    config_payload = build_config.canonical_payload()
    original = _dataset_v7().with_content_identity(
        {
            "config": config_payload,
            "feature_config_digest": content_digest(config_payload),
            "normalization_digest": "a" * 64,
            "metadata_evidence": {"source": "synthetic-market-build-v3"},
        }
    )

    artifact = write_market_dataset_files(tmp_path, original)
    restored_artifact, restored = load_published_market_dataset_artifact(tmp_path)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    payload = json.loads(restored.identity_payload_json or "{}")

    assert artifact.schema_version == "market_dataset_artifact_v4"
    assert restored_artifact.schema_version == "market_dataset_artifact_v4"
    assert payload["config"]["schema_version"] == "market_build_v3"
    assert (
        payload["config"]["feature_numerics_schema"]
        == (config_payload["feature_numerics_schema"])
    )
    assert payload["config"] == json.loads(canonical_json_bytes(config_payload))
    assert payload["feature_config_digest"] == content_digest(config_payload)
    assert payload["metadata_evidence"] == {"source": "synthetic-market-build-v3"}
    assert manifest["dataset"]["identity_payload_json"] == (
        original.identity_payload_json
    )
    assert restored.dataset_id == original.dataset_id
    assert restored.identity_verified


def test_v4_loader_rejects_missing_and_changed_funding_price_rate(
    tmp_path: Path,
) -> None:
    write_market_dataset_files(tmp_path, _dataset_v7())
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}

    arrays.pop("funding_price_rate")
    _rewrite_artifact_arrays(tmp_path, arrays)
    with pytest.raises(ValueError, match="v4 artifact.*funding_price_rate"):
        load_market_dataset_artifact(tmp_path)

    # Start from a valid v4 artifact again, then preserve its outer digests while
    # changing only the identity-bound event-price funding array.
    for entry in tmp_path.iterdir():
        entry.unlink()
    write_market_dataset_files(tmp_path, _dataset_v7())
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["funding_price_rate"][3, 0] += 0.001
    _rewrite_artifact_arrays(tmp_path, arrays)

    with pytest.raises(ValueError, match="dataset_id"):
        load_market_dataset_artifact(tmp_path)


def test_v4_artifact_cannot_be_downgraded_to_v3(tmp_path: Path) -> None:
    write_market_dataset_files(tmp_path, _dataset_v7())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("artifact_digest")
    manifest["schema_version"] = "market_dataset_artifact_v3"
    manifest["artifact_digest"] = content_digest(manifest)
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="artifact and identity schemas do not match"):
        load_market_dataset_artifact(tmp_path)


def test_legacy_data_artifacts_module_is_removed() -> None:
    root = Path(__file__).resolve().parents[2]
    assert not (root / "trade_rl" / "data" / "artifacts.py").exists()
