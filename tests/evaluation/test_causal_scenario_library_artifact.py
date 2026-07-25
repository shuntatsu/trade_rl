from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.strategies.trend import TrendConfig, TrendStrategy
from trade_rl.workflows.causal_scenario.library import (
    CausalScenarioLibraryConfig,
    build_causal_scenario_library,
)
from trade_rl.workflows.causal_scenario.library_artifact import (
    load_causal_scenario_library_artifact,
    write_causal_scenario_library_artifact,
)


def _library(market_dataset_factory: Any) -> Any:
    dataset = market_dataset_factory(n_bars=820)
    return build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 780),
        TrendStrategy(TrendConfig(fast_hours=12, base_hours=48, slow_hours=96)),
        CausalScenarioLibraryConfig(horizon_decisions=8),
    )


def test_library_artifact_is_deterministic_and_round_trips(
    tmp_path: Path,
    market_dataset_factory: Any,
) -> None:
    library = _library(market_dataset_factory)
    first = tmp_path / "first"
    second = tmp_path / "second"
    digest_a = write_causal_scenario_library_artifact(first, library)
    digest_b = write_causal_scenario_library_artifact(second, library)
    assert digest_a == digest_b
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    assert (first / "arrays.npz").read_bytes() == (second / "arrays.npz").read_bytes()

    loaded = load_causal_scenario_library_artifact(first)
    assert loaded.library_digest == library.library_digest
    np.testing.assert_array_equal(loaded.anchor_indices, library.anchor_indices)
    np.testing.assert_array_equal(loaded.raw_conditions, library.raw_conditions)
    np.testing.assert_array_equal(
        loaded.normalized_conditions, library.normalized_conditions
    )
    np.testing.assert_array_equal(loaded.normalizer.median, library.normalizer.median)
    assert loaded.anchor_indices.flags.writeable is False
    assert loaded.raw_conditions.flags.writeable is False
    assert loaded.normalized_conditions.flags.writeable is False


def test_library_artifact_rejects_file_closure_and_symlinks(
    tmp_path: Path,
    market_dataset_factory: Any,
) -> None:
    root = tmp_path / "artifact"
    write_causal_scenario_library_artifact(root, _library(market_dataset_factory))
    (root / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="closure"):
        load_causal_scenario_library_artifact(root)
    (root / "extra.txt").unlink()
    (root / "manifest-link.json").symlink_to(root / "manifest.json")
    with pytest.raises(ValueError, match="invalid file"):
        load_causal_scenario_library_artifact(root)


def test_library_artifact_rejects_manifest_and_array_tampering(
    tmp_path: Path,
    market_dataset_factory: Any,
) -> None:
    root = tmp_path / "artifact"
    write_causal_scenario_library_artifact(root, _library(market_dataset_factory))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_stop"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest digest"):
        load_causal_scenario_library_artifact(root)

    root = tmp_path / "artifact-arrays"
    write_causal_scenario_library_artifact(root, _library(market_dataset_factory))
    payload = bytearray((root / "arrays.npz").read_bytes())
    payload[-10] ^= 1
    (root / "arrays.npz").write_bytes(payload)
    with pytest.raises(ValueError, match="arrays digest"):
        load_causal_scenario_library_artifact(root)


def test_library_artifact_rejects_array_metadata_tampering(
    tmp_path: Path,
    market_dataset_factory: Any,
) -> None:
    root = tmp_path / "artifact"
    write_causal_scenario_library_artifact(root, _library(market_dataset_factory))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["array_metadata"]["anchor_indices"]["shape"] = [999]
    base = dict(manifest)
    base.pop("artifact_digest")
    from trade_rl.artifacts.hashing import content_digest

    manifest["artifact_digest"] = content_digest(base)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="shape"):
        load_causal_scenario_library_artifact(root)


def test_library_artifact_does_not_duplicate_overlapping_future_windows(
    tmp_path: Path,
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_100)
    library = build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 960),
        TrendStrategy(TrendConfig(fast_hours=12, base_hours=48, slow_hours=96)),
    )
    root = tmp_path / "compact-library"
    write_causal_scenario_library_artifact(root, library)
    assert sum(path.stat().st_size for path in root.iterdir()) < 1_000_000
