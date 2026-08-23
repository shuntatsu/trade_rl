from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.v4_context import V4ContextBlock, V4TargetContext
from trade_rl.data.v4_context_artifact import (
    load_v4_target_context_artifact,
    write_v4_target_context_artifact,
)


def _digest(char: str) -> str:
    return char * 64


def _context(*, beta_shift: float = 0.0) -> V4TargetContext:
    decision_indices = np.asarray([100, 101, 102], dtype=np.int64)
    local = V4ContextBlock(
        feature_names=("local_a", "local_b"),
        decision_indices=decision_indices,
        values=np.asarray([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]], dtype=np.float64),
        available=np.ones((3, 2), dtype=np.bool_),
        staleness_hours=np.zeros((3, 2), dtype=np.float64),
        source_digest=_digest("1"),
    )
    global_market = V4ContextBlock(
        feature_names=("global_a",),
        decision_indices=decision_indices,
        values=np.asarray([[0.1], [0.2], [0.3]], dtype=np.float64),
        available=np.ones((3, 1), dtype=np.bool_),
        staleness_hours=np.zeros((3, 1), dtype=np.float64),
        source_digest=_digest("2"),
    )
    return V4TargetContext(
        symbol="ETHUSDT",
        local=local,
        global_market=global_market,
        beta=np.asarray(
            [1.10 + beta_shift, 1.20 + beta_shift, 1.30 + beta_shift],
            dtype=np.float64,
        ),
        beta_available=np.ones(3, dtype=np.bool_),
        beta_source_digest=_digest("3"),
        profile_name="cross_market_core_v1+global_market_core_v1",
    )


def test_v4_context_artifact_round_trip(tmp_path: Path) -> None:
    context = _context()
    root = tmp_path / "ETHUSDT"
    written = write_v4_target_context_artifact(root, context)
    assert written == root
    assert (root / "manifest.json").is_file()
    assert (root / "arrays.npz").is_file()

    loaded = load_v4_target_context_artifact(root)
    assert loaded.digest == context.digest
    assert loaded.beta_source_digest == context.beta_source_digest
    assert loaded.local.feature_names == context.local.feature_names
    assert loaded.global_market.feature_names == context.global_market.feature_names
    np.testing.assert_array_equal(loaded.local.values, context.local.values)
    np.testing.assert_array_equal(loaded.local.available, context.local.available)
    np.testing.assert_array_equal(
        loaded.local.staleness_hours, context.local.staleness_hours
    )
    np.testing.assert_array_equal(
        loaded.global_market.values, context.global_market.values
    )
    np.testing.assert_array_equal(loaded.beta, context.beta)
    np.testing.assert_array_equal(loaded.beta_available, context.beta_available)
    assert not loaded.beta.flags.writeable
    assert not loaded.local.values.flags.writeable


def test_v4_context_artifact_manifest_binds_exact_schema(tmp_path: Path) -> None:
    root = tmp_path / "ETHUSDT"
    context = _context()
    write_v4_target_context_artifact(root, context)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "causal_alpha_v4_target_context_artifact_v1"
    assert manifest["symbol"] == "ETHUSDT"
    assert manifest["context_digest"] == context.digest
    assert manifest["beta_source_digest"] == context.beta_source_digest
    assert tuple(manifest["local_feature_names"]) == context.local.feature_names
    assert (
        tuple(manifest["global_feature_names"]) == context.global_market.feature_names
    )
    assert manifest["row_count"] == 3
    assert manifest["first_decision_index"] == 100
    assert manifest["last_decision_index"] == 102


def test_v4_context_artifact_write_is_idempotent_for_identical_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ETHUSDT"
    context = _context()
    first = write_v4_target_context_artifact(root, context)
    manifest_before = (root / "manifest.json").read_bytes()
    arrays_before = (root / "arrays.npz").read_bytes()
    second = write_v4_target_context_artifact(root, context)
    assert first == second == root
    assert (root / "manifest.json").read_bytes() == manifest_before
    assert (root / "arrays.npz").read_bytes() == arrays_before


def test_v4_context_artifact_rejects_different_content_at_existing_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ETHUSDT"
    write_v4_target_context_artifact(root, _context())
    with pytest.raises(FileExistsError, match="different content"):
        write_v4_target_context_artifact(root, _context(beta_shift=0.25))


def test_v4_context_artifact_rejects_tampered_beta(tmp_path: Path) -> None:
    root = tmp_path / "ETHUSDT"
    write_v4_target_context_artifact(root, _context())
    arrays_path = root / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays["beta"][1] += 0.125
    np.savez(arrays_path, **arrays)

    with pytest.raises(ValueError, match="digest|identity|array"):
        load_v4_target_context_artifact(root)


def test_v4_context_artifact_rejects_tampered_feature_order(tmp_path: Path) -> None:
    root = tmp_path / "ETHUSDT"
    write_v4_target_context_artifact(root, _context())
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["local_feature_names"] = list(reversed(manifest["local_feature_names"]))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="digest|identity|feature"):
        load_v4_target_context_artifact(root)


def test_v4_context_artifact_rejects_missing_or_extra_payload(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        load_v4_target_context_artifact(missing)

    root = tmp_path / "ETHUSDT"
    write_v4_target_context_artifact(root, _context())
    arrays_path = root / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays["unexpected"] = np.asarray([1.0], dtype=np.float64)
    np.savez(arrays_path, **arrays)
    with pytest.raises(ValueError, match="array|member|unexpected|digest"):
        load_v4_target_context_artifact(root)
