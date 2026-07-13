from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.serving.helpers import create_bundle
from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import ServingBundleManifest, load_serving_bundle


def test_baseline_bundle_validates_all_file_digests(tmp_path: Path) -> None:
    bundle = load_serving_bundle(create_bundle(tmp_path / "bundle"))

    assert bundle.manifest.policy_mode is PolicyMode.BASELINE_ONLY
    assert bundle.manifest.policy_digest is None
    assert bundle.manifest.release_digest is None
    assert bundle.manifest.observation_schema_digest == "d" * 64
    assert bundle.manifest.observation_size == 5
    assert bundle.root == tmp_path / "bundle"


def test_tampered_bundle_file_is_rejected(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "bundle")
    (root / "signal.json").write_text('{"signal":"tampered"}', encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_serving_bundle(root)


def test_baseline_bundle_rejects_policy_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="baseline_only"):
        ServingBundleManifest(
            bundle_digest="d" * 64,
            dataset_id="a" * 64,
            action_schema="baseline_residual_v1",
            observation_schema_digest="f" * 64,
            observation_size=5,
            policy_mode=PolicyMode.BASELINE_ONLY,
            policy_digest="e" * 64,
            signal_digest="b" * 64,
            selection_digest="c" * 64,
            release_digest=None,
            files=(),
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        )
