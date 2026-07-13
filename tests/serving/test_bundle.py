from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.serving.helpers import INITIAL_CAPITAL, OBSERVATION_SIZE, create_bundle
from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import ServingBundleManifest, load_serving_bundle


def test_baseline_bundle_validates_complete_environment_identity(tmp_path: Path) -> None:
    bundle = load_serving_bundle(create_bundle(tmp_path / "bundle"))

    assert bundle.manifest.policy_mode is PolicyMode.BASELINE_ONLY
    assert bundle.manifest.policy_digest is None
    assert bundle.manifest.release_digest == "f" * 64
    assert bundle.manifest.observation_schema == "baseline_residual_observation_v2"
    assert bundle.manifest.observation_size == OBSERVATION_SIZE
    assert bundle.manifest.environment_digest == "d" * 64
    assert bundle.manifest.initial_capital == pytest.approx(INITIAL_CAPITAL)
    assert bundle.root == tmp_path / "bundle"


def test_tampered_bundle_file_is_rejected(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "bundle")
    (root / "signal.json").write_text('{"signal":"tampered"}', encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_serving_bundle(root)


def test_baseline_bundle_rejects_policy_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="baseline_only"):
        ServingBundleManifest(
            bundle_digest="9" * 64,
            dataset_id="a" * 64,
            action_schema="baseline_residual_v1",
            observation_schema="baseline_residual_observation_v2",
            observation_size=OBSERVATION_SIZE,
            environment_digest="d" * 64,
            initial_capital=INITIAL_CAPITAL,
            policy_mode=PolicyMode.BASELINE_ONLY,
            policy_digest="e" * 64,
            signal_digest="b" * 64,
            selection_digest="c" * 64,
            release_digest="f" * 64,
            files=(),
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        )


def test_bundle_rejects_invalid_observation_size_or_aum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="observation_size"):
        create_bundle(tmp_path / "bad-size", observation_size=0)

    root = tmp_path / "bad-aum"
    root.mkdir()
    (root / "dataset.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="initial_capital"):
        ServingBundleManifest.build(
            root=root,
            dataset_id="a" * 64,
            action_schema="baseline_residual_v1",
            observation_schema="baseline_residual_observation_v2",
            observation_size=OBSERVATION_SIZE,
            environment_digest="d" * 64,
            initial_capital=0.0,
            policy_mode=PolicyMode.BASELINE_ONLY,
            policy_digest=None,
            signal_digest="b" * 64,
            selection_digest="c" * 64,
            release_digest="f" * 64,
            artifact_paths=("dataset.json",),
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        )
