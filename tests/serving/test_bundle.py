from __future__ import annotations

from pathlib import Path

import pytest

from tests.serving.helpers import (
    ACTION_NAMES,
    ACTION_SPEC_DIGEST,
    INITIAL_CAPITAL,
    NORMALIZER_DIGEST,
    OBSERVATION_SIZE,
    DEFAULT_NORMALIZER_DIGEST,
    create_bundle,
)
from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import load_serving_bundle


def test_v3_bundle_binds_exact_action_environment_and_normalizer_identity(
    tmp_path: Path,
) -> None:
    bundle = load_serving_bundle(create_bundle(tmp_path / "bundle"))
    manifest = bundle.manifest
    assert manifest.policy_mode is PolicyMode.BASELINE_ONLY
    assert manifest.action_names == ACTION_NAMES
    assert manifest.action_spec_digest == ACTION_SPEC_DIGEST
    assert manifest.observation_size == OBSERVATION_SIZE
    assert manifest.environment_digest == "d" * 64
    assert manifest.normalizer_digest == NORMALIZER_DIGEST
    assert manifest.initial_capital == pytest.approx(INITIAL_CAPITAL)


def test_tampered_bundle_file_is_rejected(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "bundle")
    (root / "signal.json").write_text('{"signal":"tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_serving_bundle(root)


def test_v3_bundle_rejects_missing_or_wrong_action_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="action_names"):
        create_bundle(tmp_path / "bad-names", action_names=("fast_tilt", "fast_tilt"))
    with pytest.raises(ValueError, match="sha256|digest"):
        create_bundle(tmp_path / "bad-digest", action_spec_digest="bad")


def test_bundle_rejects_undeclared_files_and_symlinks(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "extra")
    (root / "unexpected.txt").write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="file closure"):
        load_serving_bundle(root)

    root = create_bundle(tmp_path / "link")
    (root / "unsafe-link").symlink_to(root / "dataset.json")
    with pytest.raises(ValueError, match="file closure|unsafe"):
        load_serving_bundle(root)
