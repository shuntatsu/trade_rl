import json
from pathlib import Path

import pytest

from mars_lite.serving.bundle import build_manifest, load_bundle


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "model.zip").write_bytes(b"model-v1")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": "v1",
                "git_sha": "a" * 40,
                "model_kind": "single",
                "symbols": ["BTCUSDT"],
                "observation_schema_version": 1,
                "observation_progress_mode": "zero",
                "observation_dim": 5,
                "run_config": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "preprocessing.json").write_text(
        '{"feature_names":["ret"],"global_feature_names":[],"feature_norm":"none",'
        '"feature_mask":[true],"post_mask_dim":1}',
        encoding="utf-8",
    )
    (root / "risk.json").write_text(
        '{"guardrails":{},"pre_trade":{}}', encoding="utf-8"
    )
    return root


def _rewrite_metadata(root: Path, **changes) -> None:
    path = root / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(changes)
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_bundle_digest_is_deterministic_and_loadable(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    first = build_manifest(root)
    second = build_manifest(root)
    assert first.bundle_digest == second.bundle_digest
    loaded = load_bundle(root)
    assert loaded.version == "v1"
    assert loaded.bundle_digest == first.bundle_digest
    assert loaded.model_path == root / "model.zip"


def test_tampered_file_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    build_manifest(root)
    (root / "model.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_bundle(root)


def test_feature_mask_dimension_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "preprocessing.json").write_text(
        '{"feature_names":["a","b"],"global_feature_names":[],"feature_norm":"none",'
        '"feature_mask":[true,false],"post_mask_dim":1}',
        encoding="utf-8",
    )
    build_manifest(root)
    with pytest.raises(ValueError, match="post_mask_dim"):
        load_bundle(root)


def test_zero_mask_preserves_feature_dimension(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "preprocessing.json").write_text(
        '{"feature_names":["a","b"],"global_feature_names":[],"feature_norm":"none",'
        '"feature_mask":[true,false],"post_mask_dim":2}',
        encoding="utf-8",
    )
    _rewrite_metadata(root, observation_dim=6)
    build_manifest(root)
    loaded = load_bundle(root)
    assert loaded.preprocessing["post_mask_dim"] == 2


def test_episode_progress_bundle_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    _rewrite_metadata(root, observation_progress_mode="episode")
    build_manifest(root)
    with pytest.raises(ValueError, match="observation_progress_mode"):
        load_bundle(root)


def test_observation_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    _rewrite_metadata(root, observation_dim=6)
    build_manifest(root)
    with pytest.raises(ValueError, match="observation_dim"):
        load_bundle(root)


def test_missing_or_mismatched_model_kind_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    _rewrite_metadata(root, model_kind="ensemble")
    build_manifest(root)
    with pytest.raises(ValueError, match="ensemble model_kind"):
        load_bundle(root)


def test_ensemble_requires_only_seed_zip_members(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "model.zip").unlink()
    ensemble = root / "ensemble"
    ensemble.mkdir()
    (ensemble / "seed_0.zip").write_bytes(b"seed")
    (ensemble / "notes.txt").write_text("unexpected", encoding="utf-8")
    _rewrite_metadata(root, model_kind="ensemble")
    build_manifest(root)
    with pytest.raises(ValueError, match="ensemble model_kind"):
        load_bundle(root)


def test_invalid_git_sha_is_rejected_before_manifest_build(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    _rewrite_metadata(root, git_sha="abc123")
    with pytest.raises(ValueError, match="git_sha"):
        build_manifest(root)
