from pathlib import Path

import pytest

from mars_lite.serving.bundle import build_manifest, load_bundle


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "model.zip").write_bytes(b"model-v1")
    (root / "metadata.json").write_text(
        '{"schema_version":1,"model_version":"v1","git_sha":"abc123",'
        '"symbols":["BTCUSDT"],"observation_schema_version":1}',
        encoding="utf-8",
    )
    (root / "preprocessing.json").write_text(
        '{"feature_names":["ret"],"feature_norm":"none",'
        '"feature_mask":[true],"post_mask_dim":1}',
        encoding="utf-8",
    )
    (root / "risk.json").write_text(
        '{"guardrails":{},"pre_trade":{}}', encoding="utf-8"
    )
    return root


def test_bundle_digest_is_deterministic_and_loadable(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    first = build_manifest(root)
    second = build_manifest(root)
    assert first.bundle_digest == second.bundle_digest
    loaded = load_bundle(root)
    assert loaded.version == "v1"
    assert loaded.bundle_digest == first.bundle_digest


def test_tampered_file_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    build_manifest(root)
    (root / "model.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_bundle(root)


def test_feature_mask_dimension_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "preprocessing.json").write_text(
        '{"feature_names":["a","b"],"feature_norm":"none",'
        '"feature_mask":[true,false],"post_mask_dim":2}',
        encoding="utf-8",
    )
    build_manifest(root)
    with pytest.raises(ValueError, match="post_mask_dim"):
        load_bundle(root)
