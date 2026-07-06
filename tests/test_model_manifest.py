import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mars_lite.learning.manifest import (
    calculate_data_hash,
    generate_and_save_manifest,
    get_git_commit_hash,
    verify_reproducible_sharpe,
)


# ダミー FeatureSet
class DummyFeatureSet:
    def __init__(self, close, features, global_features):
        self.close = close
        self.features = features
        self.global_features = global_features


def test_git_commit_hash_success():
    with patch("subprocess.check_output") as mock_check:
        mock_check.return_value = b"abcdef1234567890\n"
        git_hash = get_git_commit_hash()
        assert git_hash == "abcdef1234567890"


def test_git_commit_hash_failure():
    with patch(
        "subprocess.check_output", side_effect=Exception("Git command not found")
    ):
        git_hash = get_git_commit_hash()
        assert git_hash == "unknown"


def test_data_hash_consistency():
    close = np.array([[1.0, 2.0], [3.0, 4.0]])
    features = np.array([[[0.1, 0.2]], [[0.3, 0.4]]])
    global_features = np.array([[0.5], [0.6]])

    fs1 = DummyFeatureSet(close.copy(), features.copy(), global_features.copy())
    fs2 = DummyFeatureSet(close.copy(), features.copy(), global_features.copy())

    hash1 = calculate_data_hash(fs1)
    hash2 = calculate_data_hash(fs2)

    # 同一データならハッシュ値も同一
    assert hash1 == hash2
    assert len(hash1) == 64

    # 異なるデータならハッシュ値が異なる
    fs3 = DummyFeatureSet(close * 2.0, features, global_features)
    hash3 = calculate_data_hash(fs3)
    assert hash1 != hash3


def test_generate_and_save_manifest(tmp_path):
    close = np.array([[1.0, 2.0]])
    features = np.array([[[0.1]]])
    global_features = np.array([[0.5]])
    fs = DummyFeatureSet(close, features, global_features)

    output_file = tmp_path / "model_manifest.json"
    hyperparams = {"learning_rate": 0.001, "batch_size": 32}
    seed = 42

    manifest = generate_and_save_manifest(
        output_filepath=str(output_file),
        fs=fs,
        hyperparams=hyperparams,
        seed=seed,
        additional_metadata={"test_run": True},
    )

    # 返り値の検証
    assert manifest["seed"] == seed
    assert manifest["hyperparameters"] == hyperparams
    assert manifest["created_at"].endswith("Z")
    assert manifest["test_run"] is True
    assert "git_commit" in manifest
    assert "data_hash" in manifest

    # ファイル書き込みの検証
    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == manifest


def test_verify_reproducible_sharpe_within_tolerance():
    baseline_manifest = {
        "git_commit": "abc",
        "data_hash": "data",
        "hyperparameters": {"learning_rate": 0.001},
        "seed": 42,
        "oos_sharpe": 1.2,
    }
    replay_manifest = dict(baseline_manifest, oos_sharpe=1.26)

    report = verify_reproducible_sharpe(
        baseline_manifest,
        replay_manifest,
        tolerance=0.1,
    )

    assert report["reproducible"] is True
    assert report["sharpe_diff"] == pytest.approx(0.06)


def test_verify_reproducible_sharpe_rejects_different_inputs():
    baseline_manifest = {
        "git_commit": "abc",
        "data_hash": "data",
        "hyperparameters": {"learning_rate": 0.001},
        "seed": 42,
        "oos_sharpe": 1.2,
    }
    replay_manifest = dict(baseline_manifest, data_hash="other", oos_sharpe=1.21)

    report = verify_reproducible_sharpe(baseline_manifest, replay_manifest)

    assert report["reproducible"] is False
    assert "data_hash" in report["mismatched_fields"]
