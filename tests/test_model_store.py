"""mars_lite.serving.model_store のテスト"""

import numpy as np
import pytest

from mars_lite.serving.model_store import (
    ModelMetadata, save_bundle, load_metadata, model_exists, list_models,
    promote, get_current, rollback,
)


class _FakeAgent:
    """agent.save(path) だけを使う保存経路をテストするためのダミー"""
    def __init__(self, tag: str):
        self.tag = tag

    def save(self, path):
        with open(str(path) + ".zip", "w") as f:
            f.write(self.tag)


def test_save_and_load_metadata_roundtrip(tmp_path):
    meta = ModelMetadata(
        symbols=["BTCUSDT", "ETHUSDT"],
        post_processor={"ema_alpha": 0.5},
        feature_mask=[True, False, True],
        metrics={"total_return": 0.5},
    )
    save_bundle(tmp_path, "model_a", _FakeAgent("a"), meta)

    assert model_exists(tmp_path, "model_a")
    loaded = load_metadata(tmp_path, "model_a")
    assert loaded.symbols == ["BTCUSDT", "ETHUSDT"]
    assert loaded.post_processor == {"ema_alpha": 0.5}
    assert loaded.feature_mask == [True, False, True]
    assert loaded.metrics == {"total_return": 0.5}


def test_load_metadata_missing_returns_none(tmp_path):
    assert load_metadata(tmp_path, "nonexistent") is None


def test_list_models(tmp_path):
    save_bundle(tmp_path, "model_a", _FakeAgent("a"), ModelMetadata(symbols=["BTCUSDT"]))
    save_bundle(tmp_path, "model_b", _FakeAgent("b"), ModelMetadata(symbols=["ETHUSDT"]))
    assert list_models(tmp_path) == ["model_a", "model_b"]


def test_promote_and_rollback(tmp_path):
    save_bundle(tmp_path, "model_a", _FakeAgent("a"), ModelMetadata(symbols=["BTCUSDT"]))
    save_bundle(tmp_path, "model_b", _FakeAgent("b"), ModelMetadata(symbols=["ETHUSDT"]))

    assert get_current(tmp_path) is None
    promote(tmp_path, "model_a")
    assert get_current(tmp_path) == "model_a"

    promote(tmp_path, "model_b")
    assert get_current(tmp_path) == "model_b"

    prev = rollback(tmp_path)
    assert prev == "model_a"
    assert get_current(tmp_path) == "model_a"

    # 履歴が尽きたらNone
    assert rollback(tmp_path) is None


def test_promote_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote(tmp_path, "does_not_exist")
