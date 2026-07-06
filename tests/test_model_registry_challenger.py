import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mars_lite.server.model_registry import ModelEntry, ModelRegistry


def test_continuous_rollback_to_limit(tmp_path):
    """
    連続ロールバックの履歴制限（100件）に関する敵対的テスト。
    120個のモデルを登録し、履歴が100件に制限された状態で、
    ロールバックを繰り返して制限の挙動を確認する。
    """
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")

    # 120個のモデルを登録
    for i in range(120):
        registry.register(model_file, version=f"v{i}")

    data = registry._load()
    # 履歴が100件に制限されていることを確認
    assert len(data["history"]) == 100
    # 残っている履歴は v20 から v119 まで
    assert data["history"][0] == "v20"
    assert data["history"][-1] == "v119"

    # 現在アクティブなのは v119
    assert registry.get_active().version == "v119"

    # ロールバックを99回繰り返す（v119 から v20 まで戻る）
    # v119の前の履歴は v118。そこから v20 まで戻れる。
    # 履歴数100件の場合、戻れる回数は 99 回。
    # 履歴：[v20, v21, ..., v119]
    # 1回目のロールバック：v118 に戻る（history の末尾から2番目）
    # 99回目のロールバック：v20 に戻る
    for expected_version_num in range(118, 19, -1):
        rolled = registry.rollback()
        assert rolled.version == f"v{expected_version_num}"
        assert registry.get_active().version == f"v{expected_version_num}"

    # 100回目のロールバックは、履歴が足りなくなる（履歴が['v20']のみになる）ため LookupError
    with pytest.raises(LookupError) as excinfo:
        registry.rollback()
    assert "no previous active model to roll back to" in str(excinfo.value)


def test_extreme_timestamp_conflict(tmp_path, monkeypatch):
    """
    タイムスタンプのミリ秒競合の自動回避に関する敵対的テスト。
    time.time が常に同じ値を返す状態で、100回連続で自動バージョン名登録を行う。
    無限ループやクラッシュが発生せず、正しく model-timestamp+N が作成されるか検証。
    """
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")

    static_time = 123456789.0
    monkeypatch.setattr(time, "time", lambda: static_time)

    base_timestamp = int(static_time * 1000)

    # 100回連続で登録する
    entries = []
    for _ in range(100):
        entry = registry.register(model_file)
        entries.append(entry)

    # 重複がないことと、バージョン名がインクリメントされていることを確認
    versions = [e.version for e in entries]
    assert len(set(versions)) == 100

    for idx, version in enumerate(versions):
        expected_version = f"model-{base_timestamp + idx}"
        assert version == expected_version


def test_path_traversal_vulnerabilities(tmp_path):
    """
    パス・トラバーサル攻撃（..や特殊文字、絶対パスなど）の多様なパターンに対する脆弱性検証。
    バージョン名に ".." や ".._v1" などのトラバーサルパターンを指定したときに、
    それが正しく拒否されるか、あるいは親ディレクトリへの書き込みを許してしまうか検証。
    """
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")

    # 悪意あるバージョン名
    # 正規表現 ^[a-zA-Z0-9_\-\.]+$ にマッチしてしまうが、パス・トラバーサルを引き起こす可能性のあるパターン
    traversal_versions = [
        "..",
        "...",
        "../model",  # これは / があるため正規表現で弾かれる
        "..\\model",  # これは \ があるため正規表現で弾かれる
        "model/../../etc",  # / があるため弾かれる
        "v1..",
        "..v1",
        "model..zip",
    ]

    for v in traversal_versions:
        with pytest.raises(ValueError) as excinfo:
            registry.register(model_file, version=v)
        assert "Invalid version format" in str(excinfo.value)


def test_atomic_save_failures(tmp_path):
    """
    一時ファイルの書き込み失敗時のアトミック性の担保テスト。
    write_text や replace が失敗した際に、元の registry.json が破損・上書きされないことを確認する。
    """
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")

    # 最初のモデルを登録
    registry.register(model_file, version="v1")
    initial_data = registry._load()
    assert initial_data["active_version"] == "v1"

    # 1. write_text が失敗する場合を模倣
    # Path.write_text をモックして OSError を発生させる
    original_write_text = Path.write_text

    def mock_write_text_fail(self_path, *args, **kwargs):
        if self_path.name.endswith(".tmp"):
            raise OSError("Disk full or permission denied during tmp write")
        return original_write_text(self_path, *args, **kwargs)

    with patch.object(Path, "write_text", new=mock_write_text_fail):
        # 登録を試みるが、一時ファイルの書き込みで失敗するはず
        with pytest.raises(OSError) as excinfo:
            registry.register(model_file, version="v2")
        assert "Disk full or permission denied" in str(excinfo.value)

    # 例外発生後、registry.json の内容が元のままであることを確認
    current_data = registry._load()
    assert current_data == initial_data

    # 2. replace が失敗する場合を模倣
    # Path.replace をモックして OSError を発生させる
    original_replace = Path.replace

    def mock_replace_fail(self_path, *args, **kwargs):
        raise OSError("Permission denied during replace")

    with patch.object(Path, "replace", new=mock_replace_fail):
        # 登録を試みるが、置換で失敗するはず
        with pytest.raises(OSError) as excinfo:
            registry.register(model_file, version="v3")
        assert "Permission denied during replace" in str(excinfo.value)

    # 例外発生後、registry.json の内容が元のままであることを確認
    current_data2 = registry._load()
    assert current_data2 == initial_data
