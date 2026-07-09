import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mars_lite.server.model_registry import ModelEntry, ModelRegistry


def test_concurrent_registration_race_condition(tmp_path):
    """
    スレッド間での同時登録時に競合状態（Race Condition）が発生し、
    一方の登録内容がロスト（Lost Update）することを実証するテスト。
    """
    registry_dir = tmp_path / "registry"
    registry = ModelRegistry(registry_dir)

    model_file_1 = tmp_path / "model_1.zip"
    model_file_2 = tmp_path / "model_2.zip"
    model_file_1.write_text("model_1_content", encoding="utf-8")
    model_file_2.write_text("model_2_content", encoding="utf-8")

    original_load = registry._load
    barrier = threading.Barrier(2)

    # 決定論的に競合を発生させるため、_load で同期を取る
    def mocked_load():
        data = original_load()
        try:
            barrier.wait(timeout=5.0)
        except threading.BrokenBarrierError:
            pass
        return data

    with patch.object(ModelRegistry, "_load", side_effect=mocked_load):
        errors = []

        def run_t1():
            try:
                registry.register(model_file_1)
            except Exception as e:
                errors.append(e)

        def run_t2():
            try:
                registry.register(model_file_2)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run_t1)
        t2 = threading.Thread(target=run_t2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # 競合が防がれ、両方のモデルが正常に登録されるため、登録モデル数が 2 つになっていることを確認
        models = registry.list_models()
        assert len(models) == 2, "Race condition was not prevented"


def test_serialize_error_leaves_garbage_files(tmp_path):
    """
    シリアライズエラー（JSONに変換できないメタデータ）が発生した際、
    インデックスの更新には失敗するが、コピーされた物理ファイルが
    モデルディレクトリ内に残ったまま（浮いた状態）になることを実証するテスト。
    """
    registry_dir = tmp_path / "registry"
    registry = ModelRegistry(registry_dir)

    model_file = tmp_path / "model.zip"
    model_file.write_text("content", encoding="utf-8")

    class Unserializable:
        pass

    # Python API からシリアライズ不能なオブジェクトを metrics に指定して登録
    # metrics の型チェックまたはシリアライズで例外が投げられる
    with pytest.raises((TypeError, ValueError)):
        registry.register(
            model_file,
            metrics={"invalid": Unserializable()},
            version="garbage_test_version",
        )

    # インデックスには登録されていないことを確認
    models = registry.list_models()
    assert not any(m.version == "garbage_test_version" for m in models)

    # 物理ファイルがクリーンアップされ、存在しないことを確認
    expected_copied_file = registry.models_dir / "garbage_test_version.zip"
    assert not expected_copied_file.exists(), "Garbage file was not cleaned up"


def test_disk_full_during_copy_leaves_partial_files(tmp_path):
    """
    ファイルコピー中にディスクフル（OSError(28)）が発生した際、
    コピー途中のファイルが残ったままになることを実証するテスト。
    """
    registry_dir = tmp_path / "registry"
    registry = ModelRegistry(registry_dir)

    model_file = tmp_path / "model.zip"
    model_file.write_text("content", encoding="utf-8")

    def mock_copy2(src, dst):
        # 途中で書き込みが失敗したように見せかけるため、空ファイルを生成してエラーを投げる
        Path(dst).write_text("partial content", encoding="utf-8")
        raise OSError(28, "No space left on device")

    with patch("shutil.copy2", side_effect=mock_copy2):
        with pytest.raises(OSError) as excinfo:
            registry.register(model_file, version="disk_full_version")
        assert "No space left on device" in str(excinfo.value)

    # インデックスには未登録
    models = registry.list_models()
    assert not any(m.version == "disk_full_version" for m in models)

    # コピー途中の物理ファイルがクリーンアップされて存在しないことを確認
    target_file = registry.models_dir / "disk_full_version.zip"
    assert not target_file.exists(), "Partial file was not cleaned up"


def test_index_file_write_permission_failure(tmp_path, monkeypatch):
    """
    インデックス（registry.json）の書き込みが失敗した場合、register は例外を
    伝播し、コピー済みの物理ファイルをクリーンアップして中途半端な状態を
    残さないことを検証する。

    注: 以前は index_path を chmod で read-only にしていたが、_save は
    アトミックな temp→replace で書くため rename はディレクトリ権限のみに依存し、
    ファイルを read-only にしても書き込みは失敗しない（かつ root 実行では chmod
    自体が無視される）。このためテストの前提が誤りで CI で常に失敗していた。
    ここでは _save の失敗を直接注入し「失敗時のクリーンアップ契約」を
    環境非依存に検証する。
    """
    registry_dir = tmp_path / "registry"
    registry = ModelRegistry(registry_dir)

    model_file = tmp_path / "model.zip"
    model_file.write_text("content", encoding="utf-8")

    # 初期状態として1つ登録しておく
    registry.register(model_file, version="v1")

    # インデックス書き込みだけを失敗させる（モデルのコピーは成功する）
    def _boom(_data):
        raise OSError("simulated index write failure")

    monkeypatch.setattr(registry, "_save", _boom)

    # register は _save の OSError を伝播するはず
    with pytest.raises((PermissionError, OSError)):
        registry.register(model_file, version="v2")

    # 実インデックスを読むためパッチを戻す（_save は書けていないので v1 のみ）
    monkeypatch.undo()

    models = registry.list_models()
    assert not any(m.version == "v2" for m in models)

    # コピー済みの物理ファイルはクリーンアップされ、存在しない
    target_file = registry.models_dir / "v2.zip"
    assert not target_file.exists(), "File was not cleaned up"


def test_invalid_metadata_type_persistence(tmp_path):
    """
    Python API経由で dict[str, float] の型に反する metrics を渡した際、
    型チェックが行われずにそのまま保存され、ロード時にも不正な値が返ることを検証する。
    """
    registry_dir = tmp_path / "registry"
    registry = ModelRegistry(registry_dir)

    model_file = tmp_path / "model.zip"
    model_file.write_text("content", encoding="utf-8")

    # 期待される型は dict[str, float] だが、リストや文字列を渡す
    invalid_metrics = {
        "sharpe": "not-a-float",
        "drawdown": [1.0, 2.0],
        "nested": {"key": "val"},
    }

    # 登録は metrics の型チェックにより ValueError がスローされる
    with pytest.raises(ValueError) as excinfo:
        registry.register(model_file, metrics=invalid_metrics, version="v1")
    assert "metrics" in str(excinfo.value)

    # インデックスに登録されていないことを検証
    models = registry.list_models()
    assert not any(m.version == "v1" for m in models)
