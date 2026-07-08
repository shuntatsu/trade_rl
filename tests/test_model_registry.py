import json
import os
import time
from pathlib import Path

import pytest

from mars_lite.server.model_registry import ModelRegistry, main


def test_model_registry_lists_activates_and_rolls_back(tmp_path):
    first_model = tmp_path / "model_a.zip"
    second_model = tmp_path / "model_b.zip"
    first_model.write_text("a", encoding="utf-8")
    second_model.write_text("b", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")
    first = registry.register(first_model, metrics={"sharpe": 1.0})
    second = registry.register(second_model, metrics={"sharpe": 1.5})

    assert [entry.version for entry in registry.list_models()] == [
        first.version,
        second.version,
    ]
    assert registry.get_active().version == second.version

    registry.activate(first.version)
    assert registry.get_active().version == first.version

    rolled_back = registry.rollback()
    assert rolled_back.version == second.version
    assert Path(rolled_back.model_path).exists()


def test_register_directory_source_like_seed_ensemble(tmp_path):
    """
    SeedEnsemble.save()はディレクトリ(seed_0.zip, seed_1.zip...を内包)を書く。
    run_pipeline STEP5がensemble>1時に`{model_name}.zip`を探して常に見つからず
    登録をスキップしていたバグの根本原因(register側がディレクトリ未対応)を
    直接テストする。
    """
    ensemble_dir = tmp_path / "portfolio_ensemble"
    ensemble_dir.mkdir()
    (ensemble_dir / "seed_0.zip").write_text("seed0", encoding="utf-8")
    (ensemble_dir / "seed_1.zip").write_text("seed1", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")
    entry = registry.register(ensemble_dir, metrics={"sharpe": 1.2}, version="ens-v1")

    target = Path(entry.model_path)
    assert target.is_dir()
    assert (target / "seed_0.zip").read_text(encoding="utf-8") == "seed0"
    assert (target / "seed_1.zip").read_text(encoding="utf-8") == "seed1"
    assert registry.get_active().version == "ens-v1"

    # ロード可能性の確認: SeedEnsemble.loadと同じglobパターンで実ファイルが拾える
    seed_files = sorted(target.glob("seed_*.zip"))
    assert len(seed_files) == 2


def test_model_registry_rejects_unknown_activation(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")

    try:
        registry.activate("missing")
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("unknown versions must be rejected")


def test_path_traversal_prevention(tmp_path):
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")

    # 許可されていない文字を含むバージョン
    invalid_versions = [
        "../invalid",
        "invalid/path",
        "model*",
        "model$1",
        "model\\dir",
        "model-1.0.0",
        "model.v3",
    ]
    for v in invalid_versions:
        with pytest.raises(ValueError) as excinfo:
            registry.register(model_file, version=v)
        assert "Invalid version format" in str(excinfo.value)

    # 許可されている文字のみのバージョン
    valid_versions = ["model_v2", "model1"]
    for v in valid_versions:
        entry = registry.register(model_file, version=v)
        assert entry.version == v


def test_duplicate_version_rejection(tmp_path):
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")
    registry.register(model_file, version="v1")

    # 重複するバージョン登録は ValueError
    with pytest.raises(ValueError) as excinfo:
        registry.register(model_file, version="v1")
    assert "already exists in the registry" in str(excinfo.value)


def test_timestamp_conflict_prevention(tmp_path, monkeypatch):
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")

    # time.time をモックして常に同じタイムスタンプを返すようにする
    static_time = 123456789.0
    monkeypatch.setattr(time, "time", lambda: static_time)

    entry1 = registry.register(model_file)  # model-123456789000
    entry2 = registry.register(model_file)  # model-123456789001

    assert entry1.version == "model-123456789000"
    assert entry2.version == "model-123456789001"
    assert entry1.version != entry2.version


def test_continuous_rollback(tmp_path):
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")
    v1 = registry.register(model_file, version="v1")
    v2 = registry.register(model_file, version="v2")
    v3 = registry.register(model_file, version="v3")

    # 登録直後は v3 がアクティブ
    assert registry.get_active().version == "v3"

    # 1回目のロールバックで v2 に戻る
    rolled_1 = registry.rollback()
    assert rolled_1.version == "v2"
    assert registry.get_active().version == "v2"

    # 2回目のロールバックで v1 に戻る
    rolled_2 = registry.rollback()
    assert rolled_2.version == "v1"
    assert registry.get_active().version == "v1"

    # 3回目はこれ以上戻る履歴がないため LookupError
    with pytest.raises(LookupError):
        registry.rollback()


def test_history_capping(tmp_path):
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")

    # 105個モデルを登録する
    for i in range(105):
        registry.register(model_file, version=f"v{i}")

    data = registry._load()
    # 履歴が100件に制限されているか
    assert len(data["history"]) == 100
    # 最新の100件が残っているか
    assert data["history"][0] == "v5"
    assert data["history"][-1] == "v104"


def test_physical_file_missing_validation(tmp_path):
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")
    entry = registry.register(model_file, version="v1")

    # 物理ファイルを削除
    os.remove(entry.model_path)

    # _entry_for を通じて FileNotFoundError がスローされることを確認
    with pytest.raises(FileNotFoundError) as excinfo:
        registry.get_active()
    assert "Model physical file does not exist" in str(excinfo.value)

    with pytest.raises(FileNotFoundError):
        registry.activate("v1")


def test_cli_register_with_metrics_and_error_handling(tmp_path, capsys):
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry_dir = str(tmp_path / "registry")

    # 1. 正常登録（--metrics のパースを含む）
    argv = [
        "--registry-dir",
        registry_dir,
        "register",
        str(model_file),
        "--version",
        "cli-v1",
        "--metrics",
        '{"sharpe": 2.1, "drawdown": 0.05}',
    ]
    ret = main(argv)
    assert ret == 0

    out, err = capsys.readouterr()
    registered_data = json.loads(out.strip())
    assert registered_data["version"] == "cli-v1"
    assert registered_data["metrics"] == {"sharpe": 2.1, "drawdown": 0.05}

    # 2. パス・トラバーサルなどによる ValueError ハンドリング（終了ステータス1、エラー出力）
    argv_err_traversal = [
        "--registry-dir",
        registry_dir,
        "register",
        str(model_file),
        "--version",
        "../invalid-v2",
    ]
    ret = main(argv_err_traversal)
    assert ret == 1
    out, err = capsys.readouterr()
    assert "Error: Invalid version format" in err
    assert not out

    # 3. 重複バージョン登録エラーハンドリング
    argv_err_dup = [
        "--registry-dir",
        registry_dir,
        "register",
        str(model_file),
        "--version",
        "cli-v1",
    ]
    ret = main(argv_err_dup)
    assert ret == 1
    out, err = capsys.readouterr()
    assert "already exists in the registry" in err

    # 4. ファイル不在エラーハンドリング (FileNotFoundError)
    argv_err_fnf = [
        "--registry-dir",
        registry_dir,
        "register",
        "nonexistent_file.zip",
        "--version",
        "cli-v2",
    ]
    ret = main(argv_err_fnf)
    assert ret == 1
    out, err = capsys.readouterr()
    assert "Error:" in err or "FileNotFoundError" in err


def test_physical_file_missing_prevents_active_update(tmp_path):
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry = ModelRegistry(tmp_path / "registry")
    entry_v1 = registry.register(model_file, version="v1")
    entry_v2 = registry.register(model_file, version="v2")

    # 登録直後は v2 がアクティブ
    assert registry.get_active().version == "v2"

    # v1 の物理ファイルを削除
    os.remove(entry_v1.model_path)

    # activate(v1) を試みると FileNotFoundError
    with pytest.raises(FileNotFoundError):
        registry.activate("v1")

    # 保存内容が更新されていないことを検証
    # （アクティブは v2 のままで、v1 に書き換わっていない）
    data = registry._load()
    assert data["active_version"] == "v2"

    # ロールバックも試みる
    # v2 がアクティブ、v1 にロールバックしようとするが、v1 が存在しないため FileNotFoundError
    with pytest.raises(FileNotFoundError):
        registry.rollback()

    # 履歴やアクティブバージョンが更新されていないことを検証
    data2 = registry._load()
    assert data2["active_version"] == "v2"
    assert data2["history"][-1] == "v2"


def test_cli_register_invalid_metrics_error(tmp_path, capsys):
    model_file = tmp_path / "model.zip"
    model_file.write_text("dummy", encoding="utf-8")

    registry_dir = str(tmp_path / "registry")

    # metrics が不正な JSON の場合
    argv_invalid_json = [
        "--registry-dir",
        registry_dir,
        "register",
        str(model_file),
        "--version",
        "cli-v2",
        "--metrics",
        "invalid-json",
    ]
    ret = main(argv_invalid_json)
    assert ret == 1
    out, err = capsys.readouterr()
    assert "Error: Invalid JSON for metrics" in err

    # metrics が JSON だが辞書型（オブジェクト）ではない場合
    argv_not_dict = [
        "--registry-dir",
        registry_dir,
        "register",
        str(model_file),
        "--version",
        "cli-v3",
        "--metrics",
        "[1, 2, 3]",
    ]
    ret = main(argv_not_dict)
    assert ret == 1
    out, err = capsys.readouterr()
    assert "Error: metrics must be a JSON object" in err
