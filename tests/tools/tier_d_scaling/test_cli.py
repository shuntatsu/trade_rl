"""Ordinary invocation always uses a bounded owned worker and artificial roots."""

import importlib
from pathlib import Path

import pytest


def test_prepare_uses_supervised_worker_without_loading_inputs(
    tmp_path: Path, monkeypatch
):
    module = importlib.import_module("tools.tier_d_scaling.__main__")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "validate_profile_source", lambda p: None)
    seen = []
    monkeypatch.setattr(
        module, "supervised", lambda command, root: seen.append((command, root)) or 0
    )
    assert module.main(["prepare", "--root", str(tmp_path / "run")]) == 0
    command, root = seen[0]
    assert root == tmp_path / "run"
    assert "--worker" in command and "prepare" in command
    assert str(tmp_path / "run/attempt") in command
    assert not root.exists()


def test_measure_rejects_foreign_case_without_launching(tmp_path: Path, monkeypatch):
    module = importlib.import_module("tools.tier_d_scaling.__main__")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "validate_profile_source", lambda p: None)
    with pytest.raises(SystemExit):
        module.main(
            [
                "measure",
                "--root",
                str(tmp_path / "run"),
                "--prepared-root",
                str(tmp_path / "prepared"),
                "--case",
                "n25",
            ]
        )
    assert not (tmp_path / "run").exists()


def test_root_outside_assigned_output_scope_is_rejected(tmp_path: Path, monkeypatch):
    module = importlib.import_module("tools.tier_d_scaling.__main__")
    monkeypatch.setattr(module, "ROOT", tmp_path / "allowed")
    monkeypatch.setattr(module, "validate_profile_source", lambda p: None)
    with pytest.raises(ValueError, match="assigned"):
        module.main(["prepare", "--root", str(tmp_path / "elsewhere")])


def test_nested_measurement_root_is_rejected_before_supervisor(
    tmp_path: Path, monkeypatch
):
    module = importlib.import_module("tools.tier_d_scaling.__main__")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        module.main(
            [
                "measure",
                "--root",
                str(tmp_path / "prepared/child"),
                "--prepared-root",
                str(tmp_path / "prepared"),
                "--case",
                "n100",
            ]
        )
    assert not (tmp_path / "prepared").exists()
