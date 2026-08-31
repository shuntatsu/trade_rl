from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import run_universal_causal_alpha_v10_research as module


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--config",
        str(tmp_path / "v7.json"),
        "--run-config",
        str(tmp_path / "selection-run.json"),
        "--runtime-manifest",
        str(tmp_path / "runtime.json"),
        "--v4-context-manifest",
        str(tmp_path / "v4.json"),
        "--frozen-metadata-root",
        str(tmp_path / "metadata"),
        "--output-root",
        str(tmp_path / "output"),
    ]


def test_cli_forwards_optional_signal_run_config(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(digest="a" * 64, passed=False)

    monkeypatch.setattr(module, "run_causal_alpha_v10_selection", fake_run)
    signal_config = tmp_path / "signal-run.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_universal_causal_alpha_v10_research",
            *_argv(tmp_path),
            "--signal-run-config",
            str(signal_config),
        ],
    )
    status = module.main()

    assert status == 3
    assert captured["run_config_path"] == tmp_path / "selection-run.json"
    assert captured["signal_run_config_path"] == signal_config
    assert json.loads(capsys.readouterr().out) == {
        "artifact_digest": "a" * 64,
        "promotion_eligible": False,
        "status": "selection_rejected",
    }
