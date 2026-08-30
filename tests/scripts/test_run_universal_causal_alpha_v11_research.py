from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import run_universal_causal_alpha_v11_research as module
from trade_rl.learning.causal_alpha_v11 import CausalAlphaV11StudyArm


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
        "--r21-output-root",
        str(tmp_path / "r21"),
        "--output-root",
        str(tmp_path / "output"),
        "--study-arm",
        "neutral_expiry_2",
    ]


def test_cli_forwards_one_study_arm(monkeypatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            digest="a" * 64,
            passed=False,
            source_v8=object(),
            study_arm=CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2,
        )

    monkeypatch.setattr(module, "run_causal_alpha_v11_selection", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_universal_causal_alpha_v11_research", *_argv(tmp_path)],
    )

    status = module.main()

    assert status == 3
    assert captured["study_arm"] is CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2
    assert json.loads(capsys.readouterr().out)["status"] == "selection_rejected"


def test_cli_returns_four_for_preflight_stop(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        module,
        "run_causal_alpha_v11_selection",
        lambda **_kwargs: SimpleNamespace(
            digest="a" * 64,
            passed=False,
            source_v8=None,
            study_arm=CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING,
        ),
    )
    argv = _argv(tmp_path)
    argv[-1] = "calibrated_edge_sizing"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_universal_causal_alpha_v11_research", *argv],
    )

    assert module.main() == 4
