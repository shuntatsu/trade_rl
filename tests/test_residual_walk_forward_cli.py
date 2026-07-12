from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest


def test_run_pipeline_dispatches_residual_wf(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--action-mode",
            "baseline-residual",
            "--phase",
            "wf",
            "--no-register",
            "--output",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        "mars_lite.eval.residual_walk_forward.run_residual_walk_forward",
        lambda args, output: calls.append("wf"),
    )
    monkeypatch.setattr(
        "mars_lite.pipeline.residual_pipeline.run_baseline_residual",
        lambda args, output: calls.append("train"),
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_path("scripts/run_pipeline.py", run_name="__main__")

    assert exc.value.code == 0
    assert calls == ["wf"]


def test_dedicated_script_dispatches_residual_wf(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline_residual.py",
            "--phase",
            "wf",
            "--output",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        "mars_lite.eval.residual_walk_forward.run_residual_walk_forward",
        lambda args, output: calls.append("wf"),
    )
    monkeypatch.setattr(
        "mars_lite.pipeline.residual_pipeline.run_baseline_residual",
        lambda args, output: calls.append("train"),
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_path("scripts/run_baseline_residual.py", run_name="__main__")

    assert exc.value.code == 0
    assert calls == ["wf"]
