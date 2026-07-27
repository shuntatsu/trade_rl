from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.cli import main
from trade_rl.workflows.causal_scenario import c3_evaluation


def test_c3_evaluate_cli_emits_machine_readable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    gate = SimpleNamespace(
        failed_condition_names=("fold_and_day_support",),
        digest="a" * 64,
        passed=False,
    )
    report = SimpleNamespace(digest="b" * 64)
    batch = SimpleNamespace(
        comparison_count=12,
        gate=gate,
        gate_artifact_digest="c" * 64,
        gate_artifact_root=output / "gate",
        report=report,
        report_artifact_digest="d" * 64,
        report_artifact_root=output / "report",
    )
    result = SimpleNamespace(
        batch=batch,
        production_status="NO-GO",
        request_digest="e" * 64,
        schema_version="causal_scenario_c3_evaluation_result_v1",
        source_run_digest="f" * 64,
    )
    monkeypatch.setattr(
        c3_evaluation,
        "execute_c3_evaluation_request",
        lambda path, *, output_root: result,
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "causal-scenario",
            "evaluate",
            "--request",
            str(request),
            "--output",
            str(output),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == "causal_scenario_c3_evaluation_result_v1"
    assert payload["status"] == "phase_a_blocked"
    assert payload["comparison_count"] == 12
    assert payload["production_status"] == "NO-GO"
    assert payload["source_run_digest"] == "f" * 64


def test_c3_evaluate_cli_fails_closed_as_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")

    def fail(path: Path, *, output_root: Path) -> None:
        raise ValueError("source run identity mismatch")

    monkeypatch.setattr(c3_evaluation, "execute_c3_evaluation_request", fail)
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "causal-scenario",
            "evaluate",
            "--request",
            str(request),
            "--output",
            str(tmp_path / "output"),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 1
    assert stdout.getvalue() == ""
    payload = json.loads(stderr.getvalue())
    assert payload == {
        "error": "source run identity mismatch",
        "error_type": "ValueError",
        "production_status": "NO-GO",
        "schema": "causal_scenario_evaluation_error_v1",
        "status": "failed",
    }
