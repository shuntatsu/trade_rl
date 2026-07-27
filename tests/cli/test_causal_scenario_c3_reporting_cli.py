from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.cli import main
from trade_rl.evaluation.causal_scenario_c3_reporting import (
    evaluate_phase_a_gate,
    load_c3_aggregate_summary,
    write_c3_report_artifact,
    write_phase_a_gate_artifact,
)
from trade_rl.workflows.causal_scenario import c3_execution


def _streams() -> tuple[io.StringIO, io.StringIO]:
    return io.StringIO(), io.StringIO()


def test_c3_publish_cli_emits_machine_readable_result(
    tmp_path: Path,
    c3_reporting,
) -> None:
    summary_path = tmp_path / "summary.json"
    c3_reporting.write_summary(summary_path)
    stdout, stderr = _streams()

    code = main(
        [
            "causal-scenario",
            "publish",
            "--summary",
            str(summary_path),
            "--output",
            str(tmp_path / "evidence"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == "causal_scenario_c3_publish_result_v1"
    assert payload["status"] == "phase_a_authorized"
    assert payload["passed"] is True
    assert payload["production_status"] == "NO-GO"
    assert Path(payload["report_artifact_path"]).is_dir()
    assert Path(payload["gate_artifact_path"]).is_dir()


def test_c3_gate_cli_reverifies_report_before_publication(
    tmp_path: Path,
    c3_reporting,
) -> None:
    summary_path = tmp_path / "summary.json"
    c3_reporting.write_summary(summary_path)
    summary = load_c3_aggregate_summary(summary_path)
    gate = evaluate_phase_a_gate(summary)
    report = write_c3_report_artifact(tmp_path / "report", summary, gate)
    stdout, stderr = _streams()

    code = main(
        [
            "causal-scenario",
            "gate",
            "--report",
            str(report.root),
            "--output",
            str(tmp_path / "gate"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == "causal_scenario_c3_gate_result_v1"
    assert payload["report_artifact_digest"] == report.artifact_digest
    assert payload["passed"] is True
    assert payload["production_status"] == "NO-GO"


def test_c3_evaluate_cli_emits_execution_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    c3_reporting,
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    c3_reporting.write_summary(summary_path)
    summary = load_c3_aggregate_summary(summary_path)
    gate = evaluate_phase_a_gate(summary)
    report = write_c3_report_artifact(tmp_path / "report", summary, gate)
    gate_artifact = write_phase_a_gate_artifact(
        tmp_path / "gate",
        gate,
        report_artifact_digest=report.artifact_digest,
    )
    result = SimpleNamespace(
        source_summary_path=summary_path,
        summary=summary,
        gate=gate,
        report_artifact_root=report.root,
        report_artifact_digest=report.artifact_digest,
        gate_artifact_root=gate_artifact.root,
        gate_artifact_digest=gate_artifact.artifact_digest,
        production_status="NO-GO",
    )
    monkeypatch.setattr(
        c3_execution,
        "execute_c3_evaluation_request",
        lambda request_path, *, output_root: result,
    )
    stdout, stderr = _streams()

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

    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == "causal_scenario_c3_evaluation_result_v1"
    assert payload["summary_digest"] == summary.summary_digest
    assert payload["status"] == "phase_a_authorized"
    assert payload["production_status"] == "NO-GO"


def test_c3_cli_failure_is_one_line_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}\n", encoding="utf-8")

    def fail(request_path: Path, *, output_root: Path):
        raise c3_execution.C3CoreBackendUnavailable("lane B backend is unavailable")

    monkeypatch.setattr(c3_execution, "execute_c3_evaluation_request", fail)
    stdout, stderr = _streams()

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
    assert stderr.getvalue().count("\n") == 1
    assert json.loads(stderr.getvalue()) == {
        "error": "lane B backend is unavailable",
        "error_type": "C3CoreBackendUnavailable",
        "production_status": "NO-GO",
        "schema": "causal_scenario_c3_error_v1",
        "status": "failed",
    }


def test_c3_publish_cli_does_not_import_sb3_runtime(
    tmp_path: Path,
    c3_reporting,
) -> None:
    summary_path = tmp_path / "summary.json"
    c3_reporting.write_summary(summary_path)
    sys.modules.pop("trade_rl.integrations.sb3_training", None)
    stdout, stderr = _streams()

    code = main(
        [
            "causal-scenario",
            "publish",
            "--summary",
            str(summary_path),
            "--output",
            str(tmp_path / "evidence"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    assert "trade_rl.integrations.sb3_training" not in sys.modules
