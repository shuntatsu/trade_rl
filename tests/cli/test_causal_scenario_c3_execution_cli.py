from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.cli import main
from trade_rl.cli import causal_scenario as module


def _streams() -> tuple[io.StringIO, io.StringIO]:
    return io.StringIO(), io.StringIO()


def _gate() -> SimpleNamespace:
    return SimpleNamespace(
        digest="2" * 64,
        failed_condition_names=(),
        passed=True,
        report_digest="1" * 64,
    )


def test_evaluate_cli_publishes_core_and_markdown_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}\n", encoding="utf-8")
    batch = SimpleNamespace(
        comparison_count=12,
        gate=_gate(),
        gate_artifact_digest="3" * 64,
        gate_artifact_root=tmp_path / "gate",
        report=SimpleNamespace(digest="1" * 64),
        report_artifact_digest="4" * 64,
        report_artifact_root=tmp_path / "report",
    )
    execution = SimpleNamespace(
        batch=batch,
        production_status="NO-GO",
        request_digest="5" * 64,
        schema_version="causal_scenario_c3_evaluation_result_v2",
        source_run_digest="6" * 64,
    )
    markdown = SimpleNamespace(
        artifact_digest="7" * 64,
        root=tmp_path / "markdown",
    )
    monkeypatch.setattr(module, "_execute", lambda *args, **kwargs: execution)
    monkeypatch.setattr(module, "_publish_markdown", lambda *args, **kwargs: markdown)
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
    assert payload["schema"] == "causal_scenario_c3_evaluation_result_v2"
    assert payload["report_artifact_digest"] == "4" * 64
    assert payload["gate_artifact_digest"] == "3" * 64
    assert payload["markdown_artifact_digest"] == "7" * 64
    assert payload["status"] == "phase_a_authorized"
    assert payload["production_status"] == "NO-GO"


def test_publish_cli_verifies_core_artifacts_before_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = SimpleNamespace(
        artifact_digest="7" * 64,
        root=tmp_path / "markdown",
        report_artifact_digest="4" * 64,
        gate_artifact_digest="3" * 64,
        report_digest="1" * 64,
        gate_digest="2" * 64,
        passed=True,
        production_status="NO-GO",
    )
    monkeypatch.setattr(module, "_publish_markdown", lambda *args, **kwargs: markdown)
    stdout, stderr = _streams()

    code = main(
        [
            "causal-scenario",
            "publish",
            "--report",
            str(tmp_path / "report"),
            "--gate",
            str(tmp_path / "gate"),
            "--output",
            str(tmp_path / "markdown"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == "causal_scenario_c3_markdown_result_v1"
    assert payload["artifact_digest"] == "7" * 64
    assert payload["production_status"] == "NO-GO"


def test_verify_cli_returns_machine_readable_core_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = SimpleNamespace(
        report_artifact_digest="4" * 64,
        gate_artifact_digest="3" * 64,
        report_digest="1" * 64,
        gate_digest="2" * 64,
        passed=False,
        failed_condition_names=("aggregate_uplift_confidence",),
        production_status="NO-GO",
    )
    monkeypatch.setattr(module, "_verify", lambda *args, **kwargs: verified)
    stdout, stderr = _streams()

    code = main(
        [
            "causal-scenario",
            "verify",
            "--report",
            str(tmp_path / "report"),
            "--gate",
            str(tmp_path / "gate"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "phase_a_blocked"
    assert payload["failed_conditions"] == ["aggregate_uplift_confidence"]
    assert payload["production_status"] == "NO-GO"


def test_cli_failure_is_one_line_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_verify",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid evidence")),
    )
    stdout, stderr = _streams()

    code = main(
        [
            "causal-scenario",
            "verify",
            "--report",
            str(tmp_path / "report"),
            "--gate",
            str(tmp_path / "gate"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue().count("\n") == 1
    assert json.loads(stderr.getvalue()) == {
        "error": "invalid evidence",
        "error_type": "ValueError",
        "production_status": "NO-GO",
        "schema": "causal_scenario_c3_error_v1",
        "status": "failed",
    }


def test_publish_cli_does_not_import_sb3_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("trade_rl.integrations.sb3_training", None)
    monkeypatch.setattr(
        module,
        "_publish_markdown",
        lambda *args, **kwargs: SimpleNamespace(
            artifact_digest="7" * 64,
            root=tmp_path / "markdown",
            report_artifact_digest="4" * 64,
            gate_artifact_digest="3" * 64,
            report_digest="1" * 64,
            gate_digest="2" * 64,
            passed=True,
            production_status="NO-GO",
        ),
    )
    stdout, stderr = _streams()

    assert (
        main(
            [
                "causal-scenario",
                "publish",
                "--report",
                str(tmp_path / "report"),
                "--gate",
                str(tmp_path / "gate"),
                "--output",
                str(tmp_path / "markdown"),
            ],
            stdout=stdout,
            stderr=stderr,
        )
        == 0
    )
    assert "trade_rl.integrations.sb3_training" not in sys.modules
