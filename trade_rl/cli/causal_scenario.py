"""Machine-readable CLI for causal-scenario C3 evaluation evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from trade_rl.evaluation.causal_scenario_c3_reporting import (
    evaluate_phase_a_gate,
    load_c3_aggregate_summary,
    load_c3_report_artifact,
    write_c3_report_artifact,
    write_phase_a_gate_artifact,
)

PRODUCTION_STATUS = "NO-GO"


def _emit(stream: TextIO, payload: dict[str, object]) -> None:
    stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    stream.write("\n")


def _status(passed: bool) -> str:
    return "phase_a_authorized" if passed else "phase_a_blocked"


def _run(action: Callable[[], dict[str, object]], *, stdout: TextIO, stderr: TextIO) -> int:
    try:
        payload = action()
    except Exception as error:
        _emit(
            stderr,
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "production_status": PRODUCTION_STATUS,
                "schema": "causal_scenario_c3_error_v1",
                "status": "failed",
            },
        )
        return 1
    _emit(stdout, payload)
    return 0


def _evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl causal-scenario evaluate")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _publish_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl causal-scenario publish")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _gate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl causal-scenario gate")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def run_evaluate(
    argv: Sequence[str],
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    arguments = _evaluate_parser().parse_args(list(argv))
    output = stdout or sys.stdout
    errors = stderr or sys.stderr

    def action() -> dict[str, object]:
        from trade_rl.workflows.causal_scenario import c3_execution

        result = c3_execution.execute_c3_evaluation_request(
            Path(arguments.request),
            output_root=Path(arguments.output),
        )
        return {
            "gate_artifact_digest": result.gate_artifact_digest,
            "gate_artifact_path": str(result.gate_artifact_root),
            "passed": result.gate.passed,
            "production_status": result.production_status,
            "report_artifact_digest": result.report_artifact_digest,
            "report_artifact_path": str(result.report_artifact_root),
            "schema": "causal_scenario_c3_evaluation_result_v1",
            "source_summary_path": str(result.source_summary_path),
            "status": _status(result.gate.passed),
            "summary_digest": result.summary.summary_digest,
        }

    return _run(action, stdout=output, stderr=errors)


def run_publish(
    argv: Sequence[str],
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    arguments = _publish_parser().parse_args(list(argv))
    output = stdout or sys.stdout
    errors = stderr or sys.stderr

    def action() -> dict[str, object]:
        destination = Path(arguments.output)
        summary = load_c3_aggregate_summary(Path(arguments.summary))
        gate = evaluate_phase_a_gate(summary)
        report = write_c3_report_artifact(destination / "report", summary, gate)
        gate_artifact = write_phase_a_gate_artifact(
            destination / "gate",
            gate,
            report_artifact_digest=report.artifact_digest,
        )
        return {
            "gate_artifact_digest": gate_artifact.artifact_digest,
            "gate_artifact_path": str(gate_artifact.root),
            "passed": gate.passed,
            "production_status": PRODUCTION_STATUS,
            "report_artifact_digest": report.artifact_digest,
            "report_artifact_path": str(report.root),
            "schema": "causal_scenario_c3_publish_result_v1",
            "status": _status(gate.passed),
            "summary_digest": summary.summary_digest,
        }

    return _run(action, stdout=output, stderr=errors)


def run_gate(
    argv: Sequence[str],
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    arguments = _gate_parser().parse_args(list(argv))
    output = stdout or sys.stdout
    errors = stderr or sys.stderr

    def action() -> dict[str, object]:
        report = load_c3_report_artifact(Path(arguments.report))
        gate_artifact = write_phase_a_gate_artifact(
            Path(arguments.output),
            report.gate,
            report_artifact_digest=report.artifact_digest,
        )
        return {
            "gate_artifact_digest": gate_artifact.artifact_digest,
            "gate_artifact_path": str(gate_artifact.root),
            "passed": report.gate.passed,
            "production_status": PRODUCTION_STATUS,
            "report_artifact_digest": report.artifact_digest,
            "schema": "causal_scenario_c3_gate_result_v1",
            "status": _status(report.gate.passed),
        }

    return _run(action, stdout=output, stderr=errors)


__all__ = ["run_evaluate", "run_gate", "run_publish"]
