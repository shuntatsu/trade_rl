"""Machine-readable CLI for integrated causal-scenario C3 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from trade_rl.evaluation.causal_scenario_c3_markdown import (
    LoadedC3MarkdownArtifact,
    VerifiedC3Evidence,
    verify_c3_evidence,
    write_c3_markdown_artifact,
)

if TYPE_CHECKING:
    from trade_rl.workflows.causal_scenario.c3_evaluation import C3EvaluationResult

PRODUCTION_STATUS = "NO-GO"


def _write_json(stream: TextIO, payload: object) -> None:
    stream.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    stream.write("\n")


def _error(stderr: TextIO, error: Exception) -> int:
    _write_json(
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


def _status(passed: bool) -> str:
    return "phase_a_authorized" if passed else "phase_a_blocked"


def _evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl causal-scenario evaluate")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _publish_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl causal-scenario publish")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _verify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl causal-scenario verify")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    return parser


def _execute(request: Path, *, output_root: Path) -> C3EvaluationResult:
    from trade_rl.workflows.causal_scenario.c3_evaluation import (
        execute_c3_evaluation_request,
    )

    return execute_c3_evaluation_request(request, output_root=output_root)


def _publish_markdown(
    output: Path,
    *,
    report_root: Path,
    gate_root: Path,
) -> LoadedC3MarkdownArtifact:
    return write_c3_markdown_artifact(
        output,
        report_root=report_root,
        gate_root=gate_root,
    )


def _verify(*, report_root: Path, gate_root: Path) -> VerifiedC3Evidence:
    return verify_c3_evidence(report_root=report_root, gate_root=gate_root)


def run_evaluate(
    argv: Sequence[str],
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    args = _evaluate_parser().parse_args(list(argv))
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    try:
        result = _execute(args.request, output_root=args.output)
        markdown = _publish_markdown(
            args.output / "markdown",
            report_root=result.batch.report_artifact_root,
            gate_root=result.batch.gate_artifact_root,
        )
    except Exception as error:
        return _error(errors, error)
    gate = result.batch.gate
    _write_json(
        output,
        {
            "comparison_count": result.batch.comparison_count,
            "failed_conditions": list(gate.failed_condition_names),
            "gate_artifact_digest": result.batch.gate_artifact_digest,
            "gate_artifact_path": str(result.batch.gate_artifact_root),
            "gate_digest": gate.digest,
            "markdown_artifact_digest": markdown.artifact_digest,
            "markdown_artifact_path": str(markdown.root),
            "passed": gate.passed,
            "production_status": result.production_status,
            "report_artifact_digest": result.batch.report_artifact_digest,
            "report_artifact_path": str(result.batch.report_artifact_root),
            "report_digest": result.batch.report.digest,
            "request_digest": result.request_digest,
            "schema": result.schema_version,
            "source_run_digest": result.source_run_digest,
            "status": _status(gate.passed),
        },
    )
    return 0


def run_publish(
    argv: Sequence[str],
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    args = _publish_parser().parse_args(list(argv))
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    try:
        artifact = _publish_markdown(
            args.output,
            report_root=args.report,
            gate_root=args.gate,
        )
    except Exception as error:
        return _error(errors, error)
    _write_json(
        output,
        {
            "artifact_digest": artifact.artifact_digest,
            "artifact_path": str(artifact.root),
            "failed_conditions": list(artifact.failed_condition_names),
            "gate_artifact_digest": artifact.gate_artifact_digest,
            "gate_digest": artifact.gate_digest,
            "passed": artifact.passed,
            "production_status": artifact.production_status,
            "report_artifact_digest": artifact.report_artifact_digest,
            "report_digest": artifact.report_digest,
            "schema": "causal_scenario_c3_markdown_result_v1",
            "status": _status(artifact.passed),
        },
    )
    return 0


def run_verify(
    argv: Sequence[str],
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    args = _verify_parser().parse_args(list(argv))
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    try:
        evidence = _verify(report_root=args.report, gate_root=args.gate)
    except Exception as error:
        return _error(errors, error)
    _write_json(
        output,
        {
            "failed_conditions": list(evidence.failed_condition_names),
            "gate_artifact_digest": evidence.gate_artifact_digest,
            "gate_digest": evidence.gate_digest,
            "passed": evidence.passed,
            "production_status": evidence.production_status,
            "report_artifact_digest": evidence.report_artifact_digest,
            "report_digest": evidence.report_digest,
            "schema": "causal_scenario_c3_verify_result_v1",
            "status": _status(evidence.passed),
        },
    )
    return 0


__all__ = ["run_evaluate", "run_publish", "run_verify"]
