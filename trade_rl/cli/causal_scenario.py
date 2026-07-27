"""Evaluation-only CLI handlers for causal-scenario evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from trade_rl.evaluation.causal_scenario_c3_artifact import (
    load_c3_aggregate_report_artifact,
    write_phase_a_gate_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_gate import evaluate_phase_a_entry_gate


def _write_json(stream: TextIO, payload: object) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    stream.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl causal-scenario gate")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run_gate(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Load a verified C3 report and publish the pure Phase A gate artifact."""

    args = _parser().parse_args(list(argv))
    try:
        loaded_report = load_c3_aggregate_report_artifact(args.report)
        gate = evaluate_phase_a_entry_gate(loaded_report.report)
        artifact_digest = write_phase_a_gate_artifact(args.output, gate)
    except Exception as error:
        _write_json(
            stderr,
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "production_status": "NO-GO",
                "schema": "causal_scenario_gate_error_v1",
                "status": "failed",
            },
        )
        return 1
    _write_json(
        stdout,
        {
            "artifact_digest": artifact_digest,
            "artifact_path": str(args.output),
            "failed_conditions": gate.failed_condition_names,
            "gate_digest": gate.digest,
            "passed": gate.passed,
            "production_status": "NO-GO",
            "report_digest": loaded_report.report.digest,
            "schema": "causal_scenario_gate_result_v1",
            "status": "phase_a_authorized" if gate.passed else "phase_a_blocked",
        },
    )
    return 0
