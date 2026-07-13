"""Extended executable CLI with artifact-producing training commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from trade_rl.cli.app import main as legacy_main
from trade_rl.workflows.training_run import execute_training_run


def _write_json(stream: TextIO, payload: object) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    stream.write("\n")


def _train_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl train run")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    return parser


def _run_training(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    args = _train_run_parser().parse_args(list(argv))
    try:
        result = execute_training_run(
            config_path=args.config,
            dataset_path=args.dataset,
            store_root=args.output,
            run_id=args.run_id,
        )
    except Exception as error:
        _write_json(
            stderr,
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "production_status": "NO-GO",
                "schema": "training_run_error_v1",
                "status": "failed",
            },
        )
        return 1
    _write_json(
        stdout,
        {
            "artifact_path": str(result.path),
            "dataset_id": result.dataset_id,
            "policy_digest": result.policy_digest,
            "production_status": result.production_status,
            "run_digest": result.run_digest,
            "run_id": result.run_id,
            "schema": "training_run_result_v1",
            "status": result.status,
        },
    )
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Dispatch artifact-producing commands and preserve the existing CLI."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    if arguments[:2] == ["train", "run"]:
        return _run_training(arguments[2:], stdout=output, stderr=errors)
    return legacy_main(arguments, stdout=output)


if __name__ == "__main__":
    raise SystemExit(main())
