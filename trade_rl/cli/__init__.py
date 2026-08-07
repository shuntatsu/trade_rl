"""Lightweight command-line entrypoint for :mod:`trade_rl`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO

_ARTIFACT_COMMANDS = {
    ("confirmation", "create"),
    ("release", "approve"),
    ("selection", "authorize"),
    ("serving", "package"),
    ("train", "run"),
    ("walk-forward", "run"),
}


def build_parser() -> argparse.ArgumentParser:
    """Load the full research CLI parser only when it is requested."""

    from trade_rl.cli.app import build_parser as _build_parser
    from trade_rl.cli.stage_a import add_stage_a_parser

    parser = _build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    add_stage_a_parser(subparsers)
    return parser


def _single_symbol_preflight_error(
    error: Exception,
    *,
    arguments: Sequence[str],
    stderr: TextIO,
) -> int:
    schema = (
        "training_run_error_v1"
        if tuple(arguments[:2]) == ("train", "run")
        else "walk_forward_run_error_v1"
    )
    stderr.write(
        json.dumps(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "production_status": "NO-GO",
                "schema": schema,
                "status": "failed",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    return 1


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Dispatch offline/artifact commands without importing the RL runtime."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    if not arguments:
        build_parser().print_help(file=output)
        return 2
    if arguments in (["-h"], ["--help"]):
        build_parser().print_help(file=output)
        return 0
    if arguments[:1] == ["studio"]:
        from trade_rl.studio.cli import main as studio_main

        return studio_main(arguments[1:])
    if arguments[:1] == ["stage-a"]:
        from trade_rl.cli.stage_a import main as stage_a_main

        return stage_a_main(arguments[1:], stdout=output, stderr=errors)
    if arguments[:2] == ["causal-scenario", "evaluate"]:
        from trade_rl.cli.causal_scenario import run_evaluate

        return run_evaluate(arguments[2:], stdout=output, stderr=errors)
    if arguments[:2] == ["causal-scenario", "publish"]:
        from trade_rl.cli.causal_scenario import run_publish

        return run_publish(arguments[2:], stdout=output, stderr=errors)
    if arguments[:2] == ["causal-scenario", "verify"]:
        from trade_rl.cli.causal_scenario import run_verify

        return run_verify(arguments[2:], stdout=output, stderr=errors)
    if tuple(arguments[:2]) in _ARTIFACT_COMMANDS:
        if tuple(arguments[:2]) in {("train", "run"), ("walk-forward", "run")}:
            from trade_rl.cli.maintained_single_symbol import (
                require_maintained_single_symbol_cli,
            )

            try:
                require_maintained_single_symbol_cli(arguments)
            except Exception as error:
                return _single_symbol_preflight_error(
                    error,
                    arguments=arguments,
                    stderr=errors,
                )
        from trade_rl.cli.extended import main as artifact_main

        return artifact_main(arguments, stdout=output, stderr=errors)
    from trade_rl.cli.app import main as application_main

    return application_main(arguments, stdout=output, stderr=errors)


__all__ = ["build_parser", "main"]
