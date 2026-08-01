"""Fail-closed Stage A evaluation command surface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO


def _not_implemented(_: argparse.Namespace, __: TextIO) -> int:
    raise NotImplementedError("Stage A command handler is not implemented")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execution-store", required=True)
    parser.add_argument("--baseline-config-digest", required=True)
    parser.add_argument("--output-root", required=True)


def _add_commands(subparsers: argparse._SubParsersAction) -> None:
    validation = subparsers.add_parser(
        "validation",
        help="evaluate and atomically publish Stage A validation",
    )
    _add_common_arguments(validation)
    validation.set_defaults(handler=_not_implemented)

    sealed_test = subparsers.add_parser(
        "sealed-test",
        help="open, evaluate, and atomically publish the Stage A sealed test",
    )
    _add_common_arguments(sealed_test)
    sealed_test.add_argument("--validation-package", required=True)
    sealed_test.add_argument("--database-url")
    sealed_test.set_defaults(handler=_not_implemented)

    complete = subparsers.add_parser(
        "run",
        help="run validation and conditionally open the Stage A sealed test",
    )
    _add_common_arguments(complete)
    complete.add_argument("--database-url")
    complete.set_defaults(handler=_not_implemented)


def add_stage_a_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register Stage A commands on the authoritative application parser."""

    stage_a = subparsers.add_parser(
        "stage-a",
        help="unseen-symbol validation and one-shot sealed-test evaluation",
    )
    commands = stage_a.add_subparsers(dest="stage_a_command", required=True)
    _add_commands(commands)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade-rl stage-a",
        description="Stage A unseen-symbol evaluation orchestration.",
    )
    commands = parser.add_subparsers(dest="stage_a_command", required=True)
    _add_commands(commands)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    del stderr
    arguments = list(sys.argv[1:] if argv is None else argv)
    output = stdout or sys.stdout
    args = build_parser().parse_args(arguments)
    handler = args.handler
    return int(handler(args, output))


__all__ = ["add_stage_a_parser", "build_parser", "main"]
