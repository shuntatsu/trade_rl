"""Lightweight command-line entrypoint for :mod:`trade_rl`."""

from __future__ import annotations

import argparse
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

    return _build_parser()


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Dispatch offline/artifact commands without importing the RL runtime."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["studio"]:
        from trade_rl.studio.cli import main as studio_main

        return studio_main(arguments[1:])
    if tuple(arguments[:2]) in _ARTIFACT_COMMANDS:
        from trade_rl.cli.extended import main as artifact_main

        return artifact_main(arguments, stdout=stdout, stderr=stderr)
    from trade_rl.cli.app import main as application_main

    return application_main(arguments, stdout=stdout, stderr=stderr)


__all__ = ["build_parser", "main"]
