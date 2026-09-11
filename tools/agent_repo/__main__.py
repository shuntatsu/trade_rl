"""Network-free CLI for source-derived repository-agent inspection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.agent_repo.git_state import read_git_state
from tools.agent_repo.source_index import SourceIndex


def _write_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--base", dest="base_ref")

    context = subparsers.add_parser("context")
    context.add_argument("path")

    impact = subparsers.add_parser("impact")
    impact.add_argument("paths", nargs="+")
    return parser


def _dispatch(args: argparse.Namespace, repository: Path) -> object:
    command = str(args.command)
    if command == "preflight":
        return asdict(read_git_state(repository, base_ref=args.base_ref))

    index = SourceIndex.build(repository)
    if command == "context":
        return asdict(index.context(str(args.path)))
    if command == "impact":
        return {
            "contexts": [asdict(value) for value in index.impact(tuple(args.paths))]
        }
    raise ValueError(f"unsupported command: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        payload: Any = _dispatch(args, Path.cwd())
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        SyntaxError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2
    _write_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
