"""Seal, collect, inspect and assess public-data paper studies. No live orders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.paper.account import timestamp
from trade_rl.evaluation.paper.assessment import evaluate_paper_study
from trade_rl.evaluation.paper.operations import (
    collect_until_finished,
    collection_status,
    seal_paper_study,
)
from trade_rl.evaluation.paper.supervisor import PaperCollector


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "run", "status", "evaluate"):
        child = commands.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        if name == "seal":
            child.add_argument("--start-at", type=timestamp, required=True)
        else:
            child.add_argument("--protocol-sha256", required=True)
        if name == "evaluate":
            child.add_argument("--expected-tip", required=True)
            child.add_argument(
                "--output",
                type=Path,
                required=True,
                help="New, write-once assessment file",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            digest = seal_paper_study(args.root, start_at=args.start_at)
            _print(
                dict(
                    protocol_sha256=digest,
                    root=str(args.root.absolute()),
                    production_eligible=False,
                )
            )
        elif args.command == "run":
            with PaperCollector(
                args.root, expected_protocol_sha256=args.protocol_sha256
            ) as collector:
                result = collect_until_finished(collector, publish=_print)
                status = result["status"]
                return int(
                    result["phase"] == "rejected"
                    or bool(status["quality_failures"])
                    or not status["terminal_flat"]
                    or status["pending"]
                )
        elif args.command == "status":
            _print(
                collection_status(
                    args.root, expected_protocol_sha256=args.protocol_sha256
                )
            )
        else:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("assessment output already exists")
            result = evaluate_paper_study(
                args.root,
                expected_protocol_sha256=args.protocol_sha256,
                expected_tip=args.expected_tip,
            )
            with args.output.open("xb") as stream:
                stream.write(canonical_json_bytes(result))
                stream.flush()
                os.fsync(stream.fileno())
            _print(result)
            return 0 if result["decision"] == "PAPER_SCREEN_PASSED" else 1
        return 0
    except Exception as error:
        print(
            json.dumps(
                dict(
                    error_type=type(error).__name__,
                    error=str(error),
                    production_eligible=False,
                )
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
