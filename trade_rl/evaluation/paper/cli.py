"""Seal, collect, inspect and assess public-data paper studies. No live orders."""

from __future__ import annotations

import argparse
import atexit
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
from trade_rl.evaluation.paper.watchdog import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_STALE_SECONDS,
    CollectorTerminated,
    ExitGuard,
    HeartbeatWriter,
    WatchdogPolicy,
    check_collection,
    install_termination_handlers,
    popen_launcher,
    supervise_collector,
)


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
            child.add_argument("--attempt-lineage-json", type=Path, required=True)
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
    watch = commands.add_parser(
        "check",
        help="Stateless heartbeat check; after the start a stale heartbeat records a failure",
    )
    watch.add_argument("--root", type=Path, required=True)
    watch.add_argument("--protocol-sha256", required=True)
    watch.add_argument("--stale-seconds", type=float, default=DEFAULT_STALE_SECONDS)
    supervise = commands.add_parser(
        "supervise",
        help="Run and relaunch the collector child inside the frozen gap limit",
    )
    supervise.add_argument("--root", type=Path, required=True)
    supervise.add_argument("--protocol-sha256", required=True)
    supervise.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Append-only child output; must be outside the sealed root",
    )
    supervise.add_argument("--stale-seconds", type=float, default=DEFAULT_STALE_SECONDS)
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            lineage_path = args.attempt_lineage_json
            if lineage_path.is_symlink() or not lineage_path.is_file():
                raise ValueError("attempt lineage must be a regular JSON file")
            attempt_lineage = json.loads(lineage_path.read_bytes())
            digest = seal_paper_study(
                args.root,
                start_at=args.start_at,
                attempt_lineage=attempt_lineage,
            )
            _print(
                dict(
                    protocol_sha256=digest,
                    root=str(args.root.absolute()),
                    production_eligible=False,
                )
            )
        elif args.command == "check":
            report = check_collection(
                args.root,
                protocol_sha256=args.protocol_sha256,
                stale_seconds=args.stale_seconds,
            )
            _print(report)
            return 0 if report["healthy"] else 3
        elif args.command == "supervise":
            outcome = supervise_collector(
                args.root,
                protocol_sha256=args.protocol_sha256,
                launch=popen_launcher(args.root, args.protocol_sha256, args.log_file),
                policy=WatchdogPolicy(stale_seconds=args.stale_seconds),
                publish=_print,
            )
            return int(outcome["exit_code"])
        elif args.command == "run":
            with PaperCollector(
                args.root, expected_protocol_sha256=args.protocol_sha256
            ) as collector:
                # Armed only while this process owns the collector lock.
                guard = ExitGuard(args.root, args.protocol_sha256)
                atexit.register(guard)
                install_termination_handlers(guard.console_event)
                result = collect_until_finished(
                    collector,
                    publish=_print,
                    heartbeat=HeartbeatWriter(args.root, args.protocol_sha256),
                    heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
                )
                guard.disarm()
                atexit.unregister(guard)
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
    except CollectorTerminated as terminated:
        # The collector context already recorded the permanent failure marker.
        print(
            json.dumps(
                dict(
                    error_type="CollectorTerminated",
                    error=str(terminated),
                    production_eligible=False,
                )
            ),
            file=sys.stderr,
            flush=True,
        )
        return 3
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
