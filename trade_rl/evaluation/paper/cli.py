"""Seal, collect, inspect and assess public-data paper studies. No live orders."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.paper.account import timestamp
from trade_rl.evaluation.paper.assessment import evaluate_paper_study
from trade_rl.evaluation.paper.lock import CollectorLockHeld
from trade_rl.evaluation.paper.operations import (
    STORAGE_PROBE_PLAN_SCHEMA,
    collect_until_finished,
    collection_status,
    seal_paper_study,
)
from trade_rl.evaluation.paper.predecessor import (
    ProtocolFormatError,
    collection_lineage_problems,
    load_canonical_protocol,
    loads_without_duplicate_keys,
)
from trade_rl.evaluation.paper.supervisor import PaperCollector
from trade_rl.evaluation.paper.watchdog import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_STALE_SECONDS,
    EXIT_DEGRADED,
    EXIT_FAILED,
    EXIT_LOCK_HELD,
    CollectorTerminated,
    ExitGuard,
    HeartbeatWriter,
    WatchdogPolicy,
    check_collection,
    install_termination_handlers,
    popen_launcher,
    supervise_collector,
)


def _refuse_non_screen_root(root: Path, protocol_sha256: str) -> None:
    """Refuse a probe root and any root whose lineage the shared classifier rejects.

    A probe is capacity evidence only (only its own entry point runs it). A v3
    ``available`` record is re-read exactly as ``attempt2 launch`` does, on every
    entry. Duplicate JSON keys are refused here, before the lock, so no marker is
    written for that input.
    """
    path = root / "protocol.json"
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        return
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol_sha256:
        return
    try:
        protocol = load_canonical_protocol(raw)["protocol"]
        plan = protocol["study"]["research_plan"]
        start_at = timestamp(protocol["settings"]["start_at"])
    except ProtocolFormatError:
        raise
    except (KeyError, TypeError, ValueError):
        return
    if isinstance(plan, dict) and plan.get("schema") == STORAGE_PROBE_PLAN_SCHEMA:
        raise ValueError("a storage probe root is not a screen; the probe runs itself")
    problems = collection_lineage_problems(
        plan, root=root, start_at=start_at, now=lambda: datetime.now(UTC)
    )
    if problems:
        raise ValueError(
            "collection lineage cannot be re-verified: " + "; ".join(problems)
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
        help="Stateless heartbeat check (exit 0 healthy, 3 marker, 5 degraded)",
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
            attempt_lineage = loads_without_duplicate_keys(lineage_path.read_bytes())
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
            # 0 healthy; 3 a failure marker exists or was just written;
            # 5 unhealthy but no marker written (nothing has been lost yet).
            if report["severity"] == "healthy":
                return 0
            return EXIT_FAILED if report["severity"] == "failed" else EXIT_DEGRADED
        elif args.command == "supervise":
            _refuse_non_screen_root(args.root, args.protocol_sha256)
            outcome = supervise_collector(
                args.root,
                protocol_sha256=args.protocol_sha256,
                launch=popen_launcher(args.root, args.protocol_sha256, args.log_file),
                policy=WatchdogPolicy(stale_seconds=args.stale_seconds),
                publish=_print,
            )
            return int(outcome["exit_code"])
        elif args.command == "run":
            _refuse_non_screen_root(args.root, args.protocol_sha256)
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
    except CollectorLockHeld as held:
        # Another live collector owns the root. This process wrote no marker and
        # touched nothing, so the owner's screen is undisturbed.
        print(
            json.dumps(
                dict(
                    error_type="CollectorLockHeld",
                    error=str(held),
                    production_eligible=False,
                )
            ),
            file=sys.stderr,
            flush=True,
        )
        return EXIT_LOCK_HELD
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
