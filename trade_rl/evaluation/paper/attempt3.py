"""Attempt 3 of the carry paper screen: draft, seal and launch gates.

Attempt 2 ended INVALID from a software defect (Issue #810 comment 6711911740). Its
preserved root is available, so ordinary predecessor verification applies: the
attempt-2 identity (protocol digest, event count, final tip, last observation time) is
read from that root with the existing v1 journal reader, opened read-only, and recorded
in lineage v4 together with the fact that the root holds the permanent failure marker
(existence only, never its content). Nothing else of the attempt-2 root is read: no
price, equity, fill or timing figure. The predecessor-root-unavailable G0 exception of
attempt 2 does not exist for attempt 3.

Like ``attempt2`` this module has no collector or network surface of its own and three
commands: ``plan`` (no side effect except an optional write-once draft), ``seal``
(writes a new, empty root; offline) and ``launch`` (the supervised collector; the only
networked step here). The storage probe is ``storage_probe``, run unchanged with its own
authorization. Every sealing or networked command refuses unless an Issue #810
authorization comment URL is supplied; the URL is checked by shape only. The sealed
plan is ``carry_paper_screen_v3`` and the attempt-2 seal is never reused.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trade_rl.evaluation.forward_study.seal import CodeIdentity, current_code_identity
from trade_rl.evaluation.paper.account import PaperSettings, timestamp
from trade_rl.evaluation.paper.attempt2 import (
    ALREADY_RUNNING,
    SCREEN_DAYS,
    TERMINAL_GRACE_SECONDS,
    Attempt2Refusal,
    _capacity_reasons,
    _checkout_reasons,
    _collector_has_run,
    _identity_block,
    _start_reasons,
    authorization_reasons,
    capacity_plan,
    write_draft,
)
from trade_rl.evaluation.paper.lock import collector_lock_held
from trade_rl.evaluation.paper.operations import (
    STORAGE_PROBE_PLAN_SCHEMA,
    screen_plan_v3,
    seal_paper_study,
    validate_attempt_lineage_v4,
)
from trade_rl.evaluation.paper.predecessor import (
    _ATTEMPT3_SEAL_TOKEN,
    LINEAGE_V4_SCHEMA,
    PredecessorIdentity,
    available_root_problems,
    canonical_lineage_v4,
    lineage_structure_problems,
    read_predecessor_status,
    same_canonical_record,
)
from trade_rl.evaluation.paper.watchdog import (
    EXIT_LOCK_HELD,
    ChildProcess,
    WatchdogPolicy,
    popen_launcher,
    supervise_collector,
)
from trade_rl.evaluation.runs import build_candidate_run_provenance

DRAFT_SCHEMA = "carry_paper_attempt3_draft_v1"
SEAL_SCHEMA = "carry_paper_attempt3_seal_v1"


class Attempt3Refusal(Attempt2Refusal):
    """Every failed precondition, collected so the operator sees all of them."""


def read_predecessor_identity(
    predecessor_root: str | Path | None,
    predecessor_protocol_sha256: str | None,
    *,
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> tuple[PredecessorIdentity | None, list[str]]:
    """The attempt-2 identity from its preserved root, read through the v1 reader.

    Only the event count, the final tip and the last observation time are taken from
    the journal report, and the failure marker's existence is checked. The reader is
    given the protocol digest the operator names (the digest the attempt-2 launch
    printed); it refuses a root whose chain does not match it.
    """
    if predecessor_root is None:
        return None, ["predecessor_root_required"]
    if (
        not isinstance(predecessor_protocol_sha256, str)
        or len(predecessor_protocol_sha256) != 64
        or predecessor_protocol_sha256 != predecessor_protocol_sha256.lower()
        or any(c not in "0123456789abcdef" for c in predecessor_protocol_sha256)
    ):
        return None, ["predecessor_protocol_sha256_required"]
    path = Path(predecessor_root).absolute()
    if path.is_symlink() or not path.is_dir():
        return None, ["predecessor_root_not_a_directory"]
    try:
        report = read(
            path, expected_protocol_sha256=predecessor_protocol_sha256, read_only=True
        )
        observed = report["status"]["observed_at"]
        events, tip = report["events"], report["tip"]
    except Exception as error:
        return None, [f"predecessor_root_unreadable:{type(error).__name__}"]
    if observed is None:
        return None, ["predecessor_root_has_no_observation"]
    try:
        observed_text = timestamp(observed).isoformat()
    except (TypeError, ValueError):
        return None, ["predecessor_root_last_observation_invalid"]
    if type(events) is not int or events < 1 or not isinstance(tip, str):
        return None, ["predecessor_root_report_invalid"]
    return (
        PredecessorIdentity(
            protocol_sha256=predecessor_protocol_sha256,
            final_tip_sha256=tip,
            last_observed_at=observed_text,
            event_count=events,
        ),
        [],
    )


def resolve_lineage(
    predecessor_root: str | Path | None,
    predecessor_protocol_sha256: str | None,
    *,
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Lineage v4 from the preserved attempt-2 root, or the reasons it cannot be built.

    The root is read once to build the record and a second time through the same gate
    every collector entry uses, which also requires the permanent failure marker.
    """
    identity, reasons = read_predecessor_identity(
        predecessor_root, predecessor_protocol_sha256, read=read
    )
    if identity is None or predecessor_root is None:
        return None, reasons
    try:
        lineage = canonical_lineage_v4(
            protocol_sha256=identity.protocol_sha256,
            final_tip_sha256=identity.final_tip_sha256,
            last_observed_at=identity.last_observed_at,
            event_count=identity.event_count,
            root_path=str(Path(predecessor_root).absolute()),
        )
    except ValueError as error:
        return None, [f"lineage_invalid:{error}"]
    problems = available_root_problems(lineage, read=read)
    if problems:
        return None, problems
    return lineage, []


def _lineage_reasons(lineage: dict[str, Any] | None, now: datetime) -> list[str]:
    if lineage is None:
        return []
    try:
        validate_attempt_lineage_v4(lineage, sealed_at=now)
    except ValueError as error:
        return [f"lineage_invalid:{error}"]
    return []


def _root_reasons(root: Path, lineage: dict[str, Any] | None) -> list[str]:
    """A new root: not beneath, nor containing, the preserved attempt-2 root."""
    if lineage is None:
        return []
    predecessor = Path(
        lineage["predecessor_root"]["verification"]["root_path"]
    ).absolute()
    target = root.absolute()
    if (
        target == predecessor
        or predecessor in target.parents
        or target in predecessor.parents
    ):
        return ["root_overlaps_predecessor_root"]
    return []


def _steps(root: Path, start_at: datetime, authorization: str) -> list[dict[str, Any]]:
    module = "python -m trade_rl.evaluation.paper.attempt3"
    base = f"{module} %s --root {root} --start-at {start_at.isoformat()}"
    auth = f"--authorization {authorization}"
    predecessor = (
        "--predecessor-root <ATTEMPT_2_ROOT> "
        "--predecessor-protocol-sha256 <ATTEMPT_2_PROTOCOL_SHA256>"
    )
    return [
        dict(
            step="seal",
            network=False,
            command=(
                f"{base % 'seal'} {auth} {predecessor} "
                "--bytes-per-observation <PROBE_CONSERVATIVE_RATE>"
            ),
        ),
        dict(
            step="launch",
            network=True,
            command=(
                f"{base % 'launch'} {auth} --allow-network "
                "--protocol-sha256 <PRINTED_BY_SEAL> --log-file <OUTSIDE_ROOT>"
            ),
        ),
        dict(
            step="monitor",
            network=False,
            command=(
                "python -m trade_rl.evaluation.paper.cli check "
                f"--root {root} --protocol-sha256 <PRINTED_BY_SEAL>"
            ),
        ),
        dict(
            step="evaluate_after_close_plus_180s",
            network=False,
            command=(
                "python -m trade_rl.evaluation.paper.cli evaluate "
                f"--root {root} --protocol-sha256 <PRINTED_BY_SEAL> "
                "--expected-tip <FINAL_TIP> --output <NEW_FILE>"
            ),
        ),
    ]


def plan_attempt3(
    root: str | Path,
    *,
    start_at: datetime,
    authorization: str | None,
    predecessor_root: str | Path | None = None,
    predecessor_protocol_sha256: str | None = None,
    bytes_per_observation: float | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    code_identity: Callable[[], CodeIdentity] = current_code_identity,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    read_predecessor: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> dict[str, Any]:
    """Return the unsealed draft. Refuses a non-future start or a missing authorization.

    Dirty or unidentifiable checkouts and missing capacity evidence do not refuse a
    plan; they are reported so the operator can fix them before ``seal`` refuses.
    """
    now = clock().astimezone(UTC)
    reasons = authorization_reasons(authorization)
    reasons += _start_reasons(start_at, now)
    lineage, lineage_problems = resolve_lineage(
        predecessor_root, predecessor_protocol_sha256, read=read_predecessor
    )
    reasons += lineage_problems + _lineage_reasons(lineage, now)
    reasons += _root_reasons(Path(root), lineage)
    if reasons or lineage is None:
        raise Attempt3Refusal(reasons)
    assert authorization is not None
    start = start_at.astimezone(UTC)
    root_path = Path(root).absolute()
    capacity = capacity_plan(
        root_path,
        bytes_per_observation=bytes_per_observation,
        disk_usage=disk_usage,
    )
    identity = code_identity()
    settings = PaperSettings(
        start_at=start, close_at=start + timedelta(days=SCREEN_DAYS)
    )
    blockers = (
        _checkout_reasons(identity)
        + _capacity_reasons(capacity)
        + (["root_exists"] if root_path.exists() or root_path.is_symlink() else [])
    )
    return dict(
        schema=DRAFT_SCHEMA,
        status="DRAFT_NOT_SEALED",
        sealed=False,
        launched=False,
        production_eligible=False,
        planned_at=now.isoformat(),
        authorization=authorization,
        authorization_check="url_shape_only_human_must_have_posted_it",
        root=str(root_path),
        start_at=settings.start_at.isoformat(),
        close_at=settings.close_at.isoformat(),
        terminal_deadline=(
            settings.close_at + timedelta(seconds=TERMINAL_GRACE_SECONDS)
        ).isoformat(),
        settings=settings.to_dict(),
        attempt_lineage=lineage,
        predecessor_root_reverified_at_seal=True,
        research_plan=screen_plan_v3(attempt_lineage=lineage),
        source_runtime_identity=_identity_block(identity),
        capacity=capacity,
        seal_blockers=list(dict.fromkeys(blockers)),
        steps=_steps(root_path, start, authorization),
        human_prerequisites=[
            "post the Issue #810 authorization comment and pass its URL",
            "independent result-blind review of the exact packet digest and heads",
            "result-blind G0 review resolving the preserved attempt-2 root",
            "fresh storage probe with its own authorization (not an attempt-2 rate)",
            "dedicated frozen checkout and private locked environment",
            "sleep, hibernation and automatic updates disabled for the whole screen",
        ],
    )


def seal_attempt3(
    root: str | Path,
    *,
    start_at: datetime,
    authorization: str | None,
    bytes_per_observation: float | None,
    predecessor_root: str | Path | None = None,
    predecessor_protocol_sha256: str | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    code_identity: Callable[[], CodeIdentity] = current_code_identity,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    seal: Callable[..., str] = seal_paper_study,
    read_predecessor: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> dict[str, Any]:
    """Seal a new root. Never touches attempts 1 or 2; refuses on any failed precondition."""
    now = clock().astimezone(UTC)
    root_path = Path(root).absolute()
    identity = code_identity()
    capacity = capacity_plan(
        root_path,
        bytes_per_observation=bytes_per_observation,
        disk_usage=disk_usage,
    )
    reasons = authorization_reasons(authorization)
    reasons += _start_reasons(start_at, now)
    lineage, lineage_problems = resolve_lineage(
        predecessor_root, predecessor_protocol_sha256, read=read_predecessor
    )
    reasons += lineage_problems + _lineage_reasons(lineage, now)
    reasons += _root_reasons(root_path, lineage)
    reasons += _checkout_reasons(identity)
    reasons += _capacity_reasons(capacity)
    if root_path.exists() or root_path.is_symlink():
        reasons.append("root_exists")
    if reasons or lineage is None:
        raise Attempt3Refusal(reasons)
    digest = seal(
        root_path,
        start_at=start_at.astimezone(UTC),
        attempt_lineage=lineage,
        lineage_seal_token=_ATTEMPT3_SEAL_TOKEN,
        clock=clock,
    )
    return dict(
        schema=SEAL_SCHEMA,
        sealed=True,
        launched=False,
        protocol_sha256=digest,
        root=str(root_path),
        start_at=start_at.astimezone(UTC).isoformat(),
        predecessor_root_reverified_at_seal=True,
        source_runtime_identity=_identity_block(identity),
        capacity=capacity,
        production_eligible=False,
        next_step="launch (needs --allow-network) before the start; record the digest",
    )


def _protocol_reasons(
    root: Path,
    digest: str,
    start: datetime,
    *,
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> list[str]:
    import hashlib

    path = root / "protocol.json"
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        return ["protocol_not_a_regular_file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        return ["protocol_digest_mismatch"]
    protocol = json.loads(raw)["protocol"]
    reasons = []
    plan = protocol["study"].get("research_plan")
    if isinstance(plan, dict) and plan.get("schema") == STORAGE_PROBE_PLAN_SCHEMA:
        reasons.append("protocol_is_a_storage_probe")
    recorded = plan.get("attempt_lineage") if isinstance(plan, dict) else None
    if (
        type(recorded) is dict
        and recorded.get("schema") == LINEAGE_V4_SCHEMA
        and not lineage_structure_problems(recorded)
    ):
        reasons += available_root_problems(recorded, read=read)
        if not same_canonical_record(plan, screen_plan_v3(attempt_lineage=recorded)):
            reasons.append("protocol_not_attempt3_lineage")
    else:
        reasons.append("protocol_not_attempt3_lineage")
    if protocol["study"].get("provenance") != build_candidate_run_provenance():
        reasons.append("protocol_provenance_changed")
    if timestamp(protocol["settings"]["start_at"]) != start:
        reasons.append("protocol_start_differs")
    return reasons


def launch_attempt3(
    root: str | Path,
    *,
    protocol_sha256: str,
    start_at: datetime,
    authorization: str | None,
    allow_network: bool,
    log_file: str | Path,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    code_identity: Callable[[], CodeIdentity] = current_code_identity,
    launcher: Callable[..., Callable[[], ChildProcess]] = popen_launcher,
    supervise: Callable[..., dict[str, Any]] = supervise_collector,
    publish: Callable[[dict[str, Any]], None] = lambda _event: None,
    read_predecessor: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> dict[str, Any]:
    """Start the supervised collector for the sealed attempt-3 root (needs network).

    Every launch, including a supervisor relaunch after the start, re-reads the
    attempt-2 root with the v1 reader and refuses if it cannot be read or differs.
    """
    now = clock().astimezone(UTC)
    root_path = Path(root).absolute()
    reasons = authorization_reasons(authorization)
    if not allow_network:
        reasons.append("network_not_allowed")
    reasons += _checkout_reasons(code_identity())
    if start_at.tzinfo is None or start_at.utcoffset() is None:
        reasons.append("start_not_timezone_aware")
    else:
        start = start_at.astimezone(UTC)
        protocol = _protocol_reasons(
            root_path, protocol_sha256, start, read=read_predecessor
        )
        reasons += protocol
        if start <= now and (
            protocol or not _collector_has_run(root_path, protocol_sha256)
        ):
            reasons.append("start_not_in_future")
    if reasons:
        raise Attempt3Refusal(reasons)
    if collector_lock_held(root_path):
        raise Attempt3Refusal([ALREADY_RUNNING])
    return supervise(
        root_path,
        protocol_sha256=protocol_sha256,
        launch=launcher(root_path, protocol_sha256, log_file),
        policy=WatchdogPolicy(),
        clock=clock,
        publish=publish,
    )


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "seal", "launch"):
        child = commands.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--start-at", type=timestamp, required=True)
        child.add_argument("--authorization", default=None)
        if name in {"plan", "seal"}:
            child.add_argument("--bytes-per-observation", type=float, default=None)
            child.add_argument("--predecessor-root", type=Path, default=None)
            child.add_argument("--predecessor-protocol-sha256", default=None)
        if name == "plan":
            child.add_argument("--output", type=Path, default=None)
        if name == "launch":
            child.add_argument("--protocol-sha256", required=True)
            child.add_argument("--allow-network", action="store_true")
            child.add_argument("--log-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            draft = plan_attempt3(
                args.root,
                start_at=args.start_at,
                authorization=args.authorization,
                predecessor_root=args.predecessor_root,
                predecessor_protocol_sha256=args.predecessor_protocol_sha256,
                bytes_per_observation=args.bytes_per_observation,
            )
            if args.output is not None:
                write_draft(draft, args.output)
            _print(draft)
            return 0
        if args.command == "seal":
            _print(
                seal_attempt3(
                    args.root,
                    start_at=args.start_at,
                    authorization=args.authorization,
                    bytes_per_observation=args.bytes_per_observation,
                    predecessor_root=args.predecessor_root,
                    predecessor_protocol_sha256=args.predecessor_protocol_sha256,
                )
            )
            return 0
        outcome = launch_attempt3(
            args.root,
            protocol_sha256=args.protocol_sha256,
            start_at=args.start_at,
            authorization=args.authorization,
            allow_network=args.allow_network,
            log_file=args.log_file,
            publish=_print,
        )
        return int(outcome["exit_code"])
    except Attempt3Refusal as refusal:
        print(
            json.dumps(dict(refused=refusal.reasons, production_eligible=False)),
            file=sys.stderr,
            flush=True,
        )
        return EXIT_LOCK_HELD if refusal.reasons == [ALREADY_RUNNING] else 2
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
