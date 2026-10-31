"""Attempt 2 of the carry paper screen: draft, seal and launch gates.

Attempt 1 is invalidated for `observation_gap`; it is never resumed or rewritten.
This module reuses the existing lineage validator, screen plan, seal and supervisor.
Its commands are (the storage probe lives in ``storage_probe`` so this module keeps
no collector or network surface):

* ``plan`` (the safe first step; the subcommand is always explicit, and it has no
  side effect except an optional write-once draft file);
* ``seal`` (writes a new, empty root and its protocol digest; no network);
* ``launch`` (starts the supervised collector; the only networked step here).

``plan`` and ``seal`` must say whether attempt 1's evidence root is ``available`` or
``unavailable``. ``available`` requires a root path that the existing v1 journal reader
verifies against the recorded protocol digest, event count, final tip and last
observation. ``unavailable`` records that the lineage rests on a prior recorded review
and was not re-verified at seal time, together with the user's attestation.

Every networked or sealing command refuses unless an Issue #810 authorization comment
URL is supplied. The URL is checked by shape only: a human must have posted the comment
first, and nothing here can prove it exists. No command computes P&L or places an order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.forward_study.seal import (
    AUTHORIZATION_PATTERN,
    CodeIdentity,
    current_code_identity,
)
from trade_rl.evaluation.paper.account import PaperSettings, timestamp
from trade_rl.evaluation.paper.lock import collector_lock_held
from trade_rl.evaluation.paper.operations import (
    STORAGE_PROBE_PLAN_SCHEMA,
    screen_plan,
    seal_paper_study,
    validate_attempt_lineage,
)
from trade_rl.evaluation.paper.predecessor import (
    _ATTEMPT2_SEAL_TOKEN,
    LINEAGE_V3_SCHEMA,
    PREDECESSOR_ROOT_STATUSES,
    _same_canonical_record,
    available_root_problems,
    canonical_lineage,
    read_predecessor_status,
    verify_predecessor_root,
)
from trade_rl.evaluation.paper.watchdog import (
    EXIT_LOCK_HELD,
    ChildProcess,
    WatchdogPolicy,
    popen_launcher,
    read_heartbeat,
    supervise_collector,
)
from trade_rl.evaluation.runs import build_candidate_run_provenance

DRAFT_SCHEMA = "carry_paper_attempt2_draft_v1"
SCREEN_DAYS = 90
TERMINAL_GRACE_SECONDS = 180
MINIMUM_SEAL_NOTICE = timedelta(minutes=5)
SCREEN_SLOTS = SCREEN_DAYS * 24 * 60 + TERMINAL_GRACE_SECONDS // 60 + 1
CAPACITY_HEADROOM = 1.25


ALREADY_RUNNING = "collector_already_running"


class Attempt2Refusal(ValueError):
    """Every failed precondition, collected so the operator sees all of them."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(dict.fromkeys(reasons))
        super().__init__("; ".join(self.reasons))


def attempt2_lineage(
    predecessor_root_status: str, predecessor_root: str | Path | None = None
) -> dict[str, Any]:
    """The attempt-1 binding. Asserted values, not an authenticated root.

    ``unavailable`` is the honest record when the root is lost; ``available`` records
    the root path and is only built by :func:`resolve_lineage` (after the root has
    been read) or by ``launch`` (which re-reads it).
    """
    return canonical_lineage(
        predecessor_root_status,
        None if predecessor_root is None else str(predecessor_root),
    )


def resolve_lineage(
    predecessor_root_status: str | None,
    predecessor_root: str | Path | None,
    *,
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Choose the lineage for the stated status, or the reasons it cannot be claimed.

    ``available`` cannot be claimed without a root path that the v1 reader verifies;
    ``unavailable`` cannot be combined with a root path (an existing root must be read).
    """
    if predecessor_root_status is None:
        return None, ["predecessor_root_status_missing"]
    if predecessor_root_status not in PREDECESSOR_ROOT_STATUSES:
        return None, ["predecessor_root_status_invalid"]
    if predecessor_root_status == "unavailable":
        if predecessor_root is not None:
            return None, ["predecessor_root_given_but_declared_unavailable"]
        return attempt2_lineage("unavailable"), []
    if predecessor_root is None:
        return None, ["predecessor_root_required_for_available"]
    reasons = verify_predecessor_root(predecessor_root, read=read)
    if reasons:
        return None, reasons
    return attempt2_lineage("available", Path(predecessor_root).absolute()), []


def accepted_lineage(
    recorded: object,
    *,
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> tuple[dict[str, Any] | None, list[str]]:
    """The lineage `launch` accepts for a sealed protocol's recorded lineage.

    ``unavailable`` is canonical. ``available`` is accepted only if the root path it
    records is read again now by the v1 reader and matches the recorded digests; a
    missing, moved or changed root is refused, even for a restart. Journal
    events and heartbeat records do not attest an earlier successful root read.
    """
    if not isinstance(recorded, dict) or recorded.get("schema") != LINEAGE_V3_SCHEMA:
        return None, []
    record = recorded.get("predecessor_root")
    if not isinstance(record, dict) or record.get("status") not in (
        PREDECESSOR_ROOT_STATUSES
    ):
        return None, []
    if record["status"] == "unavailable":
        return attempt2_lineage("unavailable"), []
    problems = available_root_problems(recorded, read=read)
    if problems:
        return None, problems
    path = record["verification"]["root_path"]
    return attempt2_lineage("available", path), []


def authorization_reasons(authorization: str | None) -> list[str]:
    if not authorization:
        return ["authorization_missing"]
    if AUTHORIZATION_PATTERN.match(authorization) is None:
        return ["authorization_malformed"]
    return []


def _start_reasons(start_at: datetime, now: datetime) -> list[str]:
    if start_at.tzinfo is None or start_at.utcoffset() is None:
        return ["start_not_timezone_aware"]
    start = start_at.astimezone(UTC)
    if start <= now:
        return ["start_not_in_future"]
    if start < now + MINIMUM_SEAL_NOTICE:
        return ["start_notice_too_short"]
    return []


def _lineage_reasons(lineage: dict[str, Any] | None, now: datetime) -> list[str]:
    if lineage is None:
        return []
    try:
        validate_attempt_lineage(lineage, sealed_at=now)
    except ValueError as error:
        return [f"lineage_invalid:{error}"]
    return []


def _checkout_reasons(identity: CodeIdentity) -> list[str]:
    if identity.git_head is None:
        return ["checkout_identity_unavailable"]
    if not identity.tracked_tree_clean:
        return ["checkout_dirty"]
    return []


def _existing_parent(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path(path.anchor or ".")


def capacity_plan(
    root: Path,
    *,
    bytes_per_observation: float | None,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    """Storage requirement for the whole screen from an operator-measured rate.

    The rate must include journal, raw captures and the hourly exchange rules (for
    example attempt 1's total bytes divided by its 188 observations); this offline
    tool never measures the network or reads attempt 1.
    """
    location = _existing_parent(root.absolute())
    free = int(disk_usage(location).free)
    plan: dict[str, Any] = dict(
        status="not_assessed",
        screen_days=SCREEN_DAYS,
        minute_slots=SCREEN_SLOTS,
        headroom=CAPACITY_HEADROOM,
        bytes_per_observation=None,
        required_bytes=None,
        free_bytes=free,
        location=str(location),
    )
    if bytes_per_observation is None:
        return plan
    if (
        isinstance(bytes_per_observation, bool)
        or not math.isfinite(bytes_per_observation)
        or bytes_per_observation <= 0
    ):
        return dict(plan, status="invalid_rate")
    required = math.ceil(SCREEN_SLOTS * bytes_per_observation * CAPACITY_HEADROOM)
    return dict(
        plan,
        status="ok" if free >= required else "insufficient",
        bytes_per_observation=bytes_per_observation,
        required_bytes=required,
    )


def _capacity_reasons(capacity: dict[str, Any]) -> list[str]:
    return [] if capacity["status"] == "ok" else [f"capacity_{capacity['status']}"]


def _identity_block(identity: CodeIdentity) -> dict[str, Any]:
    provenance = build_candidate_run_provenance()
    return dict(
        git_head=identity.git_head,
        tracked_tree_clean=identity.tracked_tree_clean,
        implementation_digest=provenance["implementation_digest"],
        runtime_environment_digest=provenance["runtime_environment_digest"],
        provenance_schema=provenance["schema_version"],
    )


def _steps(
    root: Path,
    start_at: datetime,
    authorization: str,
    bytes_per_observation: Any,
    predecessor_root_status: str,
) -> list[dict[str, Any]]:
    module = "python -m trade_rl.evaluation.paper.attempt2"
    base = f"{module} %s --root {root} --start-at {start_at.isoformat()}"
    rate = "<MEASURED_BYTES_PER_OBSERVATION>"
    if bytes_per_observation is not None:
        rate = str(bytes_per_observation)
    auth = f"--authorization {authorization}"
    predecessor = f"--predecessor-root-status {predecessor_root_status}"
    if predecessor_root_status == "available":
        predecessor += " --predecessor-root <ATTEMPT_1_ROOT>"
    return [
        dict(
            step="seal",
            network=False,
            command=(
                f"{base % 'seal'} {auth} {predecessor} --bytes-per-observation {rate}"
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


def _review_prerequisite(predecessor_root_status: str) -> str:
    if predecessor_root_status == "available":
        return "result-blind G0 review of the preserved attempt-1 root"
    return (
        "result-blind review of the lineage documentation and the pull request that "
        "introduced the lost-predecessor rule (attempt 1's root is lost and is not "
        "re-verified)"
    )


def plan_attempt2(
    root: str | Path,
    *,
    start_at: datetime,
    authorization: str | None,
    predecessor_root_status: str | None = None,
    predecessor_root: str | Path | None = None,
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
        predecessor_root_status, predecessor_root, read=read_predecessor
    )
    reasons += lineage_problems + _lineage_reasons(lineage, now)
    if reasons or lineage is None:
        raise Attempt2Refusal(reasons)
    assert authorization is not None and predecessor_root_status is not None
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
        predecessor_root_status=predecessor_root_status,
        predecessor_root_reverified_at_seal=predecessor_root_status == "available",
        research_plan=screen_plan(attempt_lineage=lineage),
        source_runtime_identity=_identity_block(identity),
        capacity=capacity,
        seal_blockers=list(dict.fromkeys(blockers)),
        steps=_steps(
            root_path,
            start,
            authorization,
            bytes_per_observation,
            predecessor_root_status,
        ),
        human_prerequisites=[
            "post the Issue #810 authorization comment and pass its URL",
            _review_prerequisite(predecessor_root_status),
            "dedicated frozen checkout and private locked environment",
            "sleep, hibernation and automatic updates disabled for the whole screen",
        ],
    )


def write_draft(draft: dict[str, Any], output: Path) -> None:
    """Write-once draft file (exclusive create, fsync)."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("attempt-2 draft output already exists")
    with output.open("xb") as stream:
        stream.write(canonical_json_bytes(draft))
        stream.flush()
        os.fsync(stream.fileno())


def seal_attempt2(
    root: str | Path,
    *,
    start_at: datetime,
    authorization: str | None,
    bytes_per_observation: float | None,
    predecessor_root_status: str | None = None,
    predecessor_root: str | Path | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    code_identity: Callable[[], CodeIdentity] = current_code_identity,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    seal: Callable[..., str] = seal_paper_study,
    read_predecessor: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> dict[str, Any]:
    """Seal a new root. Never touches attempt 1; refuses on any failed precondition."""
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
        predecessor_root_status, predecessor_root, read=read_predecessor
    )
    reasons += lineage_problems + _lineage_reasons(lineage, now)
    reasons += _checkout_reasons(identity)
    reasons += _capacity_reasons(capacity)
    if root_path.exists() or root_path.is_symlink():
        reasons.append("root_exists")
    if reasons or lineage is None:
        raise Attempt2Refusal(reasons)
    digest = seal(
        root_path,
        start_at=start_at.astimezone(UTC),
        attempt_lineage=lineage,
        attempt2_seal_token=_ATTEMPT2_SEAL_TOKEN,
        clock=clock,
    )
    return dict(
        schema="carry_paper_attempt2_seal_v1",
        sealed=True,
        launched=False,
        protocol_sha256=digest,
        root=str(root_path),
        start_at=start_at.astimezone(UTC).isoformat(),
        predecessor_root_status=predecessor_root_status,
        predecessor_root_reverified_at_seal=predecessor_root_status == "available",
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
    lineage, root_problems = accepted_lineage(
        plan.get("attempt_lineage") if isinstance(plan, dict) else None,
        read=read,
    )
    reasons += root_problems
    if lineage is None or not _same_canonical_record(
        plan, screen_plan(attempt_lineage=lineage)
    ):
        reasons.append("protocol_not_attempt2_lineage")
    if protocol["study"].get("provenance") != build_candidate_run_provenance():
        reasons.append("protocol_provenance_changed")
    if timestamp(protocol["settings"]["start_at"]) != start:
        reasons.append("protocol_start_differs")
    return reasons


def _collector_has_run(root: Path, protocol_sha256: str) -> bool:
    try:
        return read_heartbeat(root, protocol_sha256) is not None
    except ValueError:
        return False


def launch_attempt2(
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
    """Start the supervised collector for the sealed attempt-2 root (needs network).

    A first launch (before the start) re-reads an ``available`` predecessor root with
    the v1 reader and refuses if it is gone or differs.
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
            root_path,
            protocol_sha256,
            start,
            read=read_predecessor,
        )
        reasons += protocol
        # A first launch must precede the start. A supervisor relaunch after the
        # start is allowed only for a verified root whose collector already ran;
        # it can resume nothing (the supervisor refuses halted or finished roots).
        if start <= now and (
            protocol or not _collector_has_run(root_path, protocol_sha256)
        ):
            reasons.append("start_not_in_future")
    if reasons:
        raise Attempt2Refusal(reasons)
    if collector_lock_held(root_path):
        # A live collector already owns this root (for example the survivor of a
        # supervisor restart). Refuse without writing anything or any marker.
        raise Attempt2Refusal([ALREADY_RUNNING])
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
            child.add_argument(
                "--predecessor-root-status",
                default=None,
                help="available (requires --predecessor-root) or unavailable",
            )
            child.add_argument("--predecessor-root", type=Path, default=None)
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
            draft = plan_attempt2(
                args.root,
                start_at=args.start_at,
                authorization=args.authorization,
                bytes_per_observation=args.bytes_per_observation,
                predecessor_root_status=args.predecessor_root_status,
                predecessor_root=args.predecessor_root,
            )
            if args.output is not None:
                write_draft(draft, args.output)
            _print(draft)
            return 0
        if args.command == "seal":
            _print(
                seal_attempt2(
                    args.root,
                    start_at=args.start_at,
                    authorization=args.authorization,
                    bytes_per_observation=args.bytes_per_observation,
                    predecessor_root_status=args.predecessor_root_status,
                    predecessor_root=args.predecessor_root,
                )
            )
            return 0
        outcome = launch_attempt2(
            args.root,
            protocol_sha256=args.protocol_sha256,
            start_at=args.start_at,
            authorization=args.authorization,
            allow_network=args.allow_network,
            log_file=args.log_file,
            publish=_print,
        )
        return int(outcome["exit_code"])
    except Attempt2Refusal as refusal:
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
