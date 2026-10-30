"""Short public-network storage probe for the attempt-2 capacity check.

Runs the real paper collector (same capture code, journal and control files as the
screen) for a few one-minute slots into a NEW scratch root outside the repository,
then reports bytes per observation. The root carries ``probe_plan``: it names no
attempt lineage, so ``attempt2 seal``/``launch`` and ``cli evaluate`` all refuse it,
and ``evaluate`` additionally refuses the probe schema by name. A probe is capacity
evidence only; it never counts toward, replaces or pre-registers a screen, computes
no P&L and places no order.

This is the only network step besides ``attempt2 launch``. It lives apart from
``attempt2`` so that module keeps no collector or network surface (see
``tests/architecture/test_paper_operations_boundary.py``). It refuses unless
``--allow-network`` and an Issue #810 authorization comment URL are supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trade_rl.evaluation.forward_study.seal import CodeIdentity, current_code_identity
from trade_rl.evaluation.paper.account import PaperSettings
from trade_rl.evaluation.paper.attempt2 import (
    CAPACITY_HEADROOM,
    SCREEN_SLOTS,
    Attempt2Refusal,
    authorization_reasons,
    write_draft,
)
from trade_rl.evaluation.paper.operations import (
    STORAGE_PROBE_PLAN_SCHEMA,
    collect_until_finished,
)
from trade_rl.evaluation.paper.supervisor import PaperCollector, seal_collection
from trade_rl.evaluation.paper.watchdog import HeartbeatWriter

PROBE_REPORT_SCHEMA = "carry_paper_storage_probe_report_v1"
PROBE_DEFAULT_SLOTS = 20
PROBE_MINIMUM_SLOTS = 10
PROBE_MAXIMUM_SLOTS = 60
PROBE_MINIMUM_OBSERVATIONS = 5
PROBE_LEAD = timedelta(seconds=60)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def probe_plan() -> dict[str, Any]:
    """The research plan of a storage probe root: unmistakably not a screen.

    It names no attempt lineage, so ``seal``, ``launch`` and ``evaluate`` all refuse
    it, and ``evaluate`` additionally refuses this exact schema by name.
    """
    return {
        "schema": STORAGE_PROBE_PLAN_SCHEMA,
        "probe_only": True,
        "attempt_lineage": None,
        "screen_eligible": False,
        "evaluable": False,
        "production_eligible": False,
        "purpose": "bytes_per_observation_measurement_for_attempt2_capacity_only",
    }


def _probe_root_reasons(root: Path, repository_root: Path) -> list[str]:
    reasons = []
    if root.exists() or root.is_symlink():
        reasons.append("probe_root_exists")
    resolved = root.resolve()
    repository = repository_root.resolve()
    if resolved == repository or repository in resolved.parents:
        reasons.append("probe_root_inside_repository")
    if any((parent / "protocol.json").exists() for parent in root.parents):
        reasons.append("probe_root_inside_a_sealed_root")
    return reasons


def storage_breakdown(root: Path) -> dict[str, int]:
    """Bytes under a collection root, split into rule captures, snapshots and the rest."""
    rules = snapshots = other = rule_captures = snapshot_captures = 0
    seen: set[Path] = set()
    for directory, _names, files in os.walk(root):
        folder = Path(directory)
        in_sources = folder != root and folder.relative_to(root).parts[0] == "sources"
        is_rules = in_sources and (folder / "rules.json").is_file()
        for name in files:
            path = folder / name
            if path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            if not in_sources:
                other += size
                continue
            if is_rules:
                rules += size
            else:
                snapshots += size
            if folder not in seen:
                seen.add(folder)
                rule_captures += int(is_rules)
                snapshot_captures += int(not is_rules)
    return dict(
        rule_capture_bytes=rules,
        snapshot_capture_bytes=snapshots,
        other_bytes=other,
        total_bytes=rules + snapshots + other,
        rule_captures=rule_captures,
        snapshot_captures=snapshot_captures,
    )


def measure_storage(
    root: str | Path,
    *,
    authorization: str | None,
    allow_network: bool,
    slots: int = PROBE_DEFAULT_SLOTS,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    transport: Any = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    code_identity: Callable[[], CodeIdentity] = current_code_identity,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Run the real public collector for a few minutes into a scratch probe root.

    Uses the same collector, capture code and journal as the screen, so the measured
    bytes include captures, hourly rules, the event journal and control files. The
    probe root carries ``probe_plan`` and can never be sealed, launched or evaluated
    as attempt 2. The only network step; it needs ``allow_network`` and an Issue #810
    authorization URL, and tests substitute ``transport``/``clock``/``sleep``.
    """
    root_path = Path(root).absolute()
    reasons = authorization_reasons(authorization)
    if not allow_network:
        reasons.append("network_not_allowed")
    if (
        isinstance(slots, bool)
        or not isinstance(slots, int)
        or not PROBE_MINIMUM_SLOTS <= slots <= PROBE_MAXIMUM_SLOTS
    ):
        reasons.append("probe_slots_out_of_range")
    reasons += _probe_root_reasons(root_path, repository_root)
    if reasons:
        raise Attempt2Refusal(reasons)
    assert authorization is not None
    now = clock().astimezone(UTC)
    start = now + PROBE_LEAD
    settings = PaperSettings(
        start_at=start, close_at=start + timedelta(minutes=slots - 1)
    )
    digest = seal_collection(
        root_path, settings=settings, research_plan=probe_plan(), clock=clock
    )
    try:
        with PaperCollector(
            root_path,
            expected_protocol_sha256=digest,
            clock=clock,
            transport=transport,
            monotonic=monotonic,
        ) as collector:
            outcome = collect_until_finished(
                collector,
                clock=clock,
                sleep=sleep,
                publish=lambda _event: None,
                heartbeat=HeartbeatWriter(root_path, digest, clock=clock),
            )
    except Exception as error:
        raise Attempt2Refusal(
            [f"probe_collection_failed:{type(error).__name__}:{error}"]
        ) from error
    status = outcome["status"]
    observations = int(status["observations"])
    if observations < PROBE_MINIMUM_OBSERVATIONS:
        raise Attempt2Refusal(["probe_too_few_observations"])
    usage = storage_breakdown(root_path)
    measured = usage["total_bytes"] / observations
    snapshot_mean = usage["snapshot_capture_bytes"] / max(1, usage["snapshot_captures"])
    # A trade cycle takes a second snapshot (decision, then execution); a short probe
    # may not trade, so the bound assumes every observation does.
    conservative = measured + snapshot_mean
    identity = code_identity()
    return dict(
        schema=PROBE_REPORT_SCHEMA,
        probe_only=True,
        sealable_as_attempt2=False,
        evaluable=False,
        production_eligible=False,
        network_used=True,
        authorization=authorization,
        authorization_check="url_shape_only_human_must_have_posted_it",
        root=str(root_path),
        probe_protocol_sha256=digest,
        git_head=identity.git_head,
        tracked_tree_clean=identity.tracked_tree_clean,
        slots=slots,
        observations=observations,
        events=int(status["events"]),
        quality_failures=list(status["quality_failures"]),
        storage=usage,
        bytes_per_observation=math.ceil(measured),
        conservative_bytes_per_observation=math.ceil(conservative),
        required_bytes_at_conservative_rate=math.ceil(
            SCREEN_SLOTS * conservative * CAPACITY_HEADROOM
        ),
        screen_minute_slots=SCREEN_SLOTS,
        headroom=CAPACITY_HEADROOM,
        notes=[
            "measured rate includes the fixed files and one hourly rule capture of a "
            "short probe, which overstates the long-run rate",
            "conservative rate adds one extra snapshot per observation for trade cycles",
            "pass conservative_bytes_per_observation to plan and seal; the capacity "
            "check applies the 1.25 headroom itself",
        ],
    )


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorization", default=None)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--slots", type=int, default=PROBE_DEFAULT_SLOTS)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output is not None and (
            args.output.exists() or args.output.is_symlink()
        ):
            raise FileExistsError("storage probe output already exists")
        report = measure_storage(
            args.root,
            authorization=args.authorization,
            allow_network=args.allow_network,
            slots=args.slots,
        )
        if args.output is not None:
            write_draft(report, args.output)
        _print(report)
        return 0
    except Attempt2Refusal as refusal:
        print(
            json.dumps(dict(refused=refusal.reasons, production_eligible=False)),
            file=sys.stderr,
            flush=True,
        )
        return 2
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
