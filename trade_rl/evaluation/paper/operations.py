"""Fixed prospective sealing and bounded-cadence public collection operations."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

from trade_rl._validation import require_aware_datetime
from trade_rl.evaluation.paper.account import PaperAccount, PaperSettings, timestamp
from trade_rl.evaluation.paper.collection_policy import (
    ATTEMPT_PURPOSES,
    CADENCE_SECONDS,
    MAX_CONSECUTIVE_SKIPPED_SLOTS,
    MAX_FAILED_CAPTURES_PER_SLOT,
    MAX_SKIPPED_SLOTS_PER_HOUR,
    MAX_SKIPPED_SLOTS_TOTAL,
    RETRIES_PER_SLOT,
    RETRY_DELAY_SECONDS,
    RETRY_START_WINDOW_SECONDS,
    RULES_HOLD_SECONDS,
    SKIP_REASONS,
    SKIP_WINDOW_SLOTS,
)
from trade_rl.evaluation.paper.predecessor import (
    _ATTEMPT2_SEAL_TOKEN,
    _ATTEMPT3_SEAL_TOKEN,
    LINEAGE_V3_SCHEMA,
    LINEAGE_V4_SCHEMA,
    PREDECESSOR_ROOT_STATUSES,
    ROOT_READER,
    SCREEN_V3_SCHEMA,
    canonical_root_record,
    lineage_structure_problems,
    plain_json_key_problems,
    same_canonical_record,
    v4_identity,
)
from trade_rl.evaluation.paper.store import PaperJournal
from trade_rl.evaluation.paper.supervisor import PaperCollector, seal_collection

_ATTEMPT_LINEAGE_SCHEMA = "carry_paper_attempt_lineage_v2"
_ATTEMPT_LINEAGE_SCHEMA_V3 = LINEAGE_V3_SCHEMA
STORAGE_PROBE_PLAN_SCHEMA = "carry_paper_storage_probe_v1"
# The frozen observation-gap limit; a test binds it to ``PaperSettings.maximum_gap_seconds``.
FROZEN_GAP_LIMIT_SECONDS = 180
_ATTEMPT_DISPOSITIONS = {
    "invalidated",
    "incomplete",
}
_NON_ECONOMIC_ATTEMPT_REASONS = {
    "clock_reversal",
    "collector_failure",
    "deadline_missed",
    "integrity_failure",
    "late_start",
    "manual_abort",
    "observation_gap",
    "review_blocked",
    "source_unavailable",
}


class _RequiredFillCoverage(TypedDict):
    symbol_index: int
    symbol: str
    venue: str


REQUIRED_FILL_COVERAGE: tuple[_RequiredFillCoverage, ...] = (
    {"symbol_index": 0, "symbol": "BTCUSDT", "venue": "spot"},
    {"symbol_index": 1, "symbol": "BTCUSDT", "venue": "perpetual"},
    {"symbol_index": 2, "symbol": "ETHUSDT", "venue": "spot"},
    {"symbol_index": 3, "symbol": "ETHUSDT", "venue": "perpetual"},
)
REQUIRED_FILL_SYMBOL_INDICES = tuple(
    row["symbol_index"] for row in REQUIRED_FILL_COVERAGE
)


_SHA1_HEX = r"[0-9a-f]{40}"


def _validate_predecessor_root(
    value: object, predecessor: dict[str, Any], *, sealed_at: datetime | None
) -> None:
    """Structure of the recorded predecessor-root status (lineage v3 only).

    This checks shape and internal consistency. It cannot prove that a root was read
    or that a quoted statement was made: ``attempt2`` reads the root before writing an
    ``available`` record, and an ``unavailable`` record is an explicit attestation.
    """
    if not isinstance(value, dict) or value.get("status") not in (
        PREDECESSOR_ROOT_STATUSES
    ):
        raise ValueError("attempt lineage predecessor root status is invalid")
    if value["status"] == "available":
        if set(value) != {"status", "verification"}:
            raise ValueError("available predecessor root requires verification")
        proof = value["verification"]
        if (
            not isinstance(proof, dict)
            or set(proof)
            != {
                "reader",
                "event_count",
                "protocol_sha256",
                "final_tip_sha256",
                "last_observed_at",
                "root_path",
            }
            or proof["reader"] != ROOT_READER
            or not isinstance(proof["root_path"], str)
            or not proof["root_path"]
            or type(proof["event_count"]) is not int
            or proof["event_count"] < 1
            or proof["protocol_sha256"] != predecessor["protocol_sha256"]
            or proof["final_tip_sha256"] != predecessor["final_tip_sha256"]
            or proof["last_observed_at"] != predecessor["last_observed_at"]
        ):
            raise ValueError("available predecessor root verification is inconsistent")
        return
    if set(value) != {
        "status",
        "re_verified_at_seal",
        "lineage_source",
        "user_statement",
        "not_established",
        "review_rule",
    }:
        raise ValueError("unavailable predecessor root record is incomplete")
    source = value["lineage_source"]
    statement = value["user_statement"]
    if (
        value["re_verified_at_seal"] is not False
        or not isinstance(source, dict)
        or set(source) != {"kind", "document", "commit"}
        or source["kind"] != "prior_result_blind_review_record"
        or not isinstance(source["document"], str)
        or not source["document"]
        or not isinstance(source["commit"], str)
        or re.fullmatch(_SHA1_HEX, source["commit"]) is None
        or not isinstance(statement, dict)
        or set(statement) != {"quote", "date", "channel", "states", "does_not_attest"}
        or not isinstance(statement["quote"], str)
        or not statement["quote"]
        or not isinstance(statement["date"], str)
        or statement["channel"] != "chat"
        or not isinstance(value["review_rule"], str)
        or not value["review_rule"]
    ):
        raise ValueError("unavailable predecessor root record is invalid")
    if not same_canonical_record(value, canonical_root_record("unavailable")):
        # The quote, states, limits, source and review rule are fixed text, not
        # free-form: anything but the canonical record is refused.
        raise ValueError("unavailable predecessor root record is not the canonical one")
    try:
        said = datetime.strptime(statement["date"], "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError("unavailable predecessor statement date is invalid") from error
    if (
        sealed_at is not None
        and said
        > require_aware_datetime(sealed_at, field="paper seal").astimezone(UTC).date()
    ):
        raise ValueError("unavailable predecessor statement is newer than the seal")


def validate_attempt_lineage(
    value: object,
    *,
    sealed_at: datetime | None = None,
) -> dict[str, Any]:
    """Validate the immutable parent link for this attempt in the screen series."""
    key_problems = plain_json_key_problems(value)
    if key_problems:
        raise ValueError(
            "attempt lineage keys are not plain JSON strings: "
            + "; ".join(key_problems)
        )
    if isinstance(value, dict) and (
        "predecessor_root" in value or value.get("schema") == LINEAGE_V3_SCHEMA
    ):
        # One shared classifier decides every v3-shaped record, whatever its label.
        problems = lineage_structure_problems(value)
        if problems:
            raise ValueError("attempt lineage is not canonical: " + "; ".join(problems))
    base_keys = {"schema", "attempt_number", "predecessor"}
    if not isinstance(value, dict) or set(value) not in (
        base_keys,
        base_keys | {"predecessor_root"},
    ):
        raise ValueError("attempt lineage must declare schema, number, and predecessor")
    version_three = value["schema"] == _ATTEMPT_LINEAGE_SCHEMA_V3
    if value["schema"] not in (_ATTEMPT_LINEAGE_SCHEMA, _ATTEMPT_LINEAGE_SCHEMA_V3):
        raise ValueError("unsupported paper attempt lineage schema")
    if version_three != ("predecessor_root" in value):
        raise ValueError("predecessor root status belongs to lineage v3 only")
    number = value["attempt_number"]
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError("attempt lineage number must be a positive integer")
    predecessor = value["predecessor"]
    if number == 1:
        if predecessor is not None or version_three:
            raise ValueError("first attempt cannot declare a predecessor")
    else:
        expected = {
            "attempt_number",
            "protocol_sha256",
            "final_tip_sha256",
            "disposition",
            "reason_codes",
            "last_observed_at",
        }
        if not isinstance(predecessor, dict) or set(predecessor) != expected:
            raise ValueError("successor attempt requires a complete predecessor")
        previous_number = predecessor["attempt_number"]
        if (
            isinstance(previous_number, bool)
            or not isinstance(previous_number, int)
            or previous_number != number - 1
        ):
            raise ValueError("attempt lineage predecessor number must be consecutive")
        for field in ("protocol_sha256", "final_tip_sha256"):
            digest = predecessor[field]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(f"attempt lineage predecessor {field} is invalid")
        disposition = predecessor["disposition"]
        if not isinstance(disposition, str) or disposition not in _ATTEMPT_DISPOSITIONS:
            raise ValueError(
                "attempt lineage predecessor disposition is not non-economic"
            )
        reasons = predecessor["reason_codes"]
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason) is None
            for reason in reasons
        ):
            raise ValueError("attempt lineage predecessor reason codes are invalid")
        if not set(reasons).issubset(_NON_ECONOMIC_ATTEMPT_REASONS):
            raise ValueError(
                "attempt lineage predecessor reason codes must be non-economic"
            )
        if reasons != sorted(set(reasons)) or not reasons:
            raise ValueError("attempt lineage predecessor reason codes are invalid")
        last_observed = predecessor["last_observed_at"]
        if last_observed is not None:
            if not isinstance(last_observed, str):
                raise ValueError("attempt lineage predecessor timestamp is invalid")
            parsed = timestamp(last_observed)
            if parsed.isoformat() != last_observed:
                raise ValueError(
                    "attempt lineage predecessor timestamp is not canonical"
                )
            if sealed_at is not None:
                seal = require_aware_datetime(sealed_at, field="paper seal").astimezone(
                    UTC
                )
                if parsed > seal:
                    raise ValueError(
                        "attempt lineage predecessor is newer than the seal"
                    )
                if (
                    "observation_gap" in reasons
                    and (seal - parsed).total_seconds() <= 180
                ):
                    raise ValueError(
                        "attempt lineage observation gap has not exceeded 180 seconds"
                    )
        elif "observation_gap" in reasons:
            raise ValueError(
                "attempt lineage observation gap requires a last observation time"
            )
        if version_three:
            _validate_predecessor_root(
                value["predecessor_root"], predecessor, sealed_at=sealed_at
            )
    # The shared classifier has the last word for every lineage, including the v2
    # rule that only attempt 1 may use the v2 label.
    problems = lineage_structure_problems(value)
    if problems:
        raise ValueError("attempt lineage is not canonical: " + "; ".join(problems))
    return deepcopy(value)


def validate_attempt_lineage_v4(
    value: object,
    *,
    sealed_at: datetime | None = None,
) -> dict[str, Any]:
    """Validate the attempt-3 lineage v4 (attempt 2 as an INVALID predecessor).

    The shared classifier decides the canonical form; this adds the seal-time ordering
    rules. The record's identity values are read from the attempt-2 root by
    ``attempt3`` before sealing and re-read on every collector entry; the fields stay
    caller assertions in the sense of the other lineage versions.
    """
    key_problems = plain_json_key_problems(value)
    if key_problems:
        raise ValueError(
            "attempt lineage keys are not plain JSON strings: "
            + "; ".join(key_problems)
        )
    if (
        type(value) is not dict
        or type(value.get("schema")) is not str
        or value["schema"] != LINEAGE_V4_SCHEMA
    ):
        raise ValueError("attempt lineage v4 schema is required")
    problems = lineage_structure_problems(value)
    if problems:
        raise ValueError("attempt lineage is not canonical: " + "; ".join(problems))
    claimed = v4_identity(value)
    assert claimed is not None
    identity = claimed[0]
    if sealed_at is not None:
        seal = require_aware_datetime(sealed_at, field="paper seal").astimezone(UTC)
        if timestamp(identity.last_observed_at) > seal:
            raise ValueError("attempt lineage predecessor is newer than the seal")
    return deepcopy(value)


def _economic_screen_fields() -> dict[str, Any]:
    """Every economic acceptance field, shared by screens v2 and v3 byte for byte."""
    return dict(
        required_fill_coverage=[dict(row) for row in REQUIRED_FILL_COVERAGE],
        duration_days=90,
        block_days=30,
        block_count=3,
        minimum_seal_notice_seconds=300,
        comparison="cash_zero_interest",
        positive_net_profit=True,
        positive_each_block=True,
        net_profit_exceeds_recorded_fees=True,
        maximum_drawdown_exclusive=0.1,
        nonzero_funding_each_block=True,
        complete_coverage_and_actual_flat=True,
        production_eligible=False,
    )


def screen_plan(*, attempt_lineage: dict[str, Any]) -> dict[str, Any]:
    """Return the fixed economic screen with its immutable attempt lineage."""
    lineage = validate_attempt_lineage(attempt_lineage)
    return dict(
        schema="carry_paper_screen_v2",
        attempt_lineage=lineage,
        **_economic_screen_fields(),
    )


def operational_quality_clause() -> dict[str, Any]:
    """The only clause screen v3 changes: registered collection failures are allowed.

    Screen v2 rejected any failed capture or skipped slot. Screen v3 accepts a failed
    capture or a skipped slot only when the collector registered it, it has a registered
    class, and the three skip budgets and the per-slot retry allowance hold; every
    terminal marker (Class D, Class I, budget breach B) still rejects. All values are
    the pre-registered collection-policy constants.
    """
    return dict(
        failed_capture_rejects_screen=False,
        unregistered_failed_capture_rejects_screen=True,
        terminal_failure_classes=["B", "D", "I"],
        retries_per_slot=RETRIES_PER_SLOT,
        retry_delay_seconds=RETRY_DELAY_SECONDS,
        retry_start_window_seconds=RETRY_START_WINDOW_SECONDS,
        max_failed_captures_per_slot=MAX_FAILED_CAPTURES_PER_SLOT,
        max_consecutive_skipped_slots=MAX_CONSECUTIVE_SKIPPED_SLOTS,
        max_skipped_slots_per_hour=MAX_SKIPPED_SLOTS_PER_HOUR,
        max_skipped_slots_total=MAX_SKIPPED_SLOTS_TOTAL,
        skip_window_slots=SKIP_WINDOW_SLOTS,
        cadence_seconds=CADENCE_SECONDS,
        frozen_gap_limit_seconds=FROZEN_GAP_LIMIT_SECONDS,
        rules_hold_seconds=RULES_HOLD_SECONDS,
        skip_reasons=list(SKIP_REASONS),
        attempt_purposes=list(ATTEMPT_PURPOSES),
        report_skip_count_and_funding_proximity=True,
        claim_boundary="retained_slots_only",
    )


def screen_plan_v3(*, attempt_lineage: dict[str, Any]) -> dict[str, Any]:
    """Screen v3: the v2 economic screen plus the registered collection-quality clause."""
    lineage = validate_attempt_lineage_v4(attempt_lineage)
    return dict(
        schema=SCREEN_V3_SCHEMA,
        attempt_lineage=lineage,
        **_economic_screen_fields(),
        operational_quality=operational_quality_clause(),
    )


def seal_paper_study(
    root: str | Path,
    *,
    start_at: datetime,
    attempt_lineage: dict[str, Any],
    lineage_seal_token: object | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> str:
    """Seal a screen. Lineage v3 is accepted only from ``attempt2``, v4 from ``attempt3``.

    A v4 lineage seals ``carry_paper_screen_v3``; every other lineage seals
    ``carry_paper_screen_v2``.

    A v3 ``predecessor_root`` record asserts a root read (``available``) or a recorded
    statement (``unavailable``) that only ``attempt2`` produces after its own checks,
    so this generic function and ``paper.cli seal`` refuse it unless given the
    module-private token. The token is a caller contract, not authentication: every
    path that starts collection (``attempt2 launch``, ``PaperCollector``, ``cli run``,
    ``cli supervise``) re-reads an ``available`` root before the start.
    """
    start = require_aware_datetime(start_at, field="paper start").astimezone(UTC)
    now = require_aware_datetime(clock(), field="paper seal").astimezone(UTC)
    if start < now + timedelta(minutes=5):
        raise ValueError("paper study requires at least five minutes before start")
    if (
        not plain_json_key_problems(attempt_lineage)
        and type(attempt_lineage) is dict
        and type(attempt_lineage.get("schema")) is str
        and attempt_lineage["schema"] == LINEAGE_V4_SCHEMA
    ):
        if lineage_seal_token is not _ATTEMPT3_SEAL_TOKEN:
            raise ValueError("attempt lineage v4 can only be sealed through attempt3")
        lineage_v4 = validate_attempt_lineage_v4(attempt_lineage, sealed_at=now)
        return seal_collection(
            root,
            settings=PaperSettings(start_at=start, close_at=start + timedelta(days=90)),
            research_plan=screen_plan_v3(attempt_lineage=lineage_v4),
            lineage_seal_token=lineage_seal_token,
            clock=lambda: now,
        )
    lineage = validate_attempt_lineage(attempt_lineage, sealed_at=now)
    if (
        lineage["schema"] == _ATTEMPT_LINEAGE_SCHEMA_V3
        and lineage_seal_token is not _ATTEMPT2_SEAL_TOKEN
    ):
        raise ValueError("attempt lineage v3 can only be sealed through attempt2")
    return seal_collection(
        root,
        settings=PaperSettings(start_at=start, close_at=start + timedelta(days=90)),
        research_plan=screen_plan(attempt_lineage=lineage),
        lineage_seal_token=lineage_seal_token,
        clock=lambda: now,
    )


def collect_until_finished(
    collector: PaperCollector,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
    publish: Callable[[dict[str, Any]], None],
    heartbeat: Callable[[str, dict[str, Any] | None], None] | None = None,
    heartbeat_interval_seconds: float = 60,
) -> dict[str, Any]:
    """Do not retry exceptions, fabricate missed slots, or extend the deadline.

    ``heartbeat`` is an operational liveness hook only; it sees no market data, can
    not alter a slot, and a failure to write it halts the collector like any error.
    """
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat interval must be positive")
    end = collector.close_at + timedelta(seconds=180)
    previous: datetime | None = None
    last_status: dict[str, Any] | None = None

    def beat(state: str) -> None:
        if heartbeat is not None:
            heartbeat(state, last_status)

    def read_clock() -> datetime:
        nonlocal previous
        at = require_aware_datetime(clock(), field="cadence clock")
        if previous is not None and at < previous:
            raise ValueError("cadence clock moved backwards")
        previous = at
        return at

    beat("starting")
    while True:
        read_clock()
        beat("cycle_started")
        result = collector.cycle()
        if result["status"]["quality_failures"] and result["status"]["terminal_flat"]:
            result = dict(result, phase="rejected")
        publish(result)
        last_status = result["status"]
        if result["phase"] in {"finished", "rejected"}:
            beat(result["phase"])
            return result
        beat("cycle_completed")
        now = read_clock()
        elapsed = (now - collector.start_at).total_seconds()
        slot = max(0, math.floor(elapsed / CADENCE_SECONDS) + 1)
        due = min(end, collector.start_at + timedelta(seconds=slot * CADENCE_SECONDS))
        while (remaining := (due - read_clock()).total_seconds()) > 0:
            sleep(min(60, heartbeat_interval_seconds, remaining))
            beat("sleeping")


def collection_status(
    root: str | Path, *, expected_protocol_sha256: str, read_only: bool = False
) -> dict[str, Any]:
    """Operational status verifies the chain, not raw-source financial replay.

    ``read_only`` opens the journal database without write access (no hot-journal
    recovery), for verifying preserved evidence.
    """
    import json

    journal = PaperJournal(
        root, expected_protocol_sha256=expected_protocol_sha256, read_only=read_only
    )
    records = journal.events()
    if records:
        status = records[-1]["event"]["payload"]["result"]
        tip = records[-1]["sha256"]
    else:
        config = json.loads((journal.root / "protocol.json").read_bytes())["protocol"][
            "settings"
        ]
        for key in ("start_at", "close_at"):
            config[key] = datetime.fromisoformat(config[key])
        status = PaperAccount(PaperSettings(**config)).status()
        tip = expected_protocol_sha256
    return dict(
        schema="paper_operational_status_v1",
        protocol_sha256=expected_protocol_sha256,
        verification="journal_chain_only",
        economic_decision="NOT_EVALUATED",
        production_eligible=False,
        collection_failure=(journal.root / "collection-failure.json").exists(),
        events=len(records),
        tip=tip,
        status=status,
    )
