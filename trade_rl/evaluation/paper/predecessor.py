"""Attempt 1's recorded identity and the re-verification of a recorded predecessor root.

Import-light on purpose: both ``attempt2`` (plan, seal, launch) and the generic
collector (``PaperCollector``, ``cli run``/``supervise``) use it, so a lineage v3
``available`` record is re-read the same way on every path that starts collection.
It reads a preserved root only through the existing v1 journal reader, opened
read-only (no hot-journal recovery), and never writes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from trade_rl.evaluation.paper.account import timestamp

ATTEMPT_ONE_PROTOCOL_SHA256 = (
    "843467870d85d0b085e65cf904e9d458287c14fc1e31615b9aaf77029f0669b7"
)
ATTEMPT_ONE_FINAL_TIP_SHA256 = (
    "1a84746aa875ee5a21b72944a5101c2ffb5b00860c9d085a1ccb1fe780d5836b"
)
ATTEMPT_ONE_LAST_OBSERVED_AT = "2026-09-17T23:11:01.134389+00:00"
ATTEMPT_ONE_EVENT_COUNT = 188
LINEAGE_V3_SCHEMA = "carry_paper_attempt_lineage_v3"
ROOT_READER = "paper_journal_protocol_v1"
PREDECESSOR_ROOT_STATUSES = ("available", "unavailable")
# The public record of the fresh result-blind review (2026-09-27/28) that verified
# the attempt-1 values above against its root while that root still existed.
LINEAGE_RECORD_DOCUMENT = "docs/research/current-status.md"
LINEAGE_RECORD_COMMIT = "b9f1ca65b26dcff7acbb089c7643e95702d474c9"
# Verbatim chat statement (Japanese). It says: "I do not know that data. I also do not
# know where the saved results are, so you may collect new data."
# Two lines in the chat message, joined by a single newline (exact text supplied by
# the planner from the chat of 2026-10-30).
USER_STATEMENT_QUOTE = (
    "そのデータはしらない。結果保存もどこにあるわからないから、\n新しくとっていいよ"
)
USER_STATEMENT_DATE = "2026-10-30"
USER_STATEMENT_STATES = (
    "attempt_1_data_location_unknown",
    "attempt_1_result_storage_location_unknown",
    "new_data_collection_permitted",
)
USER_STATEMENT_DOES_NOT_ATTEST = (
    "predecessor_lineage_values",
    "absence_of_an_earlier_attempt",
)
UNAVAILABLE_NOT_ESTABLISHED = (
    "predecessor_protocol_digest",
    "predecessor_final_tip",
    "predecessor_observation_gap",
    "absence_of_an_earlier_unreported_attempt",
)
UNAVAILABLE_REVIEW_RULE = (
    "Attempt 1's evidence root is lost; its lineage rests on a prior recorded review "
    "and was not re-verified against the root at seal time. The review that resolves "
    "the preserved root cannot be performed. It is replaced, for this case only, by a "
    "fresh result-blind review of the lineage documentation and the pull request that "
    "introduced this rule. That replacement review cannot confirm the predecessor "
    "protocol digest, final tip or observation gap, and it does not exclude an "
    "earlier unreported attempt."
)
# Module-private capability for sealing lineage v3; ``attempt2`` is its only user.
_ATTEMPT2_SEAL_TOKEN = object()


def predecessor_verification() -> dict[str, Any]:
    """What a successful v1 read of attempt 1's root proves; fixed by the constants."""
    return {
        "reader": ROOT_READER,
        "event_count": ATTEMPT_ONE_EVENT_COUNT,
        "protocol_sha256": ATTEMPT_ONE_PROTOCOL_SHA256,
        "final_tip_sha256": ATTEMPT_ONE_FINAL_TIP_SHA256,
        "last_observed_at": ATTEMPT_ONE_LAST_OBSERVED_AT,
    }


def canonical_root_record(status: str, root_path: str | None = None) -> dict[str, Any]:
    """The only ``predecessor_root`` record each status may carry."""
    if status == "available":
        if not root_path:
            raise ValueError("an available predecessor root records its path")
        return {
            "status": "available",
            "verification": dict(predecessor_verification(), root_path=root_path),
        }
    if status == "unavailable":
        return {
            "status": "unavailable",
            "re_verified_at_seal": False,
            "lineage_source": {
                "kind": "prior_result_blind_review_record",
                "document": LINEAGE_RECORD_DOCUMENT,
                "commit": LINEAGE_RECORD_COMMIT,
            },
            "user_statement": {
                "quote": USER_STATEMENT_QUOTE,
                "date": USER_STATEMENT_DATE,
                "channel": "chat",
                "states": list(USER_STATEMENT_STATES),
                "does_not_attest": list(USER_STATEMENT_DOES_NOT_ATTEST),
            },
            "not_established": list(UNAVAILABLE_NOT_ESTABLISHED),
            "review_rule": UNAVAILABLE_REVIEW_RULE,
        }
    raise ValueError("predecessor root status must be available or unavailable")


def canonical_lineage(status: str, root_path: str | None = None) -> dict[str, Any]:
    """The attempt-2 lineage v3 for a predecessor root status."""
    return {
        "schema": LINEAGE_V3_SCHEMA,
        "attempt_number": 2,
        "predecessor": {
            "attempt_number": 1,
            "protocol_sha256": ATTEMPT_ONE_PROTOCOL_SHA256,
            "final_tip_sha256": ATTEMPT_ONE_FINAL_TIP_SHA256,
            "disposition": "invalidated",
            "reason_codes": ["observation_gap"],
            "last_observed_at": ATTEMPT_ONE_LAST_OBSERVED_AT,
        },
        "predecessor_root": canonical_root_record(status, root_path),
    }


def require_seal_token(research_plan: object, token: object) -> None:
    """Refuse to seal any research plan that carries a lineage v3 without the token."""
    lineage = (
        research_plan.get("attempt_lineage")
        if isinstance(research_plan, dict)
        else None
    )
    if (
        isinstance(lineage, dict)
        and lineage.get("schema") == LINEAGE_V3_SCHEMA
        and token is not _ATTEMPT2_SEAL_TOKEN
    ):
        raise ValueError("attempt lineage v3 can only be sealed through attempt2")


def read_predecessor_status(
    root: str | Path, *, expected_protocol_sha256: str, read_only: bool = True
) -> dict[str, Any]:
    """The v1 journal reader's operational status (default reader for verification)."""
    # Lazy: ``operations`` imports the collector, which imports this module.
    from trade_rl.evaluation.paper.operations import collection_status

    return collection_status(
        root, expected_protocol_sha256=expected_protocol_sha256, read_only=read_only
    )


def verify_predecessor_root(
    root: str | Path,
    *,
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> list[str]:
    """Read attempt 1's root with the v1 journal reader; return failed reasons.

    The database is opened read-only (no hot-journal recovery), so a preserved root is
    never mutated; a root with a hot journal fails verification instead.
    """
    path = Path(root).absolute()
    if path.is_symlink() or not path.is_dir():
        return ["predecessor_root_not_a_directory"]
    try:
        report = read(
            path,
            expected_protocol_sha256=ATTEMPT_ONE_PROTOCOL_SHA256,
            read_only=True,
        )
        observed = report["status"]["observed_at"]
        observed_at = None if observed is None else timestamp(observed)
        events, tip = report["events"], report["tip"]
    except Exception as error:
        return [f"predecessor_root_unreadable:{type(error).__name__}"]
    reasons = []
    if events != ATTEMPT_ONE_EVENT_COUNT:
        reasons.append("predecessor_root_event_count_differs")
    if tip != ATTEMPT_ONE_FINAL_TIP_SHA256:
        reasons.append("predecessor_root_tip_differs")
    if observed_at != timestamp(ATTEMPT_ONE_LAST_OBSERVED_AT):
        reasons.append("predecessor_root_last_observation_differs")
    return reasons


def available_root_problems(
    lineage: object,
    *,
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
    reverify: bool = True,
) -> list[str]:
    """Reasons a lineage v3 ``available`` record cannot be re-verified now.

    Anything that is not an ``available`` v3 record returns no reasons (canonical
    checks of ``unavailable`` and of the rest belong to ``attempt2``). With
    ``reverify=False`` only the presence of the recorded root path is required.
    """
    if not isinstance(lineage, dict) or lineage.get("schema") != LINEAGE_V3_SCHEMA:
        return []
    record = lineage.get("predecessor_root")
    if not isinstance(record, dict) or record.get("status") != "available":
        return []
    proof = record.get("verification")
    path = proof.get("root_path") if isinstance(proof, dict) else None
    if not isinstance(path, str) or not path:
        return ["predecessor_root_path_not_recorded"]
    if not reverify:
        return []
    return [
        f"predecessor_root_not_reverifiable:{problem}"
        for problem in verify_predecessor_root(path, read=read)
    ]


def _is_available(lineage: object) -> bool:
    if not isinstance(lineage, dict) or lineage.get("schema") != LINEAGE_V3_SCHEMA:
        return False
    record = lineage.get("predecessor_root")
    return isinstance(record, dict) and record.get("status") == "available"


def lineage_problems(
    lineage: object,
    *,
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
    reverify: bool = True,
) -> list[str]:
    """Reasons a lineage v3 is not one that a collector may enter.

    Any v3 lineage must equal the canonical record for its status exactly
    (``unavailable``), or the canonical record for its recorded root path
    (``available``), and an ``available`` root must also re-verify when ``reverify``.
    Lineages that are not v3 are not this module's business and return no reasons.
    """
    if not isinstance(lineage, dict) or lineage.get("schema") != LINEAGE_V3_SCHEMA:
        return []
    record = lineage.get("predecessor_root")
    status = record.get("status") if isinstance(record, dict) else None
    if status == "unavailable":
        expected = canonical_lineage("unavailable")
    elif status == "available":
        proof = record.get("verification") if isinstance(record, dict) else None
        path = proof.get("root_path") if isinstance(proof, dict) else None
        if not isinstance(path, str) or not path:
            return ["predecessor_root_path_not_recorded"]
        expected = canonical_lineage("available", path)
    else:
        return ["lineage_v3_not_canonical"]
    reasons = [] if lineage == expected else ["lineage_v3_not_canonical"]
    if status == "available":
        reasons += available_root_problems(lineage, read=read, reverify=reverify)
    return reasons


def journal_has_events(root: Path) -> bool:
    """Whether a collection root already holds a journal event (opened read-only)."""
    path = Path(root) / "journal.sqlite"
    if path.is_symlink() or not path.is_file():
        return False
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
    try:
        return connection.execute("SELECT 1 FROM events LIMIT 1").fetchone() is not None
    finally:
        connection.close()


def collection_lineage_problems(
    research_plan: object,
    *,
    root: Path,
    start_at: datetime,
    now: Callable[[], datetime],
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> list[str]:
    """Generic collector gate: re-verify a v3 ``available`` lineage like ``launch``.

    The recorded root is read again before the start or while the root has no journal
    event yet. A restart of a collector that already recorded events (after the start)
    does not need the old root, exactly like a supervisor relaunch in ``launch``. If
    the journal cannot be inspected the root is re-read.
    """
    if not isinstance(research_plan, dict):
        return []
    lineage = research_plan.get("attempt_lineage")
    if not _is_available(lineage):
        # Nothing to re-read; the clock and journal are not touched. A v3
        # ``unavailable`` record must still be the canonical one.
        return lineage_problems(lineage)
    try:
        ran = journal_has_events(root)
    except sqlite3.Error:
        ran = False
    return lineage_problems(
        lineage, read=read, reverify=not (now() >= start_at and ran)
    )
