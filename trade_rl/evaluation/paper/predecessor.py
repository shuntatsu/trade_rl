"""Attempt 1's recorded identity and the re-verification of a recorded predecessor root.

Import-light on purpose: both ``attempt2`` (plan, seal, launch) and the generic
collector (``PaperCollector``, ``cli run``/``supervise``) use it, so a lineage v3
``available`` record is re-read the same way on every path that starts collection.
It reads a preserved root only through the existing v1 journal reader, opened
read-only (no hot-journal recovery), and never writes.
"""

from __future__ import annotations

import json
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
LINEAGE_V2_SCHEMA = "carry_paper_attempt_lineage_v2"
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
    """Refuse an unsupported or non-canonical lineage, and v3 without the token.

    An ordinary dict is required; allowing an overridden ``dict.get`` to hide
    the lineage would undermine the seal-time contract. The private token is
    still a caller contract, not an authentication credential. The structure check
    is the single classifier that every other gate also uses.
    """
    if type(research_plan) is not dict:
        raise ValueError("collection research plan must be a plain dict")
    lineage = research_plan.get("attempt_lineage")
    if lineage is None:
        return
    if (
        type(lineage) is dict
        and type(lineage.get("schema")) is str
        and lineage.get("schema") == LINEAGE_V3_SCHEMA
        and token is not _ATTEMPT2_SEAL_TOKEN
    ):
        raise ValueError("attempt lineage v3 can only be sealed through attempt2")
    problems = lineage_structure_problems(lineage)
    if problems:
        raise ValueError(
            "unsupported or non-canonical collection attempt lineage: "
            + "; ".join(problems)
        )


class ProtocolFormatError(ValueError):
    """``protocol.json`` is not the canonical bytes a sealed root contains."""


class DuplicateKeyError(ProtocolFormatError):
    """A protocol JSON object repeats a key (``json.loads`` would keep the last)."""


def load_canonical_protocol(raw: bytes) -> dict[str, Any]:
    """Parse ``protocol.json`` and refuse anything but its canonical form.

    Duplicate keys, extra or reordered top-level keys, NaN and non-canonical bytes are
    refused here, before the collector takes the lock, so no failure marker is written
    for that input. The store enforces the same canonical-bytes rule after the lock.
    """
    from trade_rl.artifacts import canonical_json_bytes

    value = loads_without_duplicate_keys(raw)
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ProtocolFormatError("protocol JSON is not canonical") from error
    if (
        type(value) is not dict
        or set(value) != {"schema", "protocol"}
        or canonical != raw
    ):
        raise ProtocolFormatError("protocol JSON is not canonical")
    return value


def loads_without_duplicate_keys(raw: bytes | str) -> Any:
    """``json.loads`` that refuses duplicate object keys instead of last-wins."""

    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise DuplicateKeyError("protocol JSON repeats an object key")
        return dict(pairs)

    return json.loads(raw, object_pairs_hook=reject)


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
) -> list[str]:
    """Re-read a v3 available predecessor root on *every* collector entry.

    A native journal event or heartbeat can be forged without prior root verification,
    so a post-start restart is not grounds to skip this check. Missing roots fail
    closed even after the attempt has begun.
    """
    if type(lineage) is not dict or lineage.get("schema") != LINEAGE_V3_SCHEMA:
        return []
    record = lineage.get("predecessor_root")
    if type(record) is not dict or record.get("status") != "available":
        return []
    proof = record.get("verification")
    path = proof.get("root_path") if type(proof) is dict else None
    if type(path) is not str or not path:
        return ["predecessor_root_path_not_recorded"]
    return [
        f"predecessor_root_not_reverifiable:{problem}"
        for problem in verify_predecessor_root(path, read=read)
    ]


def same_canonical_record(actual: object, expected: object) -> bool:
    """Compare JSON-shaped lineage records without Python's bool/int/float coercion.

    The collector reads a persisted protocol, so this must enforce the schema's
    exact JSON value types rather than Python ``dict.__eq__`` semantics.
    """
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return actual.keys() == expected.keys() and all(
            same_canonical_record(actual[key], canonical)
            for key, canonical in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(actual, list)
        return len(actual) == len(expected) and all(
            same_canonical_record(value, canonical)
            for value, canonical in zip(actual, expected, strict=True)
        )
    return actual == expected


_V2_KEYS = frozenset({"schema", "attempt_number", "predecessor"})


def lineage_structure_problems(lineage: object) -> list[str]:
    """The single lineage classifier: pure, no I/O, used by every gate.

    * ``None`` (no lineage) is allowed.
    * Only an exact ``dict`` with an exact ``str`` schema is considered.
    * The v2 label requires exactly the v2 key set, so a ``predecessor_root`` (or any
      other v3-only field) cannot ride under it, and only attempt 1 may use it
      (``attempt_number`` >= 2 must use v3).
    * The v3 label requires the type-strict canonical record for its status
      (``available`` for its recorded root path).
    * Any other schema, including spelling variants, is unsupported.
    """
    if lineage is None:
        return []
    if type(lineage) is not dict:
        return ["lineage_not_a_mapping"]
    schema = lineage.get("schema")
    if type(schema) is str and schema == LINEAGE_V2_SCHEMA:
        if set(lineage) != _V2_KEYS:
            return ["lineage_v2_unexpected_fields"]
        number = lineage.get("attempt_number")
        if type(number) is int and number >= 2:
            # Pre-existing v2 rule, tightened: v2 asserts predecessor values the
            # caller supplies, so only attempt 1 (no predecessor) may use the v2
            # label. Attempt 2 and later must use the canonical v3 record.
            return ["lineage_v2_attempt_requires_v3"]
        return []
    if type(schema) is not str or schema != LINEAGE_V3_SCHEMA:
        return ["lineage_schema_unsupported"]
    record = lineage.get("predecessor_root")
    status = record.get("status") if type(record) is dict else None
    if type(status) is str and status == "unavailable":
        expected = canonical_lineage("unavailable")
    elif type(status) is str and status == "available":
        proof = record.get("verification") if type(record) is dict else None
        path = proof.get("root_path") if type(proof) is dict else None
        if type(path) is not str or not path:
            return ["predecessor_root_path_not_recorded"]
        expected = canonical_lineage("available", path)
    else:
        return ["lineage_v3_not_canonical"]
    return (
        [] if same_canonical_record(lineage, expected) else ["lineage_v3_not_canonical"]
    )


def lineage_problems(
    lineage: object,
    *,
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> list[str]:
    """Structure problems, plus a re-read of a canonical v3 ``available`` root."""
    reasons = lineage_structure_problems(lineage)
    if not reasons:
        reasons += available_root_problems(lineage, read=read)
    return reasons


def collection_lineage_problems(
    research_plan: object,
    *,
    root: Path,
    start_at: datetime,
    now: Callable[[], datetime],
    read: Callable[..., dict[str, Any]] = read_predecessor_status,
) -> list[str]:
    """Reject malformed lineage before the lock, regardless of start or journal.

    The original root must remain independently verifiable for available v3
    lineage. A public engine can append gap events without this check, so events
    and heartbeats are never proof that a past read occurred.
    """
    # These arguments remain in the generic collector signature for existing
    # callers; time and own-journal state cannot grant lineage authority.
    _ = root, start_at, now
    if research_plan is None:
        return []
    if type(research_plan) is not dict:
        return ["research_plan_not_a_plain_dict"]
    return lineage_problems(research_plan.get("attempt_lineage"), read=read)
