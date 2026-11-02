"""Fail-closed decision for exact-head software/model source review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

_REVIEW_MARKER = "<!-- independent-research-review-audit -->"
_APPROVED_DISPOSITION = "### Disposition: APPROVED"
_BLOCKED_DISPOSITION = "### Disposition: BLOCKED"
_SUBMITTED_REVIEW_STATES = {"APPROVED", "CHANGES_REQUESTED", "COMMENTED"}
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RECEIPT_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_SCHEMA = "independent_model_review_v1"
_RECEIPT_FENCE = "independent-review-receipt"
_UNVERIFIED_SOFTWARE_RECEIPT_SCHEMA = "software_source_review_v1"
_SOFTWARE_RECEIPT_SCHEMA = "software_source_review_v2"
_SOFTWARE_PAYLOAD_FIELDS = {
    "schema",
    "head_sha",
    "publisher_user_id",
    "scope",
    "disposition",
    "reviewer_model_id",
    "review_result",
    "blocking_findings",
    "report",
}
_PAYLOAD_FIELDS = {
    "schema",
    "head_sha",
    "publisher_user_id",
    "scope",
    "disposition",
    "implementation",
    "reviewer",
    "report",
}
_MODEL_PATTERN = re.compile(r"[a-z0-9]+(?:[-._][a-z0-9]+)*")
_TASK_PATTERN = re.compile(r"[a-z0-9/][a-z0-9._:/-]{0,255}")
_UNKNOWN_IDENTITIES = {
    "unknown",
    "notestablished",
    "unavailable",
    "none",
    "null",
    "pending",
    "na",
    "tbd",
}
_REPORT_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FENCE_CLOSE = re.compile(r"^ {0,3}(`+|~+)[ \t]*$")
_HTML_TAG = re.compile(r"</?([a-z][a-z0-9:-]*)(?=[ \t>/]|$)|<[!?]", re.I)
_HTML_LITERAL_NAMES = {"pre", "code", "script", "style", "textarea"}


class ReviewOutcome(StrEnum):
    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"
    PENDING = "PENDING"


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _positive_id(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _github_user_id(value: object) -> int | None:
    user = _mapping(value)
    return None if user is None else _positive_id(user.get("id"))


def _review_content(body: str) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    receipts: list[str] = []
    receipt_lines: list[str] | None = None
    html_tags: list[str] = []
    html_comment = False
    html_opaque = False
    unsupported_html = False
    fence_character: str | None = None
    fence_length = 0
    for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if fence_character is None:
            if html_comment:
                html_comment = "-->" not in line
                continue
            if html_opaque:
                html_opaque = bool(line.strip(" \t"))
                continue
            indentation = line[: len(line) - len(line.lstrip(" \t"))]
            if len(indentation.expandtabs(4)) >= 4:
                continue
            opening = _FENCE_OPEN.match(line) if not html_tags else None
            if opening is not None:
                fence = opening.group(1)
                fence_character = fence[0]
                fence_length = len(fence)
                if line[opening.end() :].strip(" \t") == _RECEIPT_FENCE:
                    receipt_lines = []
                continue
            if not html_tags and line.strip(" \t") == _REVIEW_MARKER:
                lines.append(_REVIEW_MARKER)
                continue
            if "<!--" in line:
                unsupported_html = True
                html_comment = "-->" not in line.split("<!--", 1)[1]
                continue
            tags = list(_HTML_TAG.finditer(line))
            if html_tags or tags:
                unsupported_html = True
                literal_line = bool(html_tags)
                for tag in tags:
                    name = (tag.group(1) or "").casefold()
                    if name not in _HTML_LITERAL_NAMES:
                        continue
                    literal_line = True
                    if line[tag.start() + 1] != "/":
                        html_tags.append(name)
                    elif html_tags and html_tags[-1] == name:
                        html_tags.pop()
                if not literal_line:
                    html_opaque = line.lstrip(" \t").startswith("<")
                continue
            lines.append(line.strip(" \t"))
            continue
        closing = _FENCE_CLOSE.match(line)
        if closing is not None:
            fence = closing.group(1)
            if fence[0] == fence_character and len(fence) >= fence_length:
                if receipt_lines is not None:
                    receipts.append("\n".join(receipt_lines))
                    receipt_lines = None
                fence_character = None
                fence_length = 0
                continue
        if receipt_lines is not None:
            receipt_lines.append(line)
    complete = fence_character is None and not unsupported_html
    return lines, receipts if complete else []


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _canonical_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _model_task_identity(value: object) -> tuple[str, str] | None:
    identity = _mapping(value)
    if identity is None or set(identity) != {"model_id", "task_id"}:
        return None
    model_id, task_id = identity.get("model_id"), identity.get("task_id")
    if (
        not isinstance(model_id, str)
        or not _MODEL_PATTERN.fullmatch(model_id)
        or not isinstance(task_id, str)
        or not _TASK_PATTERN.fullmatch(task_id)
        or task_id.endswith("/")
    ):
        return None
    model_key = re.sub(r"[-._]", "", model_id.casefold().strip())
    task_key = task_id.casefold().strip()
    if (
        model_key in _UNKNOWN_IDENTITIES
        or re.sub(r"[-._]", "", task_key) in _UNKNOWN_IDENTITIES
    ):
        return None
    return model_key, task_key


def _valid_software_payload(
    record: Mapping[str, Any], *, head_sha: str, publisher_id: int
) -> bool:
    """Validate a lightweight exact-head software/model source review.

    The reviewer model is publisher-attested metadata. This schema deliberately
    does not claim provider-run provenance or model independence.
    """

    if set(record) != _SOFTWARE_PAYLOAD_FIELDS:
        return False
    model = record.get("reviewer_model_id")
    report = record.get("report")
    return (
        record.get("schema") == _SOFTWARE_RECEIPT_SCHEMA
        and record.get("head_sha") == head_sha
        and bool(_SHA_PATTERN.fullmatch(head_sha))
        and _positive_id(record.get("publisher_user_id")) == publisher_id
        and record.get("scope") == "software-only"
        and record.get("disposition") == "APPROVED"
        and record.get("review_result") == "PASS"
        and type(record.get("blocking_findings")) is int
        and record.get("blocking_findings") == 0
        and isinstance(model, str)
        and bool(_MODEL_PATTERN.fullmatch(model))
        and re.sub(r"[-._]", "", model.casefold()) not in _UNKNOWN_IDENTITIES
        and isinstance(report, str)
        and len(report.strip()) >= 80
        and _REPORT_CONTROL.search(report) is None
    )


def _valid_payload(payload: object, *, head_sha: str, publisher_id: int) -> bool:
    record = _mapping(payload)
    if record is None:
        return False
    if record.get("schema") == _UNVERIFIED_SOFTWARE_RECEIPT_SCHEMA:
        # v1 mixed unverified job/diff claims into an approval receipt. Keep it
        # fail-closed rather than pretending those fields are provider evidence.
        return False
    if record.get("schema") == _SOFTWARE_RECEIPT_SCHEMA:
        return _valid_software_payload(
            record, head_sha=head_sha, publisher_id=publisher_id
        )
    if set(record) != _PAYLOAD_FIELDS:
        return False
    report = record.get("report")
    if (
        record.get("schema") != _RECEIPT_SCHEMA
        or record.get("head_sha") != head_sha
        or not _SHA_PATTERN.fullmatch(head_sha)
        or _positive_id(record.get("publisher_user_id")) != publisher_id
        or record.get("scope") != "software-only"
        or not isinstance(record.get("disposition"), str)
        or record.get("disposition") not in {"APPROVED", "BLOCKED"}
        or not isinstance(report, str)
        or not report.strip()
        or _REPORT_CONTROL.search(report)
    ):
        return False
    implementation = record.get("implementation")
    reviewer = _model_task_identity(record.get("reviewer"))
    if not isinstance(implementation, list) or not implementation or reviewer is None:
        return False
    task_ids: set[str] = set()
    for item in implementation:
        identity = _model_task_identity(item)
        if identity is None:
            return False
        model_id, task_id = identity
        if model_id == reviewer[0] or task_id == reviewer[1] or task_id in task_ids:
            return False
        task_ids.add(task_id)
    return True


def _valid_receipt(receipts: list[str], *, head_sha: str, publisher_id: int) -> bool:
    if len(receipts) != 1:
        return False
    try:
        envelope = _mapping(
            json.loads(
                receipts[0],
                object_pairs_hook=_unique_json_pairs,
                parse_constant=_reject_json_constant,
            )
        )
    except ValueError:
        return False
    if envelope is None or set(envelope) != {"payload", "sha256"}:
        return False
    payload, digest = envelope.get("payload"), envelope.get("sha256")
    if (
        not isinstance(digest, str)
        or not _RECEIPT_SHA_PATTERN.fullmatch(digest)
        or not _valid_payload(payload, head_sha=head_sha, publisher_id=publisher_id)
    ):
        return False
    record = _mapping(payload)
    assert record is not None
    return (
        record.get("disposition") == "APPROVED"
        and hashlib.sha256(_canonical_payload(record)).hexdigest() == digest
    )


def resolve_review_outcome(
    pull_request: object,
    reviews: object,
    *,
    expected_head_sha: str,
) -> ReviewOutcome:
    """Evaluate exact-head authenticated reviews with explicit assurance scope."""
    pull = _mapping(pull_request)
    if pull is None or not _SHA_PATTERN.fullmatch(expected_head_sha):
        return ReviewOutcome.PENDING
    head = _mapping(pull.get("head"))
    if head is None or head.get("sha") != expected_head_sha:
        return ReviewOutcome.PENDING
    author_id = _github_user_id(pull.get("user"))
    if (
        author_id is None
        or not isinstance(reviews, Sequence)
        or isinstance(reviews, (str, bytes))
    ):
        return ReviewOutcome.PENDING

    latest: dict[int, tuple[int, bool]] = {}
    review_ids: set[int] = set()
    ambiguous = False
    blocked = False
    for item in reviews:
        review = _mapping(item)
        if review is None or review.get("commit_id") != expected_head_sha:
            continue
        reviewer_id = _github_user_id(review.get("user"))
        review_id = _positive_id(review.get("id"))
        if reviewer_id is None or review_id is None:
            continue
        ambiguous = ambiguous or review_id in review_ids
        review_ids.add(review_id)
        body = review.get("body")
        if not isinstance(body, str):
            continue
        lines, receipts = _review_content(body)
        if _REVIEW_MARKER not in lines:
            continue
        dispositions = [line for line in lines if line.startswith("### Disposition:")]
        state = review.get("state")
        if not isinstance(state, str):
            state = ""
        if _BLOCKED_DISPOSITION in dispositions and state in _SUBMITTED_REVIEW_STATES:
            blocked = True
        approved = (
            lines.count(_REVIEW_MARKER) == 1
            and dispositions == [_APPROVED_DISPOSITION]
            and state in {"APPROVED", "COMMENTED"}
            and _valid_receipt(
                receipts, head_sha=expected_head_sha, publisher_id=reviewer_id
            )
        )
        if reviewer_id not in latest or review_id > latest[reviewer_id][0]:
            latest[reviewer_id] = review_id, approved

    if blocked:
        return ReviewOutcome.BLOCKED
    if not ambiguous and any(approved for _, approved in latest.values()):
        return ReviewOutcome.APPROVED
    return ReviewOutcome.PENDING


def _read_json(path: str) -> object:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_pairs,
        parse_constant=_reject_json_constant,
    )


def _render_receipt(payload: object) -> str:
    record = _mapping(payload)
    if record is None:
        raise ValueError("Receipt payload must be an object")
    head_sha = record.get("head_sha")
    publisher_id = _positive_id(record.get("publisher_user_id"))
    if (
        not isinstance(head_sha, str)
        or publisher_id is None
        or not _valid_payload(record, head_sha=head_sha, publisher_id=publisher_id)
    ):
        raise ValueError("Invalid software/model source review receipt")
    envelope = {
        "payload": dict(record),
        "sha256": hashlib.sha256(_canonical_payload(record)).hexdigest(),
    }
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{_REVIEW_MARKER}\n\n### Disposition: {record['disposition']}\n\n```{_RECEIPT_FENCE}\n{raw}\n```"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-json")
    parser.add_argument("--reviews-json")
    parser.add_argument("--expected-head")
    parser.add_argument("--render-payload-json")
    args = parser.parse_args(argv)
    evidence_args = (args.pull_json, args.reviews_json, args.expected_head)
    if args.render_payload_json:
        if any(evidence_args):
            parser.error("Receipt rendering cannot consume GitHub evidence arguments")
        try:
            print(_render_receipt(_read_json(args.render_payload_json)))
        except (OSError, ValueError) as error:
            print(f"Unable to render review receipt: {error}", file=sys.stderr)
            return 2
        return 0
    if not all(evidence_args):
        parser.error("--pull-json, --reviews-json and --expected-head are required")
    try:
        pull_request = _read_json(args.pull_json)
        reviews = _read_json(args.reviews_json)
    except (OSError, ValueError) as error:
        print(f"Unable to read GitHub review evidence: {error}", file=sys.stderr)
        return 2
    print(
        resolve_review_outcome(
            pull_request,
            reviews,
            expected_head_sha=args.expected_head,
        ).value
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
