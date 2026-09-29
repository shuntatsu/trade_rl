"""Fail-closed decision for an exact-head independent research review."""

from __future__ import annotations

import argparse
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
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FENCE_CLOSE = re.compile(r"^ {0,3}(`+|~+)[ \t]*$")


class ReviewOutcome(StrEnum):
    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"
    PENDING = "PENDING"


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _github_user_id(value: object) -> int | None:
    user = _mapping(value)
    if user is None:
        return None
    user_id = user.get("id")
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        return None
    return user_id


def _visible_review_lines(body: str) -> list[str]:
    lines: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in body.splitlines():
        if fence_character is None:
            opening = _FENCE_OPEN.match(line)
            if opening is not None:
                fence = opening.group(1)
                fence_character = fence[0]
                fence_length = len(fence)
                continue
            lines.append(line.strip())
            continue
        closing = _FENCE_CLOSE.match(line)
        if closing is not None:
            fence = closing.group(1)
            if fence[0] == fence_character and len(fence) >= fence_length:
                fence_character = None
                fence_length = 0
    return lines


def resolve_review_outcome(
    pull_request: object,
    reviews: object,
    *,
    expected_head_sha: str,
) -> ReviewOutcome:
    """Require an external exact-head review with one structured disposition."""
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

    approved = False
    blocked = False
    for item in reviews:
        review = _mapping(item)
        if review is None or review.get("commit_id") != expected_head_sha:
            continue
        reviewer_id = _github_user_id(review.get("user"))
        if reviewer_id is None or reviewer_id == author_id:
            continue
        body = review.get("body")
        if not isinstance(body, str):
            continue
        lines = _visible_review_lines(body)
        if _REVIEW_MARKER not in lines:
            continue
        dispositions = [
            line
            for line in lines
            if line in {_APPROVED_DISPOSITION, _BLOCKED_DISPOSITION}
        ]
        state = review.get("state")
        if _BLOCKED_DISPOSITION in dispositions and state in _SUBMITTED_REVIEW_STATES:
            blocked = True
        elif dispositions == [_APPROVED_DISPOSITION] and state == "APPROVED":
            approved = True

    if blocked:
        return ReviewOutcome.BLOCKED
    if approved:
        return ReviewOutcome.APPROVED
    return ReviewOutcome.PENDING


def _read_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-json", required=True)
    parser.add_argument("--reviews-json", required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args(argv)
    try:
        pull_request = _read_json(args.pull_json)
        reviews = _read_json(args.reviews_json)
    except (OSError, json.JSONDecodeError) as error:
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
