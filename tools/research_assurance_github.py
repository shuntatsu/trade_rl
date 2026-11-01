"""Authenticated GitHub readback for bounded G0-G2 research assurance reviews.

This is deliberately an external read-only adapter; tools.agent_repo remains
network-free. The GitHub reviewer must be a distinct real principal, not the
author asserting independence in their own review packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.agent_repo.__main__ import _assurance_json_input
from tools.agent_repo.research_assurance import (
    AssuranceResult,
    TrustedReviewEvidence,
    assurance_digest,
    evaluate_assurance,
)

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_MAX_GH_BYTES = 262_144


def _block(record: Mapping[str, object], message: str) -> AssuranceResult:
    try:
        digest = assurance_digest(record)
    except (TypeError, ValueError):
        digest = ""
    return AssuranceResult("BLOCKED", False, digest, (message,))


def _object(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(context + " must be a JSON object")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _body_json(body: object) -> Mapping[str, object]:
    if not isinstance(body, str):
        raise ValueError("source PR review body must be canonical JSON text")
    if len(body.encode("utf-8")) > _MAX_GH_BYTES:
        raise ValueError("source review body exceeds size limit")

    def strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for name, value in pairs:
            if name in out:
                raise ValueError("duplicate source review key: " + name)
            out[name] = value
        return out

    def reject_constant(value: str) -> object:
        raise ValueError("nonfinite source review constant: " + value)

    obj = _object(
        json.loads(
            body, object_pairs_hook=strict_pairs, parse_constant=reject_constant
        ),
        "source PR review body",
    )
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    )
    if canonical != body:
        raise ValueError("source review body must match exact canonical bytes")
    return obj


def _pr_snapshot(
    payload: object,
    repository: str,
    pr_number: int,
    head: str,
) -> tuple[int, str]:
    pr = _object(payload, "GitHub PR")
    if pr.get("number") != pr_number or pr.get("state") != "open":
        raise ValueError("PR is not the exact open pull request")
    base = _object(pr.get("base"), "PR.base")
    branch = _object(pr.get("head"), "PR.head")
    if branch.get("sha") != head:
        raise ValueError("PR.head.sha differs from the reviewed implementation HEAD")
    for side in (base, branch):
        repo = _object(side.get("repo"), "PR branch repository")
        if repo.get("full_name") != repository:
            raise ValueError("PR base/head repository is not the requested repository")
    author = _object(pr.get("user"), "PR author")
    actor_id = _positive_int(author.get("id"), "PR author id")
    login = author.get("login")
    if not isinstance(login, str) or not login:
        raise ValueError("PR author login missing")
    return actor_id, login


def verify_github_review(
    record: Mapping[str, object],
    *,
    repository: str,
    pr_number: int,
    review_id: int,
    api_read: Callable[[str], object],
) -> AssuranceResult:
    """Inspect authenticated PR/review objects twice to reject head races."""
    if not _REPO.fullmatch(repository) or any(
        name in {".", ".."} for name in repository.split("/")
    ):
        raise ValueError("repository must be owner/name")
    _positive_int(pr_number, "pr_number")
    _positive_int(review_id, "review_id")

    packet = evaluate_assurance(record)
    if packet.status != "UNREVIEWED":
        return packet
    identity = _object(record.get("identity"), "packet.identity")
    head = identity.get("implementation_head")
    if not isinstance(head, str) or not _SHA40.fullmatch(head):
        return _block(record, "packet implementation HEAD is invalid")
    pr_url = f"repos/{repository}/pulls/{pr_number}"
    review_url = f"{pr_url}/reviews/{review_id}"
    try:
        pr_first = api_read(pr_url)
        author_id, author_login = _pr_snapshot(pr_first, repository, pr_number, head)
        source = _object(api_read(review_url), "GitHub PR review")
        if _positive_int(source.get("id"), "review.id") != review_id:
            raise ValueError("PR review ID mismatch")
        if source.get("commit_id") != head:
            raise ValueError("PR review commit_id differs from reviewed HEAD")
        if source.get("state") not in {"COMMENTED", "APPROVED", "CHANGES_REQUESTED"}:
            raise ValueError("PR review was not submitted or was dismissed")
        if (
            not isinstance(source.get("submitted_at"), str)
            or not source["submitted_at"]
        ):
            raise ValueError("PR review has no submission timestamp")
        reviewer = _object(source.get("user"), "PR reviewer")
        reviewer_id = _positive_int(reviewer.get("id"), "reviewer id")
        reviewer_login = reviewer.get("login")
        if not isinstance(reviewer_login, str) or not reviewer_login:
            raise ValueError("PR reviewer login is missing")
        if reviewer_id == author_id or reviewer_login.lower() == author_login.lower():
            raise ValueError(
                "same-author review does not establish reviewer independence"
            )

        body = source.get("body")
        review = _body_json(body)
        assert isinstance(body, str)  # Guarded by _body_json above.
        reviewed_identity = review.get("reviewer_identity")
        if reviewed_identity != reviewer_login:
            raise ValueError(
                "reviewer identity field does not match actual GitHub principal"
            )
        source_kind = TrustedReviewEvidence(
            review_body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            reviewer_identity=reviewer_login,
            author_identity=author_login,
            reviewed_commit_sha=head,
            reviewer_run_identity=str(review.get("reviewer_execution_identity", "")),
            source_kind="authenticated_pr_review",
            authenticated=True,
        )
        pr_final = api_read(pr_url)
        final_author_id, final_author_login = _pr_snapshot(
            pr_final, repository, pr_number, head
        )
        if final_author_id != author_id or final_author_login != author_login:
            raise ValueError("PR author identity changed across readback")
        return evaluate_assurance(record, review=review, authority=source_kind)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _block(record, str(error))


def _github_api_read(path: str) -> object:
    proc = subprocess.run(
        ["gh", "api", "--method", "GET", path],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0 or len(proc.stdout) > _MAX_GH_BYTES:
        raise ValueError("authenticated GitHub readback unavailable or oversized")
    try:
        return json.loads(proc.stdout.decode("utf-8", "strict"))
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("GitHub returned invalid JSON") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--review-id", required=True, type=int)
    parser.add_argument("--packet", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        packet = _assurance_json_input(Path.cwd(), str(args.packet))
        result = verify_github_review(
            packet,
            repository=args.repository,
            pr_number=args.pr,
            review_id=args.review_id,
            api_read=_github_api_read,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(
            json.dumps({"status": "BLOCKED", "errors": [str(error)]}), file=sys.stderr
        )
        return 2
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
