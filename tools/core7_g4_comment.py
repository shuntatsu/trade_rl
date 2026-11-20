"""Authenticate the owner G4 authorization comment for a Core7 MTF Tier D run.

``run_study`` only checks the *shape* of an :class:`AuthorizationReceipt` and its binding
to the URL and the protocol digest; it cannot know the comment is real. This module is
the tools-layer check that makes the receipt trustworthy to the extent GitHub's API is:
it fetches the comment read-only through the ``gh`` CLI (which holds its own
credentials; no token is ever taken from an argument, logged or printed), and verifies

* the URL shape (issue or pull request, ``#issuecomment-<id>``) and that the fetched
  comment's id, ``html_url`` and issue number match it exactly;
* the author is the repository owner by login **and** numeric id (pinned below, checked
  once against ``gh api users/shuntatsu --jq .id``) and is a ``User``;
* the body quotes the exact protocol digest and, once known, the exact dataset id the
  driver is about to load, each as a whole 64-hex token;
* the comment was not edited between the first fetch and a second one made immediately
  before ``run_study`` (identical body SHA-256 and ``updated_at``).

Every failure, including a network or ``gh`` failure, raises :class:`G4AuthError`; the
driver turns it into a distinct non-zero exit and never runs the study.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from trade_rl.evaluation.core7_mtf_study import protocol
from trade_rl.evaluation.core7_mtf_study.protocol import AuthorizationReceipt

REPOSITORY: Final = "shuntatsu/trade_rl"
OWNER_LOGIN: Final = "shuntatsu"
#: ``gh api users/shuntatsu --jq .id``, checked 2026-11-20 and pinned.
OWNER_ID: Final = 148511409
GH_TIMEOUT_SECONDS: Final = 30.0

#: Runs ``gh <args>`` and returns stdout; raises :class:`G4AuthError` on any failure.
GhRunner = Callable[[Sequence[str]], str]

_HEX64 = r"(?<![0-9a-fA-F]){}(?![0-9a-fA-F])"


class G4AuthError(RuntimeError):
    """The G4 comment could not be fetched or does not authorize this run."""


def default_gh(args: Sequence[str]) -> str:
    """Run the read-only ``gh`` CLI with a timeout; never echo its output on failure."""
    try:
        completed = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise G4AuthError(f"gh could not be run ({type(error).__name__})") from error
    if completed.returncode != 0:
        raise G4AuthError(f"gh failed (exit {completed.returncode})")
    return completed.stdout


@dataclass(frozen=True, slots=True)
class CommentSnapshot:
    """One verified fetch of the comment."""

    comment_id: int
    html_url: str
    login: str
    user_id: int
    body: str
    body_sha256: str
    updated_at: str
    fetched_at: str


def parse_g4_url(url: str) -> tuple[str, int, int]:
    """``(kind, issue_number, comment_id)`` of a well-formed G4 comment URL."""
    cleaned = protocol.validate_authorization_url(url)
    match = re.fullmatch(
        r"https://github\.com/shuntatsu/trade_rl/(issues|pull)/([0-9]+)#issuecomment-([0-9]+)",
        cleaned,
    )
    if match is None:  # pragma: no cover - validate_authorization_url pins the shape
        raise G4AuthError("authorization URL is malformed")
    return match.group(1), int(match.group(2)), int(match.group(3))


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise G4AuthError(f"comment {name} is not an integer")
    return value


def fetch_comment(url: str, gh: GhRunner = default_gh) -> CommentSnapshot:
    """Fetch the comment read-only and verify id, URL, issue number and owner."""
    cleaned = protocol.validate_authorization_url(url)
    _, number, comment_id = parse_g4_url(cleaned)
    raw = gh(["api", f"repos/{REPOSITORY}/issues/comments/{comment_id}"])
    fetched_at = datetime.now(UTC).isoformat()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise G4AuthError("gh returned a response that is not JSON") from error
    if not isinstance(payload, dict):
        raise G4AuthError("gh returned an unexpected response shape")
    if _require_int(payload.get("id"), "id") != comment_id:
        raise G4AuthError("fetched comment id does not match the URL")
    if payload.get("html_url") != cleaned:
        raise G4AuthError("fetched comment html_url does not match the URL")
    issue_url = payload.get("issue_url")
    if issue_url != f"https://api.github.com/repos/{REPOSITORY}/issues/{number}":
        raise G4AuthError("fetched comment issue number does not match the URL")
    user = payload.get("user")
    if not isinstance(user, dict):
        raise G4AuthError("comment has no user")
    if user.get("login") != OWNER_LOGIN or user.get("type") != "User":
        raise G4AuthError("comment author is not the repository owner")
    if _require_int(user.get("id"), "user id") != OWNER_ID:
        raise G4AuthError("comment author numeric id is not the pinned owner id")
    body = payload.get("body")
    updated_at = payload.get("updated_at")
    if not isinstance(body, str) or not isinstance(updated_at, str) or not updated_at:
        raise G4AuthError("comment body or updated_at is missing")
    return CommentSnapshot(
        comment_id=comment_id,
        html_url=cleaned,
        login=OWNER_LOGIN,
        user_id=OWNER_ID,
        body=body,
        body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        updated_at=updated_at,
        fetched_at=fetched_at,
    )


def _quotes(body: str, token: str) -> bool:
    return re.search(_HEX64.format(re.escape(token)), body) is not None


def require_protocol_digest(
    snapshot: CommentSnapshot, protocol_digest: str | None = None
) -> None:
    """The comment must quote the digest (default: the Core7 MTF protocol's)."""
    digest = protocol.protocol_digest() if protocol_digest is None else protocol_digest
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise G4AuthError("protocol digest is not a lowercase sha256 digest")
    if not _quotes(snapshot.body, digest):
        raise G4AuthError("comment does not quote the exact protocol digest")


def require_protocol_id(snapshot: CommentSnapshot, protocol_id: str) -> None:
    """The comment must quote ``protocol_id`` as a whole identifier token."""
    if re.fullmatch(r"[A-Za-z0-9_]+", protocol_id) is None:
        raise G4AuthError("protocol id is malformed")
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(protocol_id)}(?![A-Za-z0-9_])"
    if re.search(pattern, snapshot.body) is None:
        raise G4AuthError("comment does not quote the exact protocol id")


def require_dataset_id(snapshot: CommentSnapshot, dataset_id: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", dataset_id) is None:
        raise G4AuthError("dataset id is not a lowercase sha256 digest")
    if not _quotes(snapshot.body, dataset_id):
        raise G4AuthError("comment does not quote the exact dataset id")


def confirm_unedited(
    url: str,
    first: CommentSnapshot,
    gh: GhRunner = default_gh,
    *,
    protocol_digest: str | None = None,
) -> AuthorizationReceipt:
    """Re-fetch right before ``run_study``; require the identical body and edit time."""
    second = fetch_comment(url, gh)
    if (
        second.body_sha256 != first.body_sha256
        or second.updated_at != first.updated_at
        or second.body != first.body
    ):
        raise G4AuthError("comment was edited after it was first fetched")
    return AuthorizationReceipt(
        authorization_url=second.html_url,
        comment_id=second.comment_id,
        author_login=second.login,
        author_id=second.user_id,
        body_sha256=second.body_sha256,
        fetched_at=second.fetched_at,
        quoted_protocol_digest=(
            protocol.protocol_digest() if protocol_digest is None else protocol_digest
        ),
    )


def g4_comment_template(dataset_id: str) -> str:
    """The minimal body text that passes: the digest and the dataset id as whole tokens."""
    return (
        "G4 authorization: Core7 MTF Tier D replay\n"
        f"protocol_id: {protocol.PROTOCOL_ID}\n"
        f"protocol_digest: {protocol.protocol_digest()}\n"
        f"dataset_id: {dataset_id}\n"
    )


__all__ = [
    "GH_TIMEOUT_SECONDS",
    "OWNER_ID",
    "OWNER_LOGIN",
    "CommentSnapshot",
    "G4AuthError",
    "GhRunner",
    "confirm_unedited",
    "default_gh",
    "fetch_comment",
    "g4_comment_template",
    "parse_g4_url",
    "require_dataset_id",
    "require_protocol_digest",
    "require_protocol_id",
]
