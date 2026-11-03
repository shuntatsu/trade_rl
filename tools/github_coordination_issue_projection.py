"""Read-only GitHub Issue status projection readback, not a Coordinator authority.

The expected publisher ID and projection are supplied by the caller. Even a
stable, matching GitHub GET snapshot cannot authenticate a single Coordinator,
enumerate closed/unregistered Tasks, or authorize any repository mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from tools.agent_repo.coordination.enrollment import EnrollmentProjection
from tools.agent_repo.coordination.github_state import (
    parse_status_comment,
    render_status_comment,
)

_MAX_ISSUE_COMMENTS = 1000
_MARKER = "<!-- agent-coordination:"


class IssueReadApi(Protocol):
    def _request_json(
        self,
        method: str,
        suffix: str,
        *,
        query: dict[str, str] | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class IssueProjectionObservation:
    """A matched issue/comment snapshot, never enrollment or cleanup authority."""

    task_id: str
    projection: EnrollmentProjection
    comment_id: int
    publisher_user_id: int
    issue_readback_verified: bool = field(default=True, init=False)
    authoritative_registry_established: bool = field(default=False, init=False)
    task_registry_complete: bool = field(default=False, init=False)
    cleanup_authorized: bool = field(default=False, init=False)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_id(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive native integer")
    return value


def _issue_state(
    api: IssueReadApi, projection: EnrollmentProjection
) -> tuple[int, int, str, str, str, int, str]:
    issue = _object(
        api._request_json("GET", f"issues/{projection.issue_number}"), "Issue"
    )
    _positive_id(issue.get("id"), "Issue id")
    number = _positive_id(issue.get("number"), "Issue number")
    if (
        number != projection.issue_number
        or issue.get("node_id") != projection.issue_node_id
        or issue.get("repository_url")
        != f"https://api.github.com/repos/{projection.repository}"
        or issue.get("state") != "open"
        or "pull_request" in issue
    ):
        raise ValueError("Issue identity, repository or open state differs")
    count = _positive_id(issue.get("comments"), "Issue comments count")
    if count > _MAX_ISSUE_COMMENTS:
        raise ValueError("Issue comments exceed bounded enumeration")
    updated_at = issue.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at.strip():
        raise ValueError("Issue updated_at is missing")
    return (
        issue["id"],
        number,
        issue["node_id"],
        issue["repository_url"],
        issue["state"],
        count,
        updated_at,
    )


def _comment_rows(
    api: IssueReadApi, *, issue_number: int, total: int
) -> tuple[tuple[int, int, str, str], ...]:
    rows: list[tuple[int, int, str, str]] = []
    seen: set[int] = set()
    for page in range(1, (total + 99) // 100 + 1):
        raw = api._request_json(
            "GET",
            f"issues/{issue_number}/comments",
            query={"per_page": "100", "page": str(page)},
        )
        expected = min(100, total - 100 * (page - 1))
        if not isinstance(raw, list) or len(raw) != expected:
            raise ValueError("Issue comments pagination is incomplete")
        for item in raw:
            comment = _object(item, "Issue comment")
            comment_id = _positive_id(comment.get("id"), "comment id")
            if comment_id in seen:
                raise ValueError("Issue comments contain duplicate IDs")
            seen.add(comment_id)
            user = _object(comment.get("user"), "comment publisher")
            author_id = _positive_id(user.get("id"), "comment publisher id")
            body = comment.get("body")
            updated_at = comment.get("updated_at")
            if (
                not isinstance(body, str)
                or not isinstance(updated_at, str)
                or not updated_at.strip()
            ):
                raise ValueError("Issue comment body or updated_at is malformed")
            rows.append((comment_id, author_id, body, updated_at))
    if len(rows) != total:
        raise ValueError("Issue comments enumeration is incomplete")
    return tuple(rows)


def observe_issue_projection(
    api: IssueReadApi,
    *,
    projection: EnrollmentProjection,
    expected_publisher_user_id: int,
) -> IssueProjectionObservation:
    """Verify one stable, supplied Issue projection; all results remain untrusted.

    Two bounded comment enumerations bracketed by three Issue reads detect
    ordinary edits and pagination drift but cannot provide an atomic snapshot.
    No network writes, registration, or lease authorization occur here.
    """
    if not isinstance(projection, EnrollmentProjection):
        raise ValueError("projection must be an EnrollmentProjection")
    _positive_id(expected_publisher_user_id, "expected publisher user id")

    initial = _issue_state(api, projection)
    first = _comment_rows(api, issue_number=projection.issue_number, total=initial[5])
    middle = _issue_state(api, projection)
    if middle != initial:
        raise ValueError("Issue changed during projection readback")
    second = _comment_rows(api, issue_number=projection.issue_number, total=initial[5])
    final = _issue_state(api, projection)
    if final != initial:
        raise ValueError("Issue changed during projection readback")
    if second != first:
        raise ValueError("Issue comments changed during projection readback")

    matching: list[tuple[int, int]] = []
    for comment_id, user_id, body, _updated_at in first:
        if _MARKER not in body:
            continue
        try:
            status = parse_status_comment(body)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Issue contains a malformed Task status projection"
            ) from error
        if body != render_status_comment(status):
            raise ValueError("Issue Task status comment must be canonical")
        if status.task_id == projection.status.task_id:
            if status != projection.status:
                raise ValueError("Issue Task projection differs from enrollment status")
            matching.append((comment_id, user_id))
    if len(matching) != 1:
        raise ValueError("Issue Task projection is missing or duplicated")
    comment_id, publisher = matching[0]
    if publisher != expected_publisher_user_id:
        raise ValueError("Issue projection author differs from expected publisher")
    return IssueProjectionObservation(
        task_id=projection.status.task_id,
        projection=projection,
        comment_id=comment_id,
        publisher_user_id=publisher,
    )


__all__ = ["IssueReadApi", "IssueProjectionObservation", "observe_issue_projection"]
