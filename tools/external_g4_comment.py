"""Read-only admission evidence for a separately authorized external study.

The caller pins the repository/principal, research and core source identities,
review and CI policy *before* fetching an owner comment. This adapter does not
grant execution, allocate single-use claims, or change legacy study receipts.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from tools.research_assurance_github import _github_api_read, verify_github_review
from trade_rl.artifacts.hashing import content_digest

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_LOGIN = re.compile(r"[A-Za-z0-9-]+")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_EVENTS = ("pull_request", "pull_request_review")


def _integer(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("expected positive integer identity")
    return value


def _hex(value: str, size: int) -> None:
    if (
        type(value) is not str
        or not re.fullmatch(rf"[0-9a-f]{{{size}}}", value)
        or not value.strip("0")
    ):
        raise ValueError("invalid nonzero hexadecimal identity")


def _object(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("expected authenticated JSON object")
    return value


def _timestamp(value: object) -> datetime:
    if type(value) is not str or not _DATE.fullmatch(value):
        raise ValueError("expected canonical UTC GitHub timestamp")
    return datetime.fromisoformat(value)


@dataclass(frozen=True, slots=True)
class WorkflowRunPin:
    run_id: int
    attempt: int
    workflow_id: int
    event: str
    required_jobs: tuple[str, ...]

    def __post_init__(self) -> None:
        for item in (self.run_id, self.attempt, self.workflow_id):
            _integer(item)
        if (
            self.event not in _EVENTS
            or type(self.required_jobs) is not tuple
            or not 0 < len(self.required_jobs) <= 100
            or any(
                type(name) is not str or not name.strip() or len(name) > 256
                for name in self.required_jobs
            )
            or tuple(sorted(set(self.required_jobs))) != self.required_jobs
        ):
            raise ValueError("expected canonical required job roster and event")


@dataclass(frozen=True, slots=True)
class ExternalG4Binding:
    repository: str
    repository_id: int
    owner_login: str
    owner_id: int
    pr_number: int
    review_id: int
    review_body_sha256: str
    assurance_record_digest: str
    research_head: str
    research_tree: str
    core_head: str
    core_tree: str
    core_wheel_sha256: str
    protocol_digest: str
    external_source_digest: str
    attempt_id: str
    stage: str
    runs: tuple[WorkflowRunPin, ...]

    def __post_init__(self) -> None:
        if (
            type(self.repository) is not str
            or not _REPO.fullmatch(self.repository)
            or any(part in (".", "..") for part in self.repository.split("/"))
            or type(self.owner_login) is not str
            or not _LOGIN.fullmatch(self.owner_login)
            or self.repository.split("/")[0] != self.owner_login
        ):
            raise ValueError("invalid owner/repository identity")
        for number in (
            self.repository_id,
            self.owner_id,
            self.pr_number,
            self.review_id,
        ):
            _integer(number)
        for value in (
            self.research_head,
            self.research_tree,
            self.core_head,
            self.core_tree,
        ):
            _hex(value, 40)
        for value in (
            self.review_body_sha256,
            self.assurance_record_digest,
            self.core_wheel_sha256,
            self.protocol_digest,
            self.external_source_digest,
        ):
            _hex(value, 64)
        if any(
            type(value) is not str or not _TOKEN.fullmatch(value)
            for value in (self.attempt_id, self.stage)
        ):
            raise ValueError("invalid study attempt or stage")
        if (
            type(self.runs) is not tuple
            or any(type(run) is not WorkflowRunPin for run in self.runs)
            or tuple(run.event for run in self.runs) != _EVENTS
            or len({run.run_id for run in self.runs}) != len(self.runs)
        ):
            raise ValueError("expected two distinct ordinary/review-event run pins")

    @property
    def digest(self) -> str:
        return content_digest(self)

    def comment_body(self) -> str:
        """Exact affirmative owner message; no free text or ambiguous quoting."""
        return json.dumps(
            {"schema_version": "external_g4_owner_comment_v1", "binding": asdict(self)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class ExternalG4CommentEvidence:
    repository_id: int
    comment_id: int
    review_id: int
    binding_digest: str
    body_sha256: str
    created_at: str
    economic_execution_authorized: bool = field(default=False, init=False)


def _repo(value: object, binding: ExternalG4Binding) -> None:
    repo = _object(value)
    if (
        repo.get("full_name") != binding.repository
        or _integer(repo.get("id")) != binding.repository_id
    ):
        raise ValueError("repository identity mismatch")


def _owner(value: object, binding: ExternalG4Binding) -> None:
    owner = _object(value)
    if (
        owner.get("login") != binding.owner_login
        or _integer(owner.get("id")) != binding.owner_id
        or owner.get("type") != "User"
    ):
        raise ValueError("owner principal mismatch")


def _check_run(
    run: WorkflowRunPin,
    read: Callable[[str], dict[str, Any]],
    binding: ExternalG4Binding,
    branch: str,
    reviewed_at: datetime,
) -> datetime:
    base = f"repos/{binding.repository}/actions/runs/{run.run_id}"
    meta = read(base)
    _repo(meta.get("repository"), binding)
    _repo(meta.get("head_repository"), binding)
    if (
        _integer(meta.get("id")) != run.run_id
        or _integer(meta.get("run_attempt")) != run.attempt
        or _integer(meta.get("workflow_id")) != run.workflow_id
        or meta.get("event") != run.event
        or meta.get("head_sha") != binding.research_head
        or meta.get("head_branch") != branch
        or meta.get("status") != "completed"
        or meta.get("conclusion") != "success"
    ):
        raise ValueError("exact successful workflow run required")
    associations = meta.get("pull_requests")
    if associations == []:
        # Review-event runs can omit the PR list. A matching HEAD alone is
        # insufficient; use the authenticated check suite only if it proves it.
        suite_id = _integer(meta.get("check_suite_id"))
        suite = read(f"repos/{binding.repository}/check-suites/{suite_id}")
        _repo(suite.get("repository"), binding)
        if (
            _integer(suite.get("id")) != suite_id
            or suite.get("head_sha") != binding.research_head
            or suite.get("head_branch") != branch
            or suite.get("status") != "completed"
            or suite.get("conclusion") != "success"
        ):
            raise ValueError("workflow check suite identity mismatch")
        associations = suite.get("pull_requests")
    if type(associations) is not list or len(associations) != 1:
        raise ValueError("workflow has no unique authenticated PR association")
    association = _object(associations[0])
    head = _object(association.get("head"))
    base_pr = _object(association.get("base"))
    if (
        _integer(association.get("number")) != binding.pr_number
        or head.get("sha") != binding.research_head
        or _integer(_object(head.get("repo")).get("id")) != binding.repository_id
        or _integer(_object(base_pr.get("repo")).get("id")) != binding.repository_id
    ):
        raise ValueError("workflow PR association mismatch")
    created = _timestamp(meta.get("created_at"))
    started = _timestamp(meta.get("run_started_at"))
    finished = _timestamp(meta.get("updated_at"))
    if not created <= started <= finished or (
        run.event == "pull_request_review" and created <= reviewed_at
    ):
        raise ValueError("workflow chronology mismatch")
    jobs_meta = read(base + f"/attempts/{run.attempt}/jobs?per_page=100")
    jobs = jobs_meta.get("jobs")
    total = _integer(jobs_meta.get("total_count"))
    if type(jobs) is not list or len(jobs) != total or total > 100:
        raise ValueError("incomplete bounded job roster")
    names: set[str] = set()
    ids: set[int] = set()
    for value in jobs:
        job = _object(value)
        name = job.get("name")
        job_id = _integer(job.get("id"))
        if (
            type(name) is not str
            or name in names
            or job_id in ids
            or _integer(job.get("run_id")) != run.run_id
            or _integer(job.get("run_attempt")) != run.attempt
            or job.get("head_sha") != binding.research_head
            or job.get("status") != "completed"
        ):
            raise ValueError("workflow job identity mismatch")
        names.add(name)
        ids.add(job_id)
        if name in run.required_jobs:
            if job.get("conclusion") != "success":
                raise ValueError("required job was not successful")
            if (
                not started
                <= _timestamp(job.get("started_at"))
                <= _timestamp(job.get("completed_at"))
                <= finished
            ):
                raise ValueError("required job chronology mismatch")
    if not set(run.required_jobs).issubset(names):
        raise ValueError("required job missing")
    return finished


def verify_external_g4_comment(
    record: Mapping[str, object],
    *,
    binding: ExternalG4Binding,
    url: str,
    api_read: Callable[[str], object] = _github_api_read,
) -> ExternalG4CommentEvidence:
    """Verify a never-edited standalone owner comment after exact review and CI.

    api_read must be a trusted authenticated GitHub REST GET adapter. No
    authority is established by calling with untrusted or fabricated objects.
    Read back all admission metadata again before returning. The caller must
    still independently establish the core wheel/source pin, applicable study
    policy and one-shot ledger, and revalidate before any separately authorized
    execution. This object cannot be used as an economic authorization receipt.
    """
    if type(binding) is not ExternalG4Binding or type(url) is not str:
        raise ValueError("expected exact binding and comment URL")
    match = re.fullmatch(
        rf"https://github\.com/{re.escape(binding.repository)}/(?:issues|pull)/{binding.pr_number}#issuecomment-([1-9][0-9]*)",
        url,
    )
    if match is None:
        raise ValueError("authorization URL must bind the exact repository and PR")
    comment_id = int(match.group(1))
    prefix = f"repos/{binding.repository}"
    pr_path = f"{prefix}/pulls/{binding.pr_number}"
    review_path = f"{pr_path}/reviews/{binding.review_id}"
    comment_path = f"{prefix}/issues/comments/{comment_id}"
    snapshots: dict[str, dict[str, Any]] = {}

    def read(path: str) -> dict[str, Any]:
        value = _object(api_read(path))
        if path in snapshots and value != snapshots[path]:
            raise ValueError("authenticated admission metadata changed")
        # API adapters must return independent JSON values, not mutable aliases.
        stable = json.loads(json.dumps(value, allow_nan=False))
        snapshots[path] = stable
        return stable

    try:
        repository = read(prefix)
        _repo(repository, binding)
        _owner(repository.get("owner"), binding)
        pr = read(pr_path)
        if (
            _integer(pr.get("number")) != binding.pr_number
            or pr.get("state") != "open"
            or pr.get("html_url")
            != f"https://github.com/{binding.repository}/pull/{binding.pr_number}"
        ):
            raise ValueError("exact open PR required")
        for side in ("base", "head"):
            _repo(_object(pr.get(side)).get("repo"), binding)
        head = _object(pr.get("head"))
        if head.get("sha") != binding.research_head:
            raise ValueError("research head mismatch")
        branch = head.get("ref")
        if type(branch) is not str or not branch:
            raise ValueError("research branch missing")
        commit = read(f"{prefix}/git/commits/{binding.research_head}")
        if (
            commit.get("sha") != binding.research_head
            or _object(commit.get("tree")).get("sha") != binding.research_tree
        ):
            raise ValueError("research tree mismatch")
        review = read(review_path)
        reviewer = _object(review.get("user"))
        body = review.get("body")
        if (
            _integer(review.get("id")) != binding.review_id
            or review.get("pull_request_url") != f"https://api.github.com/{pr_path}"
            or review.get("html_url")
            != f"https://github.com/{binding.repository}/pull/{binding.pr_number}#pullrequestreview-{binding.review_id}"
            or reviewer.get("type") != "User"
            or _integer(reviewer.get("id")) == binding.owner_id
            or type(reviewer.get("login")) is not str
            or reviewer["login"].casefold() == binding.owner_login.casefold()
            or type(body) is not str
            or hashlib.sha256(body.encode("utf-8")).hexdigest()
            != binding.review_body_sha256
        ):
            raise ValueError("exact independent review identity mismatch")
        result = verify_github_review(
            record,
            repository=binding.repository,
            pr_number=binding.pr_number,
            review_id=binding.review_id,
            api_read=read,
        )
        identity = _object(record.get("identity"))
        if (
            result.status != "PASS"
            or result.record_digest != binding.assurance_record_digest
            or identity.get("implementation_head") != binding.research_head
        ):
            raise ValueError("bounded authenticated assurance review required")
        reviewed_at = _timestamp(review.get("submitted_at"))
        completed = max(
            _check_run(run, read, binding, branch, reviewed_at) for run in binding.runs
        )
        comment = read(comment_path)
        _owner(comment.get("user"), binding)
        created = _timestamp(comment.get("created_at"))
        if (
            _integer(comment.get("id")) != comment_id
            or comment.get("html_url") != url
            or comment.get("issue_url")
            != f"https://api.github.com/{prefix}/issues/{binding.pr_number}"
            or comment.get("updated_at") != comment.get("created_at")
            or created <= max(completed, reviewed_at)
            or comment.get("body") != binding.comment_body()
        ):
            raise ValueError(
                "exact unedited owner comment after review and CI required"
            )
        for path in tuple(snapshots):
            read(path)
        return ExternalG4CommentEvidence(
            binding.repository_id,
            comment_id,
            binding.review_id,
            binding.digest,
            hashlib.sha256(binding.comment_body().encode("utf-8")).hexdigest(),
            created.isoformat(),
        )
    except (KeyError, TypeError, OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(
            "authenticated admission readback unavailable or invalid"
        ) from exc


__all__ = [
    "ExternalG4Binding",
    "ExternalG4CommentEvidence",
    "WorkflowRunPin",
    "verify_external_g4_comment",
]
