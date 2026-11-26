"""External owner-comment admission: fabricated server metadata only."""

from __future__ import annotations

import copy
from hashlib import sha256

import pytest

from tests.agent_repo.test_research_assurance_github import samples
from tools.agent_repo.research_assurance import assurance_digest
from tools.external_g4_comment import (
    ExternalG4Binding,
    WorkflowRunPin,
    verify_external_g4_comment,
)

REPO = "example/research"
REPO_PATH = "repos/" + REPO
API = "https://api.github.com/" + REPO_PATH
PR = REPO_PATH + "/pulls/432"
REVIEW = PR + "/reviews/9815"
COMMENT = REPO_PATH + "/issues/comments/9876"
HEAD = "a" * 40


@pytest.fixture
def fixture():
    author, _, pr, review = samples()
    head = author["identity"]["implementation_head"]
    identity = {"id": 123, "full_name": REPO}
    owner = {"id": 10, "login": "example", "type": "User"}
    pr.update(user=owner, html_url=f"https://github.com/{REPO}/pull/432")
    for side in ("base", "head"):
        pr[side]["repo"] = identity.copy()
    pr["base"]["ref"] = "main"
    pr["head"]["ref"] = "research/test"
    review.update(
        html_url=f"https://github.com/{REPO}/pull/432#pullrequestreview-9815",
        pull_request_url=API + "/pulls/432",
    )
    review["user"]["type"] = "User"
    jobs = ("Installed-wheel synthetic regression", "Trained PPO bundle runtime")
    runs = (
        WorkflowRunPin(111, 1, 80, "pull_request", jobs),
        WorkflowRunPin(222, 1, 80, "pull_request_review", jobs),
    )
    binding = ExternalG4Binding(
        repository=REPO,
        repository_id=123,
        owner_login="example",
        owner_id=10,
        pr_number=432,
        review_id=9815,
        review_body_sha256=sha256(review["body"].encode()).hexdigest(),
        assurance_record_digest=assurance_digest(author),
        research_head=head,
        research_tree="b" * 40,
        core_head="c" * 40,
        core_tree="d" * 40,
        core_wheel_sha256="e" * 64,
        protocol_digest="f" * 64,
        external_source_digest="1" * 64,
        attempt_id="SYNTHETIC-1",
        stage="SYNTHETIC_ONLY",
        runs=runs,
    )
    url = f"https://github.com/{REPO}/pull/432#issuecomment-9876"
    comment = {
        "id": 9876,
        "html_url": url,
        "issue_url": API + "/issues/432",
        "user": owner,
        "created_at": "2026-11-02T02:00:00Z",
        "updated_at": "2026-11-02T02:00:00Z",
        "body": binding.comment_body(),
    }
    records = {
        REPO_PATH: {**identity, "owner": owner},
        PR: pr,
        REVIEW: review,
        COMMENT: comment,
        REPO_PATH + "/git/commits/" + head: {"sha": head, "tree": {"sha": "b" * 40}},
    }
    for run in runs:
        base = REPO_PATH + f"/actions/runs/{run.run_id}"
        records[base] = {
            "id": run.run_id,
            "run_attempt": run.attempt,
            "workflow_id": run.workflow_id,
            "event": run.event,
            "repository": identity,
            "head_repository": identity,
            "head_sha": head,
            "head_branch": "research/test",
            "pull_requests": [
                {
                    "number": 432,
                    "head": {"sha": head, "repo": {"id": 123}},
                    "base": {"repo": {"id": 123}},
                }
            ],
            "status": "completed",
            "conclusion": "success",
            "created_at": "2026-11-02T00:01:00Z",
            "run_started_at": "2026-11-02T00:01:00Z",
            "updated_at": "2026-11-02T01:00:00Z",
        }
        records[base + "/attempts/1/jobs?per_page=100"] = {
            "total_count": len(jobs),
            "jobs": [
                {
                    "id": run.run_id * 10 + n,
                    "run_id": run.run_id,
                    "run_attempt": 1,
                    "head_sha": head,
                    "name": name,
                    "status": "completed",
                    "conclusion": "success",
                    "started_at": "2026-11-02T00:02:00Z",
                    "completed_at": "2026-11-02T01:00:00Z",
                }
                for n, name in enumerate(jobs)
            ],
        }
    return author, binding, url, records


def run(fixture, mutate=None):
    record, binding, url, records = fixture
    if mutate:
        mutate(records)
    return verify_external_g4_comment(
        record,
        binding=binding,
        url=url,
        api_read=lambda path: copy.deepcopy(records[path]),
    )


def test_exact_synthetic_authenticated_metadata_is_bound(fixture):
    result = run(fixture)
    assert result.comment_id == 9876
    assert result.repository_id == 123
    assert result.review_id == 9815
    assert result.binding_digest == fixture[1].digest
    assert result.economic_execution_authorized is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r[REPO_PATH].update(id=999),
        lambda r: r[PR]["head"]["repo"].update(id=999),
        lambda r: r[PR]["head"].update(sha="0" * 40),
        lambda r: r[REVIEW]["user"].update(id=10),
        lambda r: r[REVIEW].update(body=r[REVIEW]["body"] + " "),
        lambda r: r[REVIEW].update(state="DISMISSED"),
        lambda r: r[COMMENT]["user"].update(id=11),
        lambda r: r[COMMENT].update(
            html_url="https://github.com/example/core/pull/432#issuecomment-9876"
        ),
        lambda r: r[COMMENT].update(issue_url=API + "/issues/433"),
        lambda r: r[COMMENT].update(
            created_at="2026-11-02T00:00:00Z", updated_at="2026-11-02T00:00:00Z"
        ),
        lambda r: r[COMMENT].update(updated_at="2026-11-02T02:01:00Z"),
        lambda r: r[COMMENT].update(body="NOT AUTHORIZED\n" + r[COMMENT]["body"]),
        lambda r: r[REPO_PATH + "/actions/runs/111"].update(conclusion="failure"),
        lambda r: r[REPO_PATH + "/actions/runs/222"].update(run_attempt=2),
        lambda r: r[REPO_PATH + "/actions/runs/222"].update(pull_requests=[]),
        lambda r: r[REPO_PATH + "/actions/runs/222"].update(
            created_at="2026-11-01T00:01:00Z"
        ),
        lambda r: r[REPO_PATH + "/actions/runs/222/attempts/1/jobs?per_page=100"][
            "jobs"
        ][0].update(conclusion="skipped"),
        lambda r: r[
            REPO_PATH + "/actions/runs/222/attempts/1/jobs?per_page=100"
        ].update(total_count=1000),
        lambda r: r.pop(COMMENT),
    ],
)
def test_external_auth_fails_closed(fixture, mutation):
    with pytest.raises(ValueError):
        run(fixture, mutation)


def test_second_authenticated_read_refuses_edited_comment(fixture):
    record, binding, url, records = fixture
    count = 0

    def read(path):
        nonlocal count
        result = copy.deepcopy(records[path])
        if path == COMMENT:
            count += 1
            if count == 2:
                result["updated_at"] = "2026-11-02T02:01:00Z"
        return result

    with pytest.raises(ValueError):
        verify_external_g4_comment(record, binding=binding, url=url, api_read=read)


def test_wrong_repository_url_refused_before_api(fixture):
    record, binding, url, _ = fixture
    with pytest.raises(ValueError):
        verify_external_g4_comment(
            record,
            binding=binding,
            url=url.replace(REPO, "example/research-other"),
            api_read=lambda path: pytest.fail("unexpected API access"),
        )


def test_empty_review_event_pr_roster_requires_exact_check_suite(fixture):
    def associate(records):
        run = records[REPO_PATH + "/actions/runs/222"]
        run["check_suite_id"] = 333
        records[REPO_PATH + "/check-suites/333"] = {
            "id": 333,
            "repository": run["repository"],
            "head_sha": run["head_sha"],
            "head_branch": run["head_branch"],
            "status": "completed",
            "conclusion": "success",
            "pull_requests": run["pull_requests"],
        }
        run["pull_requests"] = []

    assert run(fixture, associate).comment_id == 9876
    with pytest.raises(ValueError):
        run(
            fixture,
            lambda records: records[REPO_PATH + "/check-suites/333"].update(
                pull_requests=[]
            ),
        )
