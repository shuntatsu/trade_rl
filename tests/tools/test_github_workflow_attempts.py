"""Fail-closed read-only workflow-attempt observation regression cases."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from types import ModuleType

import pytest

from tools.github_workflow_attempts import main, observe_workflow_attempt

HEAD = "a" * 40
BRANCH = "feature/fenced"
RUN_ID = 1234


def _run(*, attempt: int = 2, status: str = "completed") -> dict[str, object]:
    return {
        "id": RUN_ID,
        "run_attempt": attempt,
        "head_branch": BRANCH,
        "head_sha": HEAD,
        "status": status,
        "conclusion": "success" if status == "completed" else None,
        "updated_at": "2026-11-01T12:00:00Z",
    }


def _job(number: int, *, attempt: int = 2) -> dict[str, object]:
    return {
        "id": number,
        "run_id": RUN_ID,
        "run_attempt": attempt,
        "head_branch": BRANCH,
        "head_sha": HEAD,
        "name": f"job-{number}",
        "status": "completed",
        "conclusion": "success",
    }


class FakeReader:
    def __init__(self, count: int = 2) -> None:
        self.runs = [_run(), _run()]
        self.pages: list[dict[str, object]] = [
            {"total_count": count, "jobs": [_job(i) for i in range(1, count + 1)]}
        ]
        self.requests: list[tuple[str, str, dict[str, str] | None]] = []
        self.fail = False

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        self.requests.append((method, suffix, query))
        if self.fail:
            raise RuntimeError("GitHub API unavailable")
        if suffix == f"actions/runs/{RUN_ID}":
            return deepcopy(self.runs.pop(0))
        if suffix == f"actions/runs/{RUN_ID}/attempts/2/jobs":
            assert query is not None
            return deepcopy(self.pages[int(query["page"]) - 1])
        raise AssertionError(f"unexpected API endpoint: {suffix}")


def test_attempt_scoped_jobs_are_used_even_if_unscoped_jobs_would_be_empty() -> None:
    api = FakeReader()
    report = observe_workflow_attempt(api, run_id=RUN_ID)
    assert report["job_count"] == 2
    assert [item["id"] for item in report["jobs"]] == [1, 2]
    assert report["job_source"] == "attempt_scoped_only"
    assert report["task_registry_complete"] is False
    assert report["workflow_activity_checked"] is False
    assert report["cleanup_authorized"] is False
    assert api.requests == [
        ("GET", f"actions/runs/{RUN_ID}", None),
        (
            "GET",
            f"actions/runs/{RUN_ID}/attempts/2/jobs",
            {"per_page": "100", "page": "1"},
        ),
        ("GET", f"actions/runs/{RUN_ID}", None),
    ]


def test_attempt_is_taken_from_current_run_and_never_falls_back_to_another_attempt() -> (
    None
):
    api = FakeReader()
    api.runs[1]["run_attempt"] = 3
    with pytest.raises(ValueError, match="changed during"):
        observe_workflow_attempt(api, run_id=RUN_ID)


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "b" * 40),
        ("head_branch", "elsewhere"),
        ("status", "in_progress"),
        ("updated_at", "new"),
    ],
)
def test_second_run_read_detects_observation_drift(field: str, value: object) -> None:
    api = FakeReader()
    api.runs[1][field] = value
    if field == "status":
        api.runs[1]["conclusion"] = None
    with pytest.raises(ValueError, match="changed during"):
        observe_workflow_attempt(api, run_id=RUN_ID)


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "b" * 40),
        ("run_id", 999),
        ("run_attempt", 3),
        ("head_branch", "other"),
    ],
)
def test_foreign_or_stale_job_is_rejected(field: str, value: object) -> None:
    api = FakeReader()
    api.pages[0]["jobs"][0][field] = value
    with pytest.raises(ValueError, match="(?:another run|another run attempt|differs)"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_empty_attempt_jobs_is_unknown_not_no_work() -> None:
    api = FakeReader(count=0)
    with pytest.raises(ValueError, match="nonzero"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_incomplete_page_and_duplicate_job_id_are_rejected() -> None:
    api = FakeReader(count=2)
    api.pages[0]["jobs"] = [_job(1)]
    with pytest.raises(ValueError, match="pagination"):
        observe_workflow_attempt(api, run_id=RUN_ID)

    api = FakeReader(count=2)
    api.pages[0]["jobs"] = [_job(1), _job(1)]
    with pytest.raises(ValueError, match="duplicate"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_multipage_jobs_require_complete_consistent_total_count() -> None:
    api = FakeReader(count=101)
    api.pages = [
        {"total_count": 101, "jobs": [_job(i) for i in range(1, 101)]},
        {"total_count": 101, "jobs": [_job(101)]},
    ]
    result = observe_workflow_attempt(api, run_id=RUN_ID)
    assert result["job_count"] == 101
    assert result["jobs"][-1]["id"] == 101

    api = FakeReader(count=101)
    api.pages = [
        {"total_count": 101, "jobs": [_job(i) for i in range(1, 101)]},
        {"total_count": 100, "jobs": []},
    ]
    with pytest.raises(ValueError, match="total_count changed"):
        observe_workflow_attempt(api, run_id=RUN_ID)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", True),
        ("run_attempt", 2.0),
        ("head_branch", None),
        ("head_sha", "not-a-sha"),
        ("status", "unknown"),
    ],
)
def test_noncanonical_run_identity_fails_closed(field: str, value: object) -> None:
    api = FakeReader()
    api.runs[0][field] = value
    with pytest.raises(ValueError):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_run_and_job_status_contradictions_are_rejected() -> None:
    api = FakeReader()
    api.pages[0]["jobs"][0]["status"] = "in_progress"
    api.pages[0]["jobs"][0]["conclusion"] = None
    with pytest.raises(ValueError, match="completed run contains unfinished"):
        observe_workflow_attempt(api, run_id=RUN_ID)

    api = FakeReader()
    api.pages[0]["jobs"][0]["status"] = "in_progress"
    with pytest.raises(ValueError, match="unfinished status"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_successful_run_rejects_failed_or_cancelled_completed_job() -> None:
    api = FakeReader()
    api.pages[0]["jobs"][0]["conclusion"] = "failure"
    with pytest.raises(ValueError, match="successful run contains failed"):
        observe_workflow_attempt(api, run_id=RUN_ID)


@pytest.mark.parametrize("invalid", [True, 0, 1.0])
def test_run_id_requires_native_positive_integer(invalid: object) -> None:
    api = FakeReader()
    with pytest.raises(ValueError, match="positive native integer"):
        observe_workflow_attempt(api, run_id=invalid)  # type: ignore[arg-type]


def test_network_failure_propagates_without_fabricating_snapshot() -> None:
    api = FakeReader()
    api.fail = True
    with pytest.raises(RuntimeError, match="unavailable"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_cli_audits_one_run_with_read_only_gets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reader = FakeReader()

    class FakeGitHubApi:
        def __init__(self, *, repository: str, token: str, api_url: str) -> None:
            assert repository == "shuntatsu/trade_rl"
            assert token == "fake-token"
            assert api_url == "https://api.github.com"

        def _request_json(
            self, method: str, suffix: str, *, query: dict[str, str] | None = None
        ) -> object:
            return reader._request_json(method, suffix, query=query)

    module = ModuleType("tools.branch_hygiene")
    module.GitHubApi = FakeGitHubApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.branch_hygiene", module)
    monkeypatch.setenv("GH_TOKEN", "fake-token")
    assert main(["--run-id", str(RUN_ID)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["job_count"] == 2
    assert result["cleanup_authorized"] is False
    assert all(method == "GET" for method, _, _ in reader.requests)


def test_cli_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="2"):
        main(["--run-id", str(RUN_ID)])
