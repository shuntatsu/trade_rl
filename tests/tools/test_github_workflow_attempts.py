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
    with pytest.raises(ValueError, match="unfinished job"):
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


@pytest.mark.parametrize("location", ["initial-run", "readback-run", "job"])
def test_unfinished_status_requires_explicit_null_conclusion(location: str) -> None:
    api = FakeReader()
    api.runs = [_run(status="in_progress"), _run(status="in_progress")]
    job = api.pages[0]["jobs"][0]
    job["status"] = "in_progress"
    job["conclusion"] = None
    if location == "initial-run":
        del api.runs[0]["conclusion"]
    elif location == "readback-run":
        del api.runs[1]["conclusion"]
    else:
        del job["conclusion"]
    with pytest.raises(ValueError, match="conclusion is missing"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_unfinished_status_accepts_explicit_null_conclusion() -> None:
    api = FakeReader()
    api.runs = [_run(status="in_progress"), _run(status="in_progress")]
    job = api.pages[0]["jobs"][0]
    job["status"] = "in_progress"
    job["conclusion"] = None
    result = observe_workflow_attempt(api, run_id=RUN_ID)
    assert result["status"] == "in_progress"
    assert result["conclusion"] is None
    assert result["jobs"][0]["conclusion"] is None
    assert result["cleanup_authorized"] is False


@pytest.mark.parametrize("count", [100, 200])
def test_full_final_attempt_page_requires_empty_terminal_page(count: int) -> None:
    api = FakeReader(count=count)
    api.pages = [
        {"total_count": count, "jobs": [_job(i) for i in range(start, start + 100)]}
        for start in range(1, count + 1, 100)
    ]
    api.pages.append({"total_count": count, "jobs": []})
    result = observe_workflow_attempt(api, run_id=RUN_ID)
    assert result["job_count"] == count
    assert api.requests[-2][2] == {"per_page": "100", "page": str(count // 100 + 1)}


def test_full_final_attempt_page_rejects_hidden_extra_job() -> None:
    api = FakeReader(count=100)
    api.pages.append({"total_count": 100, "jobs": [_job(101)]})
    with pytest.raises(ValueError, match="pagination"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_full_final_attempt_page_rejects_changed_total() -> None:
    api = FakeReader(count=100)
    api.pages.append({"total_count": 101, "jobs": [_job(101)]})
    with pytest.raises(ValueError, match="total_count changed"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_matching_null_run_and_job_heads_are_not_exact_commits() -> None:
    api = FakeReader()
    for run in api.runs:
        run["head_sha"] = "0" * 40
    for job in api.pages[0]["jobs"]:
        job["head_sha"] = "0" * 40
    with pytest.raises(ValueError, match="SHA"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_null_job_head_is_rejected_even_with_valid_run_head() -> None:
    api = FakeReader()
    api.pages[0]["jobs"][0]["head_sha"] = "0" * 40
    with pytest.raises(ValueError, match="job head SHA must not be a null SHA"):
        observe_workflow_attempt(api, run_id=RUN_ID)


class _ForgedIdentity(str):
    def __eq__(self, other: object) -> bool:
        return other != "0" * 40

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = str.__hash__


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", _ForgedIdentity("b" * 40)),
        ("head_branch", _ForgedIdentity("foreign")),
    ],
)
def test_attempt_job_rejects_foreign_identity(field: str, value: str) -> None:
    api = FakeReader()
    api.pages[0]["jobs"][0][field] = value
    with pytest.raises(ValueError, match="job head"):
        observe_workflow_attempt(api, run_id=RUN_ID)


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", _ForgedIdentity("b" * 40)),
        ("head_branch", _ForgedIdentity("foreign")),
        ("updated_at", _ForgedIdentity("different")),
    ],
)
def test_attempt_readback_rejects_foreign_identity(field: str, value: str) -> None:
    api = FakeReader()
    api.runs[1][field] = value
    with pytest.raises(ValueError, match="workflow (?:head|update)"):
        observe_workflow_attempt(api, run_id=RUN_ID)


class _IdentityTrap(str):
    def strip(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("foreign strip invoked")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("foreign equality invoked")

    def __ne__(self, other: object) -> bool:
        raise AssertionError("foreign inequality invoked")

    __hash__ = str.__hash__


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", _IdentityTrap(HEAD)),
        ("head_branch", _IdentityTrap(BRANCH)),
        ("updated_at", _IdentityTrap("2026-11-01T12:00:00Z")),
    ],
)
def test_attempt_initial_run_rejects_foreign_hooks(field: str, value: str) -> None:
    api = FakeReader()
    api.runs[0][field] = value
    with pytest.raises(ValueError, match="workflow (?:head|update)"):
        observe_workflow_attempt(api, run_id=RUN_ID)


def test_attempt_job_name_rejects_foreign_hook() -> None:
    api = FakeReader()
    api.pages[0]["jobs"][0]["name"] = _IdentityTrap("job-1")
    with pytest.raises(ValueError, match="job name"):
        observe_workflow_attempt(api, run_id=RUN_ID)


class _ResponseObjectTrap(dict):
    def get(self, *args, **kwargs):
        raise AssertionError("foreign object lookup")

    def __getitem__(self, key):
        raise AssertionError("foreign object item lookup")

    def __contains__(self, key):
        raise AssertionError("foreign object membership")

    def __iter__(self):
        raise AssertionError("foreign object iteration")


class _ResponseRowsTrap(list):
    def __len__(self):
        raise AssertionError("foreign rows length")

    def __iter__(self):
        raise AssertionError("foreign rows iteration")


class _ResponseKey(str):
    pass


class _ForeignAttemptReader(FakeReader):
    def __init__(self, location):
        super().__init__()
        self.location = location

    def _request_json(self, method, suffix, *, query=None):
        value = super()._request_json(method, suffix, query=query)
        location = self.location
        if suffix == f"actions/runs/{RUN_ID}":
            stage = "initial" if self.runs else "readback"
            if location == stage:
                return _ResponseObjectTrap(value)
            if location == stage + "_key":
                return {
                    _ResponseKey(k) if k == "head_sha" else k: v
                    for k, v in value.items()
                }
        else:
            if location == "page":
                return _ResponseObjectTrap(value)
            if location == "page_key":
                return {
                    _ResponseKey(k) if k == "jobs" else k: v for k, v in value.items()
                }
            if location == "rows":
                value["jobs"] = _ResponseRowsTrap(value["jobs"])
            elif location == "job":
                value["jobs"][0] = _ResponseObjectTrap(value["jobs"][0])
            elif location == "job_key":
                value["jobs"][0] = {
                    _ResponseKey(k) if k == "id" else k: v
                    for k, v in value["jobs"][0].items()
                }
        return value


@pytest.mark.parametrize(
    "location",
    [
        "initial",
        "initial_key",
        "readback",
        "readback_key",
        "page",
        "page_key",
        "rows",
        "job",
        "job_key",
    ],
)
def test_attempt_observation_refuses_foreign_json_containers(location):
    with pytest.raises(ValueError, match="native JSON|pagination"):
        observe_workflow_attempt(_ForeignAttemptReader(location), run_id=RUN_ID)


@pytest.mark.parametrize(
    "branch",
    ["bad\0ref", "bad\nref", "bad\ud800ref", "bad//ref", "bad..ref", "bad.lock"],
)
@pytest.mark.parametrize("readback", [False, True])
def test_attempt_rejects_malformed_head_ref(branch: str, readback: bool) -> None:
    reader = FakeReader(count=1)
    if readback:
        reader.runs[1]["head_branch"] = branch
    else:
        for run in reader.runs:
            run["head_branch"] = branch
        reader.pages = [
            {"total_count": 1, "jobs": [{**_job(1), "head_branch": branch}]}
        ]
    with pytest.raises(ValueError, match="workflow head branch is malformed"):
        observe_workflow_attempt(reader, run_id=RUN_ID)
    if not readback:
        assert reader.requests == [("GET", f"actions/runs/{RUN_ID}", None)]


@pytest.mark.parametrize("branch", ["main", "feature/日本語", "v1.2.3", "a" * 40])
def test_attempt_accepts_valid_head_ref_without_authority(branch: str) -> None:
    reader = FakeReader(count=1)
    for run in reader.runs:
        run["head_branch"] = branch
    reader.pages = [{"total_count": 1, "jobs": [{**_job(1), "head_branch": branch}]}]
    report = observe_workflow_attempt(reader, run_id=RUN_ID)
    assert report["head_branch"] == branch
    assert report["workflow_activity_checked"] is False
    assert report["task_registry_complete"] is False
    assert report["cleanup_authorized"] is False
