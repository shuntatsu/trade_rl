import base64
from pathlib import Path

import pytest

from tools.branch_hygiene import BranchInfo


def test_delete_uses_atomic_exact_sha_leases(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.branch_hygiene import GitHubApi

    observed: dict[str, object] = {}

    class Process:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command: list[str], **kwargs: object) -> Process:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr("tools.branch_hygiene.subprocess.run", fake_run)
    api = GitHubApi(
        repository="owner/repo",
        token="secret-token",
        api_url="https://api.github.com",
    )
    branches = (
        BranchInfo("verify/one", "1" * 40, False),
        BranchInfo("tmp/two", "2" * 40, False),
    )

    api.delete_branches_with_leases(branches)

    command = observed["command"]
    assert isinstance(command, list)
    assert "--atomic" in command
    assert "--porcelain" in command
    assert (
        f"--force-with-lease=refs/heads/{branches[0].name}:{branches[0].sha}" in command
    )
    assert (
        f"--force-with-lease=refs/heads/{branches[1].name}:{branches[1].sha}" in command
    )
    assert f":refs/heads/{branches[0].name}" in command
    assert f":refs/heads/{branches[1].name}" in command
    assert all("secret-token" not in part for part in command)
    # Base64 still exposes reversible credentials if placed on process argv.
    encoded = base64.b64encode(b"x-access-token:secret-token").decode("ascii")
    assert not any("AUTHORIZATION:" in arg or encoded in arg for arg in command)
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    git_env = kwargs["env"]
    assert isinstance(git_env, dict)
    assert git_env["GIT_CONFIG_COUNT"] == "1"
    assert git_env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert git_env["GIT_CONFIG_VALUE_0"] == f"AUTHORIZATION: basic {encoded}"


def test_branch_hygiene_workflow_runs_from_default_branch() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    workflow = (
        repository_root / ".github/workflows/retire-merged-pr-branches.yml"
    ).read_text(encoding="utf-8")

    assert "push:" in workflow
    assert "- main" in workflow
    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "pull_request_target:" not in workflow
    assert "contents: write" in workflow
    assert "--retire-merged --apply --max-deletes 100" in workflow
    assert "if: github.repository == 'shuntatsu/trade_rl'" in workflow
    assert "APPLY_REQUESTED" not in workflow
    assert "pull-requests: read" in workflow
    assert "actions: read" in workflow
    assert "ref: ${{ github.event.repository.default_branch }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "python3 -m tools.branch_hygiene" in workflow


def test_unfinished_workflow_statuses_all_protect_branch_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test",
    )
    seen_statuses: list[str] = []

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        assert method == "GET"
        assert suffix == "actions/runs"
        assert query is not None
        status = query["status"]
        seen_statuses.append(status)
        return {
            "total_count": 1,
            "workflow_runs": [
                {"id": 1, "head_branch": f"task/{status}", "status": status}
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    assert api.active_workflow_branch_names() == frozenset(
        {f"task/{status}" for status in seen_statuses}
    )
    assert seen_statuses == ["requested", "waiting", "pending", "queued", "in_progress"]


@pytest.mark.parametrize(
    "head_branch_fields",
    [
        pytest.param({}, id="missing"),
        pytest.param({"head_branch": None}, id="null"),
        pytest.param({"head_branch": ""}, id="empty"),
        pytest.param({"head_branch": " \t "}, id="whitespace-only"),
        pytest.param({"head_branch": 7}, id="invalid-type"),
    ],
)
def test_unfinished_workflow_without_branch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    head_branch_fields: dict[str, object],
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test",
    )

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        assert method == "GET"
        assert suffix == "actions/runs"
        assert query is not None
        return {
            "total_count": 1,
            "workflow_runs": [
                {"id": 1, "status": query["status"], **head_branch_fields}
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    with pytest.raises(RuntimeError, match="head_branch"):
        api.active_workflow_branch_names()


@pytest.mark.parametrize(
    ("count", "should_raise"),
    [(999, False), (1000, True), (1001, True)],
)
def test_unfinished_workflow_search_limit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    should_raise: bool,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test",
    )

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        assert method == "GET"
        assert suffix == "actions/runs"
        assert query is not None
        if query["status"] != "requested":
            return {"total_count": 0, "workflow_runs": []}
        page = int(query["page"])
        start = (page - 1) * 100 + 1
        end = min(page * 100, count) + 1
        return {
            "total_count": count,
            "workflow_runs": [
                {"id": i, "status": "requested", "head_branch": f"task/{i}"}
                for i in range(start, end)
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    if should_raise:
        with pytest.raises(RuntimeError, match="1000-result limit"):
            api.active_workflow_branch_names()
    else:
        assert api.active_workflow_branch_names() == frozenset(
            f"task/{i}" for i in range(1, count + 1)
        )


def test_unfinished_workflow_missing_result_count_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test",
    )

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        assert method == "GET"
        assert suffix == "actions/runs"
        return {"workflow_runs": []}

    monkeypatch.setattr(api, "_request_json", fake_request)
    with pytest.raises(RuntimeError, match="total_count"):
        api.active_workflow_branch_names()


@pytest.mark.parametrize(
    ("variant", "error_pattern"),
    [
        ("duplicate-id", "duplicate workflow run id"),
        ("count-drift", "total_count changed between pages"),
        ("status-mismatch", "status differs from requested status"),
        ("missing-id", "workflow run id is missing or invalid"),
    ],
)
def test_unfinished_workflow_scan_rejects_inconsistent_pagination(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    error_pattern: str,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test",
    )

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        assert method == "GET"
        assert suffix == "actions/runs"
        assert query is not None
        status = query["status"]
        page = int(query["page"])
        if status != "requested":
            return {"total_count": 0, "workflow_runs": []}
        if variant == "status-mismatch":
            return {
                "total_count": 1,
                "workflow_runs": [
                    {"id": 1, "status": "completed", "head_branch": "task/1"}
                ],
            }
        if variant == "missing-id":
            return {
                "total_count": 1,
                "workflow_runs": [{"status": status, "head_branch": "task/1"}],
            }
        if page == 1:
            return {
                "total_count": 101,
                "workflow_runs": [
                    {"id": i, "status": status, "head_branch": f"task/{i}"}
                    for i in range(1, 101)
                ],
            }
        if variant == "count-drift":
            return {"total_count": 100, "workflow_runs": []}
        return {
            "total_count": 101,
            "workflow_runs": [{"id": 1, "status": status, "head_branch": "task/1"}],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    monkeypatch.setattr("tools.branch_hygiene.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match=error_pattern):
        api.active_workflow_branch_names()


def test_transient_workflow_count_mismatch_restarts_the_complete_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo", token="token", api_url="https://api.github.test"
    )
    requested_count = 0
    seen: list[str] = []

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        nonlocal requested_count
        assert method == "GET" and suffix == "actions/runs" and query is not None
        status = query["status"]
        seen.append(status)
        if status == "requested":
            requested_count += 1
            return {
                "total_count": 1,
                "workflow_runs": [
                    {
                        "id": requested_count,
                        "status": status,
                        "head_branch": (
                            "feature/stale" if requested_count == 1 else "feature/fresh"
                        ),
                    }
                ],
            }
        if status == "waiting" and requested_count == 1:
            # GitHub can return inconsistent count and items while jobs transition.
            return {
                "total_count": 2,
                "workflow_runs": [
                    {"id": 10, "status": status, "head_branch": "feature/partial"}
                ],
            }
        return {"total_count": 0, "workflow_runs": []}

    monkeypatch.setattr(api, "_request_json", fake_request)
    monkeypatch.setattr("tools.branch_hygiene.time.sleep", lambda _: None)
    assert api.active_workflow_branch_names() == frozenset({"feature/fresh"})
    assert seen == [
        "requested",
        "waiting",
        "requested",
        "waiting",
        "pending",
        "queued",
        "in_progress",
    ]


def test_bounded_backoff_recovers_after_three_transient_incomplete_rosters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo", token="token", api_url="https://api.github.test"
    )
    attempts = 0
    sleeps: list[float] = []

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        nonlocal attempts
        assert method == "GET" and suffix == "actions/runs" and query is not None
        status = query["status"]
        if status == "requested":
            attempts += 1
            # GitHub may briefly report a count larger than the returned page.
            # A partial snapshot is never accepted as a branch deletion permit.
            return {
                "total_count": 2 if attempts <= 3 else 1,
                "workflow_runs": [
                    {"id": 1, "status": status, "head_branch": "feature/stable"}
                ],
            }
        return {"total_count": 0, "workflow_runs": []}

    monkeypatch.setattr(api, "_request_json", fake_request)
    monkeypatch.setattr("tools.branch_hygiene.time.sleep", sleeps.append)
    assert api.active_workflow_branch_names() == frozenset({"feature/stable"})
    assert attempts == 4
    assert sleeps == [0.2, 0.4, 0.8]


def test_six_transient_incomplete_workflow_rosters_recover_with_fresh_complete_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Busy CI may shift status counts repeatedly; never reuse partial pages."""
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo", token="token", api_url="https://api.github.test"
    )
    attempted = 0
    statuses_seen: list[tuple[int, str]] = []
    sleeps: list[float] = []

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        nonlocal attempted
        assert method == "GET" and suffix == "actions/runs" and query is not None
        assert query["page"] == "1"
        status = query["status"]
        if status == "requested":
            attempted += 1
        statuses_seen.append((attempted, status))
        if attempted <= 6:
            # A mid-transition request is not a complete proof of active refs.
            return {
                "total_count": 2,
                "workflow_runs": [
                    {"id": attempted, "status": status, "head_branch": "task/stale"}
                ],
            }
        return {
            "total_count": 1,
            "workflow_runs": [
                {"id": 100 + attempted, "status": status, "head_branch": "task/current"}
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    monkeypatch.setattr("tools.branch_hygiene.time.sleep", sleeps.append)
    assert api.active_workflow_branch_names() == frozenset({"task/current"})
    assert attempted == 7
    assert statuses_seen == [
        *[(attempt, "requested") for attempt in range(1, 7)],
        *[
            (7, status)
            for status in ("requested", "waiting", "pending", "queued", "in_progress")
        ],
    ]
    assert sleeps == [0.2, 0.4, 0.8, 1.6, 3.2, 6.4]


def test_persistent_workflow_count_mismatch_still_refuses_all_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo", token="token", api_url="https://api.github.test"
    )
    calls = 0

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert method == "GET" and suffix == "actions/runs" and query is not None
        assert query["status"] == "requested"
        return {
            "total_count": 2,
            "workflow_runs": [
                {"id": 1, "status": "requested", "head_branch": "feature/survives"}
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    monkeypatch.setattr("tools.branch_hygiene.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="pagination is incomplete"):
        api.active_workflow_branch_names()
    assert calls == 8


def test_malformed_workflow_entry_is_not_hidden_by_snapshot_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo", token="token", api_url="https://api.github.test"
    )
    calls = 0

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "total_count": 1,
            "workflow_runs": [
                {"status": "requested", "head_branch": "feature/missing-id"}
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    with pytest.raises(RuntimeError, match="workflow run id"):
        api.active_workflow_branch_names()
    assert calls == 1


def test_unfinished_workflow_overfull_final_page_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test",
    )

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        assert method == "GET"
        assert suffix == "actions/runs"
        assert query is not None
        status = query["status"]
        if status != "requested":
            return {"total_count": 0, "workflow_runs": []}
        page = int(query["page"])
        ids = range(1, 101) if page == 1 else range(101, 103)
        return {
            "total_count": 101,
            "workflow_runs": [
                {"id": i, "status": status, "head_branch": f"task/{i}"} for i in ids
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    monkeypatch.setattr("tools.branch_hygiene.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="pagination"):
        api.active_workflow_branch_names()


def test_unfinished_workflow_exact_final_page_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test",
    )

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        assert method == "GET"
        assert suffix == "actions/runs"
        assert query is not None
        status = query["status"]
        if status != "requested":
            return {"total_count": 0, "workflow_runs": []}
        page = int(query["page"])
        assert page in (1, 2)
        ids = range(1, 101) if page == 1 else range(101, 102)
        return {
            "total_count": 101,
            "workflow_runs": [
                {"id": i, "status": status, "head_branch": f"task/{i}"} for i in ids
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    assert api.active_workflow_branch_names() == frozenset(
        f"task/{i}" for i in range(1, 102)
    )


def test_unfinished_workflow_multiple_status_pages_are_all_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test",
    )
    statuses = ("requested", "waiting", "pending", "queued", "in_progress")
    observed_requests: list[tuple[str, int]] = []

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> dict[str, object]:
        assert method == "GET"
        assert suffix == "actions/runs"
        assert query is not None
        assert query["per_page"] == "100"
        status = query["status"]
        page = int(query["page"])
        observed_requests.append((status, page))
        assert status in statuses
        assert page in (1, 2)
        ids = range(1, 101) if page == 1 else range(101, 102)
        return {
            "total_count": 101,
            "workflow_runs": [
                {
                    "id": statuses.index(status) * 1000 + i,
                    "status": status,
                    "head_branch": f"{status}/task/{i}",
                }
                for i in ids
            ],
        }

    monkeypatch.setattr(api, "_request_json", fake_request)
    assert api.active_workflow_branch_names() == frozenset(
        f"{status}/task/{i}" for status in statuses for i in range(1, 102)
    )
    assert observed_requests == [
        (status, page) for status in statuses for page in (1, 2)
    ]


def test_branch_hygiene_cli_help_marks_apply_disabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tools.branch_hygiene import parse_args

    with pytest.raises(SystemExit) as exited:
        parse_args(["--help"])
    assert exited.value.code == 0
    assert "--apply is disabled" in capsys.readouterr().out


def test_old_retention_planner_is_not_a_live_interface() -> None:
    """Old grace/anchor/delete planners must not compete with current retirement."""
    import tools.branch_hygiene as hygiene

    for obsolete in (
        "plan_cleanup",
        "retention_tip_observations",
        "anchor_shas",
        "anchor_branch_names",
        "merged_pr_retention_title",
        "parse_retention_log",
        "render_summary",
        "apply_cleanup",
    ):
        assert not hasattr(hygiene, obsolete), obsolete


def test_legacy_cli_fails_before_github_api_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting --retire-merged must never load an old deletion inventory."""
    import tools.branch_hygiene as hygiene

    def reject_io(**_kwargs: object) -> None:
        raise AssertionError("unexpected GitHub access")

    monkeypatch.setattr(hygiene, "GitHubApi", reject_io)
    for additional in ((), ("--apply",)):
        with pytest.raises(RuntimeError, match="--retire-merged"):
            hygiene.main(
                ["--repository", "shuntatsu/trade_rl", "--token", "unused", *additional]
            )


def test_repository_root_url_has_no_trailing_slash() -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="owner/repo",
        token="token",
        api_url="https://api.github.test/",
    )

    assert api._url("") == "https://api.github.test/repos/owner/repo"
    assert (
        api._url("branches", {"page": "1"})
        == "https://api.github.test/repos/owner/repo/branches?page=1"
    )
