"""Falsifiable synthetic coverage for fixed six-ref 2026-11-21 retirement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import archived_obsolete_20261121 as mod
from tools.branch_hygiene import BranchInfo, OpenPullRequestRefs


def _ref(name: str, sha: str, protected: bool = False) -> BranchInfo:
    return BranchInfo(name, sha, protected)


def _setup(monkeypatch: pytest.MonkeyPatch) -> Any:
    targets = dict(mod.HISTORICAL_MAIN_ANCESTORS)
    targets.update({name: pair[0] for name, pair in mod.CLOSED_PR_SOURCES.items()})
    rows = [
        _ref("main", mod._SNAPSHOT_MAIN),
        _ref(mod.ARCHIVE_BRANCH, mod.ARCHIVE_HEAD),
        *(_ref(name, sha) for name, sha in sorted(targets.items())),
    ]
    frozen = {row.name: row.sha for row in rows}

    monkeypatch.setattr(mod, "_archive_manifest", lambda: frozen)
    monkeypatch.setattr(mod, "_git_main", lambda: mod._SNAPSHOT_MAIN)
    monkeypatch.setattr(mod, "_is_ancestor", lambda older, newer: True)
    monkeypatch.setattr(mod, "require_retirement_origin", lambda: None)

    class FakeApi:
        _repository = mod.REPOSITORY

        def __init__(self) -> None:
            self.refs = list(rows)
            self.opens: tuple[OpenPullRequestRefs, ...] = ()
            self.active: frozenset[str] = frozenset()
            self.pr_wrong_heads: set[int] = set()
            self.pr_wrong_repos: set[int] = set()
            self.pr_merged: set[int] = set()
            self.pr_open: set[int] = set()
            self.deleted: list[tuple[BranchInfo, ...]] = []
            self.post_extra = False

        def repository_default_branch(self) -> str:
            return "main"

        def branches(self) -> tuple[BranchInfo, ...]:
            return tuple(self.refs)

        def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
            return self.opens

        def active_workflow_branch_names(self) -> frozenset[str]:
            return self.active

        def _request_json(self, method: str, path: str) -> dict[str, Any]:
            assert method == "GET" and path.startswith("pulls/")
            num = int(path.rsplit("/", 1)[-1])
            name = next(
                n for n, pair in mod.CLOSED_PR_SOURCES.items() if pair[1] == num
            )
            branch = next(x for x in rows if x.name == name)
            return {
                "number": num,
                "state": "open" if num in self.pr_open else "closed",
                "closed_at": "2026-11-21T05:00:00Z",
                "merged_at": "2026-11-21T05:00:00Z" if num in self.pr_merged else None,
                "head": {
                    "ref": name,
                    "sha": "a" * 40 if num in self.pr_wrong_heads else branch.sha,
                    "repo": {
                        "full_name": (
                            "attacker/repo"
                            if num in self.pr_wrong_repos
                            else mod.REPOSITORY
                        )
                    },
                },
                "base": {"ref": "main", "repo": {"full_name": mod.REPOSITORY}},
            }

        def delete_branches_with_leases(self, entries: tuple[BranchInfo, ...]) -> None:
            self.deleted.append(entries)
            self.refs = [x for x in self.refs if x not in entries]
            if self.post_extra:
                self.refs.append(_ref("new/foreign", "f" * 40))

    return FakeApi()


def _owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    event = tmp_path / "owner-event.json"
    event.write_text(
        json.dumps(
            {
                "action": "created",
                "issue": {"number": mod.ISSUE_NUMBER},
                "comment": {
                    "body": mod.COMMAND,
                    "user": {"login": "shuntatsu", "id": mod.OWNER_ID},
                },
            }
        ),
        encoding="utf-8",
    )
    for key, val in (
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", mod.REPOSITORY),
        ("GITHUB_EVENT_NAME", "issue_comment"),
        ("GITHUB_ACTOR", "shuntatsu"),
        ("GITHUB_EVENT_PATH", str(event)),
        ("GITHUB_SHA", mod._SNAPSHOT_MAIN),
    ):
        monkeypatch.setenv(key, val)
    return event


def test_only_six_exact_nonreserved_targets() -> None:
    assert len(mod.CLOSED_PR_SOURCES) == 4
    assert len(mod.HISTORICAL_MAIN_ANCESTORS) == 2
    assert mod.EXPECTED_COUNT == 6
    names = set(mod.CLOSED_PR_SOURCES) | set(mod.HISTORICAL_MAIN_ANCESTORS)
    assert len(names) == 6
    assert all(
        not name.startswith(
            (
                "fix/m1-",
                "research/",
                "audit/",
                "infra/",
                "provenance/",
                "archive/",
            )
        )
        and name != "main"
        for name in names
    )
    assert mod.COMMAND.endswith(mod.ARCHIVE_HEAD)
    assert "fix/m2-fd-exception-close-20261120" not in names
    assert "fix/issue1131-json-staging-foreign-cleanup-20261119" not in names


def test_dry_run_selects_six_and_never_mutates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _setup(monkeypatch)
    total, selected, deleted = mod.retire(api, apply=False)
    assert total == 8
    assert len(selected) == 6
    assert deleted == ()
    assert api.deleted == []


@pytest.mark.parametrize("fault", ["open", "merged", "head", "repo"])
def test_open_merged_wrong_head_or_wrong_repo_pr_is_not_eligible(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    api = _setup(monkeypatch)
    number = mod.CLOSED_PR_SOURCES["review/issue810-trend-oos-20261120-protocol"][1]
    if fault == "open":
        api.pr_open.add(number)
    elif fault == "merged":
        api.pr_merged.add(number)
    elif fault == "head":
        api.pr_wrong_heads.add(number)
    else:
        api.pr_wrong_repos.add(number)
    _, selected, _ = mod.retire(api, apply=False)
    assert "review/issue810-trend-oos-20261120-protocol" not in selected
    assert len(selected) == 5


def test_live_open_pr_active_workflow_and_protected_refs_are_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _setup(monkeypatch)
    head = "review/issue810-trend-oos-20261120-runner"
    workflow = "review/issue810-trend-oos-20261120-protocol"
    protected = "review/issue810-trend-oos-20261120-base"
    api.opens = (
        OpenPullRequestRefs(head, mod.CLOSED_PR_SOURCES[head][0], "main", "a" * 40),
    )
    api.active = frozenset({workflow})
    api.refs = [
        _ref(row.name, row.sha, protected=True) if row.name == protected else row
        for row in api.refs
    ]
    _, selected, _ = mod.retire(api, apply=False)
    assert len(selected) == 3
    assert head not in selected
    assert workflow not in selected
    assert protected not in selected


def test_sha_drift_fails_closed_before_any_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _setup(monkeypatch)
    name = "fix/core7-daily-fallback-digest-20261120"
    api.refs = [
        _ref(row.name, "f" * 40) if row.name == name else row for row in api.refs
    ]
    with pytest.raises(RuntimeError, match="SHA"):
        mod.retire(api, apply=False)
    assert api.deleted == []


def test_main_ancestry_or_archive_reachability_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _setup(monkeypatch)
    monkeypatch.setattr(mod, "_is_ancestor", lambda a, b: False)
    with pytest.raises(RuntimeError, match="archive lacks"):
        mod.retire(api, apply=False)


def test_owner_only_exact_created_comment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    event = _owner(monkeypatch, tmp_path)
    mod._owner_created_event()
    body = json.loads(event.read_text(encoding="utf-8"))
    body["action"] = "edited"
    event.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact owner-created"):
        mod._owner_created_event()
    body["action"] = "created"
    body["comment"]["body"] = "/retire-another-command"
    event.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact owner-created"):
        mod._owner_created_event()
    body["comment"]["body"] = mod.COMMAND
    body["comment"]["user"]["id"] = 1
    event.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact owner-created"):
        mod._owner_created_event()


def test_successful_exact_sha_atomic_batch_then_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _setup(monkeypatch)
    _owner(monkeypatch, tmp_path)
    total, planned, deleted = mod.retire(api, apply=True)
    assert total == 8
    assert len(planned) == 6
    assert deleted == planned
    assert len(api.deleted) == 1
    assert tuple(x.name for x in api.deleted[0]) == planned
    assert len(api.refs) == 2
    assert {x.name for x in api.refs} == {"main", mod.ARCHIVE_BRANCH}
    assert mod.retire(api, apply=True) == (2, (), ())


def test_missing_or_wrong_github_sha_rejects_before_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _setup(monkeypatch)
    _owner(monkeypatch, tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    with pytest.raises(RuntimeError, match="fresh current main"):
        mod.retire(api, apply=True)
    assert api.deleted == []


def test_inconsistent_post_delete_roster_is_never_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _setup(monkeypatch)
    _owner(monkeypatch, tmp_path)
    api.post_extra = True
    with pytest.raises(RuntimeError, match="post-delete exact"):
        mod.retire(api, apply=True)


def test_workflow_command_job_and_token_counts() -> None:
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/retire-merged-pr-branches.yml"
    ).read_text(encoding="utf-8")
    assert "archived-obsolete-20261121:" in workflow
    assert mod.COMMAND in workflow
    assert "github.event.action == 'created'" in workflow
    assert "python3 -m tools.archived_obsolete_20261121 --apply" in workflow
    assert workflow.count("GH_TOKEN: " + "$" + "{{ github.token }}") == 16


def test_archive_snapshot_source_is_pinned_and_never_overwritten() -> None:
    assert mod.ARCHIVE_BRANCH == "archive/branch-snapshot-20261121"
    assert mod.ARCHIVE_HEAD == "b95286e38df9e46a10f9fb43fbfac14dd3c51499"
    assert "archive/" not in mod.CLOSED_PR_SOURCES


def _snapshot_manifest() -> dict[str, Any]:
    """A valid synthetic 56-tip archive, including the fixed six source tips."""
    entries = [
        {"name": "main", "sha": mod._SNAPSHOT_MAIN, "protected": False},
        {
            "name": "archive/branch-snapshot-20261120",
            "sha": "1" * 40,
            "protected": False,
        },
    ]
    entries.extend(
        {"name": name, "sha": sha, "protected": False}
        for name, sha in sorted(
            {
                **mod.HISTORICAL_MAIN_ANCESTORS,
                **{name: pair[0] for name, pair in mod.CLOSED_PR_SOURCES.items()},
            }.items()
        )
    )
    entries.extend(
        {
            "name": f"synthetic/extra-{index:02d}",
            "sha": f"{index + 100:040x}",
            "protected": False,
        }
        for index in range(56 - len(entries))
    )
    assert len(entries) == 56
    return {
        "schema": "trade_rl_branch_tip_snapshot_v2",
        "repository": mod.REPOSITORY,
        "archive_branch": mod.ARCHIVE_BRANCH,
        "main_head": mod._SNAPSHOT_MAIN,
        "branches": entries,
    }


def _fake_git_archive(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    *,
    ref_sha: str | None = None,
) -> None:
    def run(*argv: str) -> str:
        if argv == ("rev-parse", f"refs/remotes/origin/{mod.ARCHIVE_BRANCH}"):
            return mod.ARCHIVE_HEAD if ref_sha is None else ref_sha
        if argv == ("show", f"{mod.ARCHIVE_HEAD}:branch-tip-manifest.json"):
            return json.dumps(payload)
        raise AssertionError(f"unexpected Git access: {argv!r}")

    monkeypatch.setattr(mod, "_git", run)


def test_archive_manifest_parses_all_56_tips_and_pins_each_exact_name_and_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _snapshot_manifest()
    _fake_git_archive(monkeypatch, payload)
    parsed = mod._archive_manifest()
    assert len(parsed) == 56
    assert parsed["main"] == mod._SNAPSHOT_MAIN
    for name, pair in mod.CLOSED_PR_SOURCES.items():
        assert parsed[name] == pair[0]
    for name, sha in mod.HISTORICAL_MAIN_ANCESTORS.items():
        assert parsed[name] == sha


@pytest.mark.parametrize(
    "fault",
    [
        "archive-ref-drift",
        "wrong-schema",
        "wrong-repository",
        "wrong-archive-name",
        "wrong-snapshot-main",
        "missing-tip",
        "duplicate-name",
        "invalid-sha",
        "non-boolean-protected",
        "missing-main-tip",
        "non-object-tip",
    ],
)
def test_archive_manifest_rejects_corrupted_metadata_and_ref_drift(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    payload = _snapshot_manifest()
    ref_sha: str | None = None
    if fault == "archive-ref-drift":
        ref_sha = "f" * 40
    elif fault == "wrong-schema":
        payload["schema"] = "unknown"
    elif fault == "wrong-repository":
        payload["repository"] = "other/repo"
    elif fault == "wrong-archive-name":
        payload["archive_branch"] = "archive/other"
    elif fault == "wrong-snapshot-main":
        payload["main_head"] = "f" * 40
    elif fault == "missing-tip":
        payload["branches"].pop()
    elif fault == "duplicate-name":
        payload["branches"][-1]["name"] = payload["branches"][1]["name"]
    elif fault == "invalid-sha":
        payload["branches"][2]["sha"] = "invalid"
    elif fault == "non-boolean-protected":
        payload["branches"][3]["protected"] = 1
    elif fault == "missing-main-tip":
        payload["branches"][0]["name"] = "synthetic/missing-original-main"
    elif fault == "non-object-tip":
        payload["branches"][4] = "malformed"
    else:
        raise AssertionError(fault)
    _fake_git_archive(monkeypatch, payload, ref_sha=ref_sha)
    with pytest.raises(RuntimeError, match="archive"):
        mod._archive_manifest()


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("fix/core7-daily-fallback-digest-20261120", "Core7 fallback code"),
        ("review/issue810-trend-oos-20261120-base", "old review base"),
        ("review/issue810-trend-oos-20261120-test-part1", "old review base"),
    ],
)
def test_each_target_main_ancestry_loss_fails_closed(
    monkeypatch: pytest.MonkeyPatch, target: str, error: str
) -> None:
    api = _setup(monkeypatch)
    tip = (
        mod.CLOSED_PR_SOURCES[target][0]
        if target in mod.CLOSED_PR_SOURCES
        else mod.HISTORICAL_MAIN_ANCESTORS[target]
    )
    # Both review bases share an exact SHA; fail-closed on the first matching one.
    monkeypatch.setattr(
        mod,
        "_is_ancestor",
        lambda older, newer: not (older == tip and newer == mod._SNAPSHOT_MAIN),
    )
    with pytest.raises(RuntimeError, match=error):
        mod.retire(api, apply=False)
    assert api.deleted == []


def test_live_main_tip_mismatch_blocks_all_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _setup(monkeypatch)
    _owner(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "_git_main", lambda: "f" * 40)
    with pytest.raises(RuntimeError, match="main/archive identity"):
        mod.retire(api, apply=True)
    assert api.deleted == []


def test_pre_apply_branch_roster_drift_aborts_without_deleting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _setup(monkeypatch)
    _owner(monkeypatch, tmp_path)
    original = api.branches
    calls = 0

    def concurrent_branch_creation() -> tuple[BranchInfo, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            api.refs.append(_ref("foreign/concurrent-branch", "f" * 40))
        return original()

    api.branches = concurrent_branch_creation
    with pytest.raises(RuntimeError, match="branch roster changed before apply"):
        mod.retire(api, apply=True)
    assert calls == 2
    assert api.deleted == []


def test_post_first_check_open_pr_drift_aborts_without_deleting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _setup(monkeypatch)
    _owner(monkeypatch, tmp_path)
    calls = 0
    target = "review/issue810-trend-oos-20261120-runner"

    def changing_open_pr_refs() -> tuple[OpenPullRequestRefs, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ()
        return (
            OpenPullRequestRefs(
                target,
                mod.CLOSED_PR_SOURCES[target][0],
                "main",
                mod._SNAPSHOT_MAIN,
            ),
        )

    api.open_pull_request_refs = changing_open_pr_refs
    with pytest.raises(RuntimeError, match="eligibility drift"):
        mod.retire(api, apply=True)
    assert calls == 2
    assert api.deleted == []


def test_main_cli_preview_then_exact_created_owner_apply_routes_correctly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = _setup(monkeypatch)
    _owner(monkeypatch, tmp_path)
    monkeypatch.setenv("GH_TOKEN", "synthetic-ephemeral-unit-token")
    constructions: list[dict[str, str]] = []

    def make_api(*, repository: str, token: str, api_url: str) -> Any:
        constructions.append(
            {"repository": repository, "token": token, "api_url": api_url}
        )
        return api

    monkeypatch.setattr(mod, "GitHubApi", make_api)
    assert mod.main([]) == 0
    preview = capsys.readouterr().out
    assert "- mode: preview" in preview
    assert "- exact eligible candidates: 6" in preview
    assert "- deleted and verified: 0" in preview
    assert api.deleted == []

    assert mod.main(["--apply"]) == 0
    applied = capsys.readouterr().out
    assert "- mode: apply" in applied
    assert "- deleted and verified: 6" in applied
    assert len(api.deleted) == 1
    assert len(api.deleted[0]) == 6
    assert mod.main(["--apply"]) == 0
    assert "- deleted and verified: 0" in capsys.readouterr().out
    assert len(api.deleted) == 1
    assert (
        constructions
        == [
            {
                "repository": mod.REPOSITORY,
                "token": "synthetic-ephemeral-unit-token",
                "api_url": "https://api.github.com",
            }
        ]
        * 3
    )


def test_main_cli_requires_real_token_even_for_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GitHub Actions token"):
        mod.main([])
