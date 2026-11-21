"""Regression for two owner-specified archived closed PR ref names (software only)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import archived_closed_pr_retirement as mod
from tools.branch_hygiene import BranchInfo, OpenPullRequestRefs

MAIN_SHA = "1" * 40


def _ref(name: str, sha: str, protected: bool = False) -> BranchInfo:
    return BranchInfo(name, sha, protected)


def _setup(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, tuple[BranchInfo, ...]]:
    main = _ref("main", MAIN_SHA)
    archive = _ref(mod.ARCHIVE_BRANCH, mod.ARCHIVE_HEAD)
    names = sorted(mod.FOLLOWUP_CLOSED_PR_ALLOWLIST)
    old = tuple(_ref(n, f"{i + 2:040x}") for i, n in enumerate(names))
    all_refs = (main, archive, *old)
    monkeypatch.setattr(
        mod, "_archive_manifest", lambda: {x.name: x.sha for x in all_refs}
    )
    monkeypatch.setattr(mod, "_git_main", lambda: MAIN_SHA)
    monkeypatch.setattr(mod, "_git_ancestor", lambda tip, dest: True)

    class FakeApi:
        _repository = mod.REPOSITORY

        def __init__(self) -> None:
            self.refs = list(all_refs)
            self.open_refs: tuple[OpenPullRequestRefs, ...] = ()
            self.active = frozenset()
            self.pr_states: dict[int, str] = {}
            self.pr_wrong_heads: set[int] = set()
            self.pr_wrong_repos: set[int] = set()
            self.deleted: list[tuple[BranchInfo, ...]] = []

        def repository_default_branch(self) -> str:
            return "main"

        def branches(self) -> tuple[BranchInfo, ...]:
            return tuple(self.refs)

        def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
            return self.open_refs

        def active_workflow_branch_names(self) -> frozenset[str]:
            return self.active

        def _request_json(self, method: str, path: str) -> dict[str, Any]:
            assert method == "GET" and path.startswith("pulls/")
            n = int(path.split("/")[-1])
            ref_name = next(
                key
                for key, value in mod.FOLLOWUP_CLOSED_PR_ALLOWLIST.items()
                if value == n
            )
            branch = next(x for x in all_refs if x.name == ref_name)
            repo = "bad/repo" if n in self.pr_wrong_repos else mod.REPOSITORY
            return {
                "number": n,
                "state": self.pr_states.get(n, "closed"),
                "closed_at": "2026-11-20T04:00:00Z",
                "head": {
                    "ref": ref_name,
                    "sha": "f" * 40 if n in self.pr_wrong_heads else branch.sha,
                    "repo": {"full_name": repo},
                },
                "base": {"ref": "main", "repo": {"full_name": mod.REPOSITORY}},
            }

        def delete_branches_with_leases(self, names: tuple[BranchInfo, ...]) -> None:
            self.deleted.append(names)
            self.refs = [x for x in self.refs if x not in names]

    return FakeApi(), all_refs


def _owner_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    event = tmp_path / "owner-event.json"
    event.write_text(
        json.dumps(
            {
                "action": "created",
                "issue": {"number": 1057},
                "comment": {
                    "body": mod.FOLLOWUP_COMMAND,
                    "user": {"login": "shuntatsu", "id": 148511409},
                },
            }
        ),
        encoding="utf-8",
    )
    for key, value in (
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", mod.REPOSITORY),
        ("GITHUB_ACTOR", "shuntatsu"),
        ("GITHUB_EVENT_NAME", "issue_comment"),
        ("GITHUB_EVENT_PATH", str(event)),
        ("GITHUB_SHA", MAIN_SHA),
    ):
        monkeypatch.setenv(key, value)
    repository = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "remote",
            "add",
            "origin",
            "https://github.com/shuntatsu/trade_rl.git",
        ],
        check=True,
    )
    monkeypatch.chdir(repository)
    return event


def test_allowlist_exact_two_unchanged_original_28() -> None:
    assert mod.FOLLOWUP_COUNT == 2
    assert mod.MAX_DELETES == 28
    assert len(mod.CLOSED_PR_ALLOWLIST) == 28
    assert mod.FOLLOWUP_CLOSED_PR_ALLOWLIST == {
        "fix/issue810-crossasset-symlink-test-portability-20261120": 1286,
        "fix/m2-study-json-exclusive-publication-20261110": 1205,
    }
    assert set(mod.FOLLOWUP_CLOSED_PR_ALLOWLIST).isdisjoint(mod.CLOSED_PR_ALLOWLIST)
    assert mod.COMMAND != mod.FOLLOWUP_COMMAND


def test_only_followup_mode_selects_new_branch_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, refs = _setup(monkeypatch)
    assert mod.retire(api, apply=False) == (4, (), ())
    assert mod.retire(api, apply=False, followup=True) == (
        4,
        tuple(x.name for x in refs[2:]),
        (),
    )
    assert not api.deleted


def test_closed_pr_exact_sha_and_repo_and_open_or_active_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, refs = _setup(monkeypatch)
    n0 = mod.FOLLOWUP_CLOSED_PR_ALLOWLIST[refs[2].name]
    n1 = mod.FOLLOWUP_CLOSED_PR_ALLOWLIST[refs[3].name]
    api.pr_states[n0] = "open"
    api.pr_wrong_heads.add(n1)
    assert mod.retire(api, apply=False, followup=True)[1] == ()
    api.pr_states.clear()
    api.pr_wrong_heads.clear()
    api.pr_wrong_repos.add(n0)
    assert mod.retire(api, apply=False, followup=True)[1] == (refs[3].name,)
    api.pr_wrong_repos.clear()
    api.open_refs = (OpenPullRequestRefs(refs[2].name, refs[2].sha, "main", MAIN_SHA),)
    api.active = frozenset({refs[3].name})
    assert mod.retire(api, apply=False, followup=True)[1] == ()
    api.open_refs = ()
    api.active = frozenset()
    api.refs[2] = _ref(refs[2].name, "e" * 40)
    assert mod.retire(api, apply=False, followup=True)[1] == (refs[3].name,)


def test_original_command_and_non_created_action_do_not_authorize_followup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event = _owner_created(monkeypatch, tmp_path)
    mod._owner_comment_event(followup=True)
    data = json.loads(event.read_text())
    data["comment"]["body"] = mod.COMMAND
    event.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="exact owner"):
        mod._owner_comment_event(followup=True)
    data["comment"]["body"] = mod.FOLLOWUP_COMMAND
    data["action"] = "edited"
    event.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="exact owner"):
        mod._owner_comment_event(followup=True)


def test_valid_followup_applies_only_two_leased_refs_and_keeps_main_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, refs = _setup(monkeypatch)
    _owner_created(monkeypatch, tmp_path)
    count, selected, deleted = mod.retire(api, apply=True, followup=True)
    assert count == 4
    assert selected == deleted == tuple(x.name for x in refs[2:])
    assert api.deleted == [refs[2:]]
    assert api.refs == list(refs[:2])


def test_post_delete_readback_failure_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, _ = _setup(monkeypatch)
    _owner_created(monkeypatch, tmp_path)
    original = api.delete_branches_with_leases

    def divergent(items: tuple[BranchInfo, ...]) -> None:
        original(items)
        api.refs.append(_ref("new/foreign", "f" * 40))

    api.delete_branches_with_leases = divergent
    with pytest.raises(RuntimeError, match="post-delete exact"):
        mod.retire(api, apply=True, followup=True)


def test_workflow_exact_owner_followup_command_and_tokens() -> None:
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/retire-merged-pr-branches.yml"
    ).read_text(encoding="utf-8")
    assert "frozen-closed-pr-followup:" in workflow
    assert mod.FOLLOWUP_COMMAND in workflow
    assert "github.event.action == 'created'" in workflow
    assert workflow.count("GH_TOKEN: " + "$" + "{{ github.token }}") == 14
    assert (
        "python3 -m tools.archived_closed_pr_retirement --followup --apply" in workflow
    )
