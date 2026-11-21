"""Fail-closed synthetic branch pointer retirement, never economic execution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import archived_superseded_sd2_retirement as mod
from tools.branch_hygiene import BranchInfo, OpenPullRequestRefs

MAIN = "1" * 40
SUCC = "b" * 40


def _ref(name: str, sha: str, protected: bool = False) -> BranchInfo:
    return BranchInfo(name, sha, protected)


def _fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, tuple[BranchInfo, ...]]:
    entries = (
        _ref("main", MAIN),
        _ref(mod.ARCHIVE_BRANCH, mod.ARCHIVE_HEAD),
        _ref(mod.SUCCEEDING_REF, SUCC),
        *(_ref(n, sha) for n, sha in sorted(mod.EXACT.items())),
    )
    monkeypatch.setattr(
        mod, "_archive_manifest", lambda: {b.name: b.sha for b in entries}
    )
    monkeypatch.setattr(mod, "_git_main", lambda: MAIN)
    monkeypatch.setattr(mod, "_git_ancestor", lambda a, b: True)
    monkeypatch.setattr(mod, "reachable_commits", lambda a: frozenset({MAIN}))

    class Api:
        _repository = mod.REPOSITORY

        def __init__(self) -> None:
            self.refs = list(entries)
            self.pr_state = "open"
            self.pr_sha = SUCC
            self.active = frozenset()
            self.opens: tuple[OpenPullRequestRefs, ...] = ()
            self.deleted: list[tuple[BranchInfo, ...]] = []

        def repository_default_branch(self) -> str:
            return "main"

        def branches(self) -> tuple[BranchInfo, ...]:
            return tuple(self.refs)

        def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
            return self.opens

        def active_workflow_branch_names(self) -> frozenset[str]:
            return self.active

        def _request_json(self, method: str, suffix: str) -> dict[str, Any]:
            assert method == "GET" and suffix == "pulls/1247"
            return {
                "number": 1247,
                "state": self.pr_state,
                "merged_at": "2026-11-20T10:00:00Z"
                if self.pr_state == "closed"
                else None,
                "head": {
                    "ref": mod.SUCCEEDING_REF,
                    "sha": self.pr_sha,
                    "repo": {"full_name": mod.REPOSITORY},
                },
                "base": {"ref": "main", "repo": {"full_name": mod.REPOSITORY}},
            }

        def delete_branches_with_leases(self, refs: tuple[BranchInfo, ...]) -> None:
            self.deleted.append(refs)
            self.refs = [r for r in self.refs if r not in refs]

    return Api(), entries


def _owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "action": "created",
                "issue": {"number": 1057},
                "comment": {
                    "body": mod.COMMAND,
                    "user": {"login": "shuntatsu", "id": mod.OWNER_ID},
                },
            }
        ),
        encoding="utf-8",
    )
    for k, v in (
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", mod.REPOSITORY),
        ("GITHUB_ACTOR", "shuntatsu"),
        ("GITHUB_EVENT_NAME", "issue_comment"),
        ("GITHUB_EVENT_PATH", str(event)),
        ("GITHUB_SHA", MAIN),
    ):
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            "https://github.com/shuntatsu/trade_rl.git",
        ],
        check=True,
    )
    monkeypatch.chdir(repo)


def test_exact_three_only() -> None:
    assert len(mod.EXACT) == 3
    assert len(set(mod.EXACT.values())) == 3
    assert set(mod.EXACT) == {
        "fix/issue1146-sd2-manifest-first-array-gate-20261113",
        "review/issue1146-bounded-codeonly-base-20261114",
        "review/issue1146-pre-budget-source-20261114",
    }


def test_successor_identity_and_pr_state_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, refs = _fixture(monkeypatch)
    assert mod._select(api, refs) == refs[3:]
    api.pr_sha = "f" * 40
    with pytest.raises(RuntimeError, match="live branch SHA"):
        mod._select(api, refs)
    api.pr_sha = SUCC
    api.pr_state = "abandoned"
    with pytest.raises(RuntimeError, match="abandoned"):
        mod._select(api, refs)


def test_successor_lost_history_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    api, refs = _fixture(monkeypatch)
    monkeypatch.setattr(mod, "_git_ancestor", lambda a, b: b == mod.ARCHIVE_HEAD)
    with pytest.raises(RuntimeError, match="lost the superseded"):
        mod._select(api, refs)


def test_preserves_open_pr_ci_protected_and_ref_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, refs = _fixture(monkeypatch)
    a, b, c = refs[3:]
    api.opens = (OpenPullRequestRefs(a.name, a.sha, "main", MAIN),)
    api.active = frozenset({b.name})
    protected = (*refs[:5], _ref(c.name, c.sha, True))
    assert mod._select(api, protected) == ()
    api.opens = ()
    api.active = frozenset()
    changed = (*refs[:3], _ref(a.name, "f" * 40), b, c)
    assert mod._select(api, changed) == (b, c)
    with pytest.raises(RuntimeError, match="archive"):
        mod._select(api, (refs[0], *refs[2:]))


def test_owner_created_event_and_wrong_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _owner(monkeypatch, tmp_path)
    mod._require_owner_event()
    path = tmp_path / "event.json"
    obj = json.loads(path.read_text())
    obj["action"] = "edited"
    path.write_text(json.dumps(obj))
    with pytest.raises(RuntimeError, match="incorrect owner"):
        mod._require_owner_event()


def test_actual_origin_and_atomic_batch_readback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, refs = _fixture(monkeypatch)
    before, selected, deleted = mod.retire(api, apply=False)
    assert before == 6 and len(selected) == 3 and deleted == ()
    _owner(monkeypatch, tmp_path)
    before, selected, deleted = mod.retire(api, apply=True)
    assert before == 6 and len(selected) == len(deleted) == 3
    assert api.deleted == [refs[3:]]
    assert api.refs == list(refs[:3])


def test_wrong_post_delete_readback_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, _ = _fixture(monkeypatch)
    _owner(monkeypatch, tmp_path)
    original = api.delete_branches_with_leases

    def bad(refs: tuple[BranchInfo, ...]) -> None:
        original(refs)
        api.refs.append(_ref("fix/foreign-new", "f" * 40))

    api.delete_branches_with_leases = bad
    with pytest.raises(RuntimeError, match="post-delete"):
        mod.retire(api, apply=True)


def test_workflow_exact_owner_command_and_valid_token_count() -> None:
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/retire-merged-pr-branches.yml"
    ).read_text()
    assert "superseded-sd2-ref-retirement:" in workflow
    assert "github.event.action == 'created'" in workflow
    assert mod.COMMAND in workflow
    assert workflow.count("GH_TOKEN: " + "$" + "{{ github.token }}") == 14
