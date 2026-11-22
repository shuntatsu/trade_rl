"""Fail-closed synthetic tests for 13 archived, no-longer-active SD2 reviewer refs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import archived_sd2_review_retirement_20261121 as mod
from tools.branch_hygiene import BranchInfo, OpenPullRequestRefs


def _tip(name: str, sha: str, protected: bool = False) -> BranchInfo:
    return BranchInfo(name, sha, protected)


def _scenario(monkeypatch: pytest.MonkeyPatch) -> Any:
    source = {n: pair[0] for n, pair in mod.PR_SOURCE.items()}
    source.update(mod.REVIEW_BASES)
    refs = [
        _tip("main", "e" * 40),
        _tip(mod.ARCHIVE_BRANCH, mod.ARCHIVE_HEAD),
        *(_tip(n, sha) for n, sha in sorted(source.items())),
    ]
    frozen = {b.name: b.sha for b in refs}
    monkeypatch.setattr(mod, "_archive_manifest", lambda: dict(frozen))
    monkeypatch.setattr(mod, "_git_main", lambda: "e" * 40)
    monkeypatch.setattr(mod, "_is_ancestor", lambda old, target: True)
    monkeypatch.setattr(mod, "require_retirement_origin", lambda: None)
    monkeypatch.setattr(
        mod,
        "_closed_pr_proof",
        lambda api, branch, number: (
            branch.name in mod.PR_SOURCE and mod.PR_SOURCE[branch.name][1] == number
        ),
    )

    class Api:
        _repository = mod.REPOSITORY

        def __init__(self) -> None:
            self.refs = list(refs)
            self.open_refs: tuple[OpenPullRequestRefs, ...] = ()
            self.active = frozenset()
            self.deleted: list[tuple[BranchInfo, ...]] = []
            self.post_insert = False
            self.drift_on_call: int | None = None
            self.calls = 0

        def repository_default_branch(self) -> str:
            return "main"

        def branches(self) -> tuple[BranchInfo, ...]:
            self.calls += 1
            if self.drift_on_call == self.calls:
                self.refs.append(_tip("foreign/concurrent", "a" * 40))
            return tuple(self.refs)

        def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
            return self.open_refs

        def active_workflow_branch_names(self) -> frozenset[str]:
            return self.active

        def delete_branches_with_leases(self, selected: tuple[BranchInfo, ...]) -> None:
            self.deleted.append(tuple(selected))
            self.refs = [b for b in self.refs if b not in selected]
            if self.post_insert:
                self.refs.append(_tip("foreign/reappeared", "a" * 40))

    return Api()


def _created_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    file = tmp_path / "event.json"
    file.write_text(
        json.dumps(
            {
                "action": "created",
                "issue": {"number": 1057},
                "comment": {
                    "body": mod.COMMAND,
                    "user": {"login": mod.OWNER_LOGIN, "id": mod.OWNER_ID},
                },
            }
        ),
        encoding="utf-8",
    )
    for key, value in [
        ("GITHUB_ACTIONS", "true"),
        ("GITHUB_REPOSITORY", mod.REPOSITORY),
        ("GITHUB_ACTOR", mod.OWNER_LOGIN),
        ("GITHUB_EVENT_NAME", "issue_comment"),
        ("GITHUB_EVENT_PATH", str(file)),
        ("GITHUB_SHA", "e" * 40),
    ]:
        monkeypatch.setenv(key, value)
    return file


def test_allowlist_exact_scope_only() -> None:
    assert len(mod.PR_SOURCE) == 8
    assert len(mod.REVIEW_BASES) == 5
    assert mod.EXACT_COUNT == 13
    assert set(mod.MAIN_ANCESTOR_BASES).issubset(mod.REVIEW_BASES)
    names = set(mod.PR_SOURCE) | set(mod.REVIEW_BASES)
    assert len(names) == 13
    assert "feat/issue1146-sd2-runner-20261113" in names
    for n in names:
        assert n.startswith(("review/issue1146-", "feat/issue1146-"))
        assert not n.startswith(("audit/", "research/", "infra/", "fix/m1-"))
    assert mod.COMMAND.endswith(mod.ARCHIVE_HEAD)


def test_preview_selects_13_without_mutating(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _scenario(monkeypatch)
    total, planned, deleted = mod.retire(api, apply=False)
    assert total == 15
    assert len(planned) == 13
    assert not deleted
    assert not api.deleted


def test_owner_apply_atomic_13_and_second_run_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _scenario(monkeypatch)
    _created_event(monkeypatch, tmp_path)
    total, planned, deleted = mod.retire(api, apply=True)
    assert total == 15
    assert len(planned) == 13
    assert planned == deleted
    assert len(api.deleted) == 1 and len(api.deleted[0]) == 13
    assert {b.name for b in api.refs} == {"main", mod.ARCHIVE_BRANCH}
    assert mod.retire(api, apply=True) == (2, (), ())


@pytest.mark.parametrize(
    "ineligible", ["missing", "protected", "active", "open-pr", "unproven-closed-pr"]
)
def test_owner_apply_refuses_partial_13_ref_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ineligible: str
) -> None:
    api = _scenario(monkeypatch)
    _created_event(monkeypatch, tmp_path)
    target = next(iter(mod.PR_SOURCE))
    if ineligible == "missing":
        api.refs = [item for item in api.refs if item.name != target]
    elif ineligible == "protected":
        api.refs = [
            _tip(item.name, item.sha, True) if item.name == target else item
            for item in api.refs
        ]
    elif ineligible == "active":
        api.active = frozenset({target})
    elif ineligible == "open-pr":
        api.open_refs = (
            OpenPullRequestRefs(target, mod.PR_SOURCE[target][0], "main", "e" * 40),
        )
    else:
        original = mod._closed_pr_proof
        monkeypatch.setattr(
            mod,
            "_closed_pr_proof",
            lambda instance, branch, number: (
                branch.name != target and original(instance, branch, number)
            ),
        )
    with pytest.raises(RuntimeError, match="all 13"):
        mod.retire(api, apply=True)
    assert api.deleted == []


def test_owner_apply_refuses_all_ineligible_not_silent_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _scenario(monkeypatch)
    _created_event(monkeypatch, tmp_path)
    api.active = frozenset(mod.PR_SOURCE) | frozenset(mod.REVIEW_BASES)
    with pytest.raises(RuntimeError, match="all 13"):
        mod.retire(api, apply=True)
    assert api.deleted == []


@pytest.mark.parametrize(
    "reason", ["owner-id", "wrong-command", "edited", "wrong-issue"]
)
def test_owner_scope_is_exact_created_comment_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reason: str
) -> None:
    api = _scenario(monkeypatch)
    event = _created_event(monkeypatch, tmp_path)
    obj = json.loads(event.read_text())
    if reason == "owner-id":
        obj["comment"]["user"]["id"] = 0
    elif reason == "wrong-command":
        obj["comment"]["body"] = "/retire-different-command"
    elif reason == "edited":
        obj["action"] = "edited"
    else:
        obj["issue"]["number"] = 1056
    event.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(RuntimeError, match="owner-created exact"):
        mod.retire(api, apply=True)
    assert api.deleted == []


def test_missing_token_refuses_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GitHub Actions token"):
        mod.main([])


def test_live_main_drift_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _scenario(monkeypatch)
    monkeypatch.setattr(mod, "_git_main", lambda: "d" * 40)
    with pytest.raises(RuntimeError, match="live main/archive"):
        mod.retire(api, apply=False)
    assert not api.deleted


@pytest.mark.parametrize(
    "item",
    [
        "review/issue1146-sd2-3037-base-20261120",
        "review/issue1146-sd2-after-main1293-base-20261121",
        "review/issue1146-sd2-owner-docs-after1305-base-20261121",
    ],
)
def test_main_ancestor_bases_must_stay_in_main(
    monkeypatch: pytest.MonkeyPatch, item: str
) -> None:
    api = _scenario(monkeypatch)
    tip = mod.REVIEW_BASES[item]
    monkeypatch.setattr(
        mod,
        "_is_ancestor",
        lambda older, newer: not (older == tip and newer == "e" * 40),
    )
    with pytest.raises(RuntimeError, match="main lost"):
        mod.retire(api, apply=False)
    assert not api.deleted


def test_archive_reachability_loss_blocks_any_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _scenario(monkeypatch)
    monkeypatch.setattr(mod, "_is_ancestor", lambda older, newer: False)
    with pytest.raises(RuntimeError, match="archive lost"):
        mod.retire(api, apply=False)


def test_exact_sha_drift_is_not_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _scenario(monkeypatch)
    target = next(iter(mod.PR_SOURCE))
    api.refs = [_tip(b.name, "f" * 40) if b.name == target else b for b in api.refs]
    with pytest.raises(RuntimeError, match="immutable archive SHA"):
        mod.retire(api, apply=False)
    assert not api.deleted


def test_open_pr_active_ci_and_protected_skip_only_their_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _scenario(monkeypatch)
    source = next(iter(mod.PR_SOURCE))
    active = list(mod.PR_SOURCE)[1]
    protected = list(mod.REVIEW_BASES)[0]
    api.open_refs = (
        OpenPullRequestRefs(source, mod.PR_SOURCE[source][0], "main", "e" * 40),
    )
    api.active = frozenset({active})
    api.refs = [
        _tip(b.name, b.sha, True) if b.name == protected else b for b in api.refs
    ]
    _, planned, _ = mod.retire(api, apply=False)
    assert len(planned) == 10
    for name in (source, active, protected):
        assert name not in planned


def test_closed_pr_no_longer_proven_is_not_deletable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _scenario(monkeypatch)
    target = next(iter(mod.PR_SOURCE))
    original = mod._closed_pr_proof
    monkeypatch.setattr(
        mod,
        "_closed_pr_proof",
        lambda instance, branch, number: (
            branch.name != target and original(instance, branch, number)
        ),
    )
    _, selected, _ = mod.retire(api, apply=False)
    assert len(selected) == 12 and target not in selected


def test_roster_change_before_atomic_push_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _scenario(monkeypatch)
    _created_event(monkeypatch, tmp_path)
    api.drift_on_call = 2
    with pytest.raises(RuntimeError, match="roster changed"):
        mod.retire(api, apply=True)
    assert not api.deleted


def test_post_delete_readback_failure_is_not_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _scenario(monkeypatch)
    _created_event(monkeypatch, tmp_path)
    api.post_insert = True
    with pytest.raises(RuntimeError, match="post-retirement"):
        mod.retire(api, apply=True)


def test_github_sha_mismatch_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _scenario(monkeypatch)
    _created_event(monkeypatch, tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "0" * 40)
    with pytest.raises(RuntimeError, match="event checkout"):
        mod.retire(api, apply=True)
    assert not api.deleted


def test_read_only_cli_preview_then_apply_routes_to_github(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = _scenario(monkeypatch)
    _created_event(monkeypatch, tmp_path)
    monkeypatch.setenv("GH_TOKEN", "synthetic-unit-only-token")
    identity: list[tuple[str, str, str]] = []

    def create_api(*, repository: str, token: str, api_url: str) -> Any:
        identity.append((repository, token, api_url))
        return api

    monkeypatch.setattr(mod, "GitHubApi", create_api)
    assert mod.main([]) == 0
    assert "- eligible: 13" in capsys.readouterr().out
    assert not api.deleted
    assert mod.main(["--apply"]) == 0
    assert "- deleted and verified: 13" in capsys.readouterr().out
    assert mod.main(["--apply"]) == 0
    assert "- deleted and verified: 0" in capsys.readouterr().out
    assert len(identity) == 3


def test_dedicated_workflow_has_nonreusable_exact_created_comment() -> None:
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/retire-merged-pr-branches.yml"
    ).read_text(encoding="utf-8")
    assert "archived-sd2-review-13:" in workflow
    assert mod.COMMAND in workflow
    assert "github.event.action == 'created'" in workflow
    assert "github.event.issue.number == 1057" in workflow
    assert "tools.archived_sd2_review_retirement_20261121 --apply" in workflow
    assert workflow.count("GH_TOKEN: " + "$" + "{{ github.token }}") == 16
