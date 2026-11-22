"""Source-bound read-only Task enrollment consistency tests."""

from __future__ import annotations

import base64
import hashlib
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tools.agent_repo.coordination.enrollment import EnrollmentProjection
from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import LeaseRecord, lease_branch_name
from tools.agent_repo.coordination.model import (
    ExecutionMode,
    TaskCondition,
    TaskPacket,
    TaskPhase,
    WriteScope,
)
from tools.agent_repo.coordination.registry import CoordinatorRegistry
from tools.github_coordination_enrollment_audit import audit_task_enrollment

REPO = "shuntatsu/trade-rl-coordination-disposable"
REF = "refs/heads/coordination/task-registry"
PATH = "coordination/registry.json"
MAIN, BRANCH, COMMIT, TREE = (letter * 40 for letter in "abcd")
NOW = datetime(2026, 11, 22, tzinfo=timezone.utc)
TASK = "T1067-01"
ISSUE = 1067


def inputs() -> tuple[TaskPacket, LeaseRecord, TaskStatusRecord]:
    packet = TaskPacket(
        task_id=TASK,
        task_revision=1,
        parent_issue=ISSUE,
        title="Task",
        objective="Secure registration",
        execution_mode=ExecutionMode.WRITE,
        base_sha=MAIN,
        write_scope=WriteScope(allow=("tools/",)),
    )
    lease = LeaseRecord(
        task_id=TASK,
        task_revision=1,
        task_contract_digest=packet.contract_digest(),
        base_sha=MAIN,
        owner="worker-a",
        epoch=1,
        lease_branch=lease_branch_name(TASK, 1, "worker-a"),
        head_sha=BRANCH,
        last_heartbeat_at=NOW - timedelta(minutes=5),
        expires_at=NOW + timedelta(minutes=5),
        ttl_seconds=600,
    )
    status = TaskStatusRecord(
        task_id=TASK,
        task_revision=1,
        task_contract_digest=packet.contract_digest(),
        phase=TaskPhase.EXECUTING,
        condition=TaskCondition.HEALTHY,
        base_sha=MAIN,
        owner=lease.owner,
        lease_epoch=1,
        lease_branch=lease.lease_branch,
        head_sha=BRANCH,
    )
    return packet, lease, status


class FakeGitHub:
    def __init__(self, registry: CoordinatorRegistry | None = None) -> None:
        self.registry = registry or CoordinatorRegistry(REPO)
        self.data = self.registry.canonical_bytes()
        self.blob = hashlib.sha1(
            b"blob " + str(len(self.data)).encode() + b"\x00" + self.data
        ).hexdigest()
        self.calls: list[tuple[str, str]] = []
        self.counts: Counter[str] = Counter()
        self.drifts: dict[str, tuple[int, dict[str, Any]]] = {}
        self.fail: str | None = None

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        self.calls.append((method, suffix))
        self.counts[suffix] += 1
        if suffix == self.fail:
            raise RuntimeError("GitHub API unavailable")
        if suffix == "":
            data: dict[str, Any] = {
                "id": 123,
                "node_id": "R_exact",
                "full_name": REPO,
                "default_branch": "main",
            }
        elif suffix == f"issues/{ISSUE}":
            data = {
                "id": 456,
                "number": ISSUE,
                "node_id": "I_issue",
                "state": "open",
                "repository_url": f"https://api.github.com/repos/{REPO}",
                "updated_at": "2026-11-22T10:00:00Z",
            }
        elif suffix.startswith("git/ref/"):
            ref = "refs/" + suffix.removeprefix("git/ref/")
            sha = {
                "refs/heads/main": MAIN,
                "refs/heads/" + inputs()[1].lease_branch: BRANCH,
                REF: COMMIT,
            }[ref]
            data = {"ref": ref, "object": {"type": "commit", "sha": sha}}
        elif suffix == f"git/commits/{COMMIT}":
            data = {"sha": COMMIT, "tree": {"sha": TREE}}
        elif suffix == f"git/trees/{TREE}":
            assert query == {"recursive": "1"}
            data = {
                "sha": TREE,
                "truncated": False,
                "tree": [
                    {
                        "path": PATH,
                        "type": "blob",
                        "mode": "100644",
                        "sha": self.blob,
                        "size": len(self.data),
                    }
                ],
            }
        elif suffix == f"git/blobs/{self.blob}":
            data = {
                "sha": self.blob,
                "size": len(self.data),
                "encoding": "base64",
                "content": base64.b64encode(self.data).decode(),
            }
        else:
            raise AssertionError(f"unexpected GET {suffix}")
        if suffix in self.drifts:
            threshold, change = self.drifts[suffix]
            if self.counts[suffix] >= threshold:
                data.update(change)
        return data


def audit(api: FakeGitHub, **overrides: object):
    packet, lease, status = inputs()
    kwargs: dict[str, Any] = dict(
        packet=packet,
        lease=lease,
        status=status,
        repository=REPO,
        registry_ref=REF,
        registry_path=PATH,
        expected_issue_node_id="I_issue",
        now=NOW,
    )
    kwargs.update(overrides)
    return audit_task_enrollment(api, **kwargs)


def test_binds_stable_issue_main_branch_and_registry_without_authority() -> None:
    api = FakeGitHub()
    observation = audit(api)
    assert observation.plan.operation == "create"
    assert observation.registry_oid == COMMIT
    assert observation.registry_digest == api.registry.digest()
    assert observation.projection.issue_node_id == "I_issue"
    assert observation.projection.status.head_sha == BRANCH
    assert observation.main_sha == MAIN and observation.branch_sha == BRANCH
    assert observation.stable_readback_verified is True
    assert observation.authoritative_registry_established is False
    assert observation.task_registry_complete is False
    assert observation.enrollment_persisted is False
    assert observation.lease_reassignment_authorized is False
    assert observation.cleanup_authorized is False
    assert all(method == "GET" for method, _ in api.calls)
    assert api.counts[f"issues/{ISSUE}"] == 2
    assert api.counts["git/ref/heads/main"] == 2
    assert api.counts["git/ref/heads/" + inputs()[1].lease_branch] == 2
    assert api.counts["git/ref/heads/coordination/task-registry"] == 3


def test_existing_task_phase_update_preserves_other_registry_entries() -> None:
    _, _, status = inputs()
    prior = EnrollmentProjection(REPO, ISSUE, "I_issue", status)
    # Unrelated task must retain its exact issue, digest, phase and lease fence.
    unrelated_status = replace(
        status,
        task_id="T1067-02",
        task_contract_digest="e" * 64,
        lease_branch=lease_branch_name("T1067-02", 1, "worker-a"),
    )
    unrelated = EnrollmentProjection(REPO, ISSUE + 1, "I_unrelated", unrelated_status)
    api = FakeGitHub(CoordinatorRegistry(REPO, (unrelated, prior)))
    updated = replace(status, phase=TaskPhase.REVIEW)

    observation = audit(api, status=updated)

    assert observation.plan.operation == "update"
    assert observation.plan.after_registry.entries == (
        EnrollmentProjection(REPO, ISSUE, "I_issue", updated),
        unrelated,
    )
    assert observation.plan.after_registry.digest() == observation.plan.after_digest
    assert observation.plan.cleanup_authorized is False


@pytest.mark.parametrize(
    ("suffix", "change", "threshold"),
    [
        ("", {"node_id": "R_new"}, 4),
        (f"issues/{ISSUE}", {"updated_at": "new"}, 2),
        (f"issues/{ISSUE}", {"node_id": "I_other"}, 2),
        ("git/ref/heads/main", {"object": {"type": "commit", "sha": "e" * 40}}, 2),
        (
            "git/ref/heads/" + inputs()[1].lease_branch,
            {"object": {"type": "commit", "sha": "e" * 40}},
            2,
        ),
    ],
)
def test_repository_issue_main_or_branch_drift_rejected(
    suffix: str, change: dict[str, Any], threshold: int
) -> None:
    api = FakeGitHub()
    api.drifts[suffix] = (threshold, change)
    with pytest.raises(ValueError, match="changed during enrollment"):
        audit(api)
    assert all(method == "GET" for method, _ in api.calls)


def test_registry_changes_after_its_inner_double_read() -> None:
    api = FakeGitHub()
    api.drifts["git/ref/heads/coordination/task-registry"] = (
        3,
        {"object": {"type": "commit", "sha": "e" * 40}},
    )
    with pytest.raises(ValueError, match="registry ref changed"):
        audit(api)


@pytest.mark.parametrize(
    "change",
    [
        {"state": "closed"},
        {"node_id": "I_other"},
        {"number": 9999},
        {"repository_url": "https://api.github.com/repos/other/repo"},
        {"pull_request": {"url": "dummy"}},
    ],
)
def test_closed_foreign_or_pr_issue_rejected(change: dict[str, Any]) -> None:
    api = FakeGitHub()
    api.drifts[f"issues/{ISSUE}"] = (1, change)
    with pytest.raises(ValueError, match="Issue"):
        audit(api)


@pytest.mark.parametrize(
    ("suffix", "change"),
    [
        ("", {"full_name": "other/repo"}),
        ("", {"default_branch": "develop"}),
        ("git/ref/heads/main", {"object": {"type": "tag", "sha": MAIN}}),
        ("git/ref/heads/main", {"ref": "refs/heads/not-main"}),
    ],
)
def test_wrong_repository_default_branch_or_main_type_rejected(
    suffix: str, change: dict[str, Any]
) -> None:
    api = FakeGitHub()
    api.drifts[suffix] = (1, change)
    with pytest.raises(ValueError):
        audit(api)


def test_malformed_status_stale_main_and_duplicate_record_rejected() -> None:
    packet, _, status = inputs()
    with pytest.raises(ValueError, match="status, packet and lease"):
        audit(FakeGitHub(), status=replace(status, task_contract_digest="f" * 64))
    api = FakeGitHub()
    api.drifts["git/ref/heads/main"] = (
        1,
        {"object": {"type": "commit", "sha": "e" * 40}},
    )
    with pytest.raises(ValueError, match="main has changed"):
        audit(api, packet=packet)
    prior = EnrollmentProjection(REPO, ISSUE, "I_issue", status)
    with pytest.raises(ValueError, match="no-op"):
        audit(FakeGitHub(CoordinatorRegistry(REPO, (prior,))))


def test_missing_branch_or_api_visibility_never_grants_registration() -> None:
    for suffix in [
        f"issues/{ISSUE}",
        "git/ref/heads/" + inputs()[1].lease_branch,
        "git/blobs/" + FakeGitHub().blob,
    ]:
        api = FakeGitHub()
        api.fail = suffix
        with pytest.raises(RuntimeError, match="API unavailable"):
            audit(api)
        assert all(method == "GET" for method, _ in api.calls)


def test_invalid_task_mode_or_time_is_not_enrollment() -> None:
    packet, _, _ = inputs()
    api = FakeGitHub()
    with pytest.raises(ValueError):
        audit(api, now=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="read_only"):
        audit(
            api,
            packet=replace(
                packet, execution_mode=ExecutionMode.READ_ONLY, write_scope=WriteScope()
            ),
        )
    assert all(method == "GET" for method, _ in api.calls)
