"""Real existing model/inventory compatibility for the hold-only evidence combiner."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from tools.agent_repo.coordination.enrollment import EnrollmentProjection
from tools.agent_repo.coordination.github_state import (
    TaskStatusRecord,
    render_status_comment,
)
from tools.agent_repo.coordination.leases import LeaseRecord, lease_branch_name
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase
from tools.agent_repo.coordination.registry import CoordinatorRegistry
from tools.branch_hygiene import BranchInfo
from tools.github_coordination_registry_snapshot import RegistryRefSnapshot
from tools.github_lease_reconciliation import inspect_expired_lease_evidence

REPOSITORY = "shuntatsu/trade_rl"
TASK = "T1067-01"
BASE = "a" * 40
HEAD = "b" * 40
REF = "c" * 40
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


def _actual_inputs() -> tuple[RegistryRefSnapshot, LeaseRecord, dict[str, object]]:
    branch = lease_branch_name(TASK, 1, "agent-a")
    lease = LeaseRecord(
        task_id=TASK,
        task_revision=1,
        task_contract_digest="d" * 64,
        base_sha=BASE,
        owner="agent-a",
        epoch=1,
        lease_branch=branch,
        head_sha=HEAD,
        last_heartbeat_at=NOW - timedelta(minutes=30),
        expires_at=NOW - timedelta(minutes=15),
        ttl_seconds=900.0,
    )
    status = TaskStatusRecord(
        task_id=TASK,
        task_revision=1,
        task_contract_digest=lease.task_contract_digest,
        phase=TaskPhase.EXECUTING,
        condition=TaskCondition.HEALTHY,
        base_sha=BASE,
        owner=lease.owner,
        lease_epoch=lease.epoch,
        lease_branch=lease.lease_branch,
        head_sha=HEAD,
    )
    entry = EnrollmentProjection(
        repository=REPOSITORY,
        issue_number=1067,
        issue_node_id="I_kwDO1067",
        status=status,
    )
    registry = CoordinatorRegistry(REPOSITORY, (entry,))
    snapshot = RegistryRefSnapshot(
        repository=REPOSITORY,
        repository_node_id="R_kwDOrepo",
        ref_name="refs/heads/coordination/task-registry",
        ref_oid=REF,
        registry=registry,
        registry_digest=registry.digest(),
    )

    class InventoryGet:
        repository = REPOSITORY

        def repository_default_branch(self) -> str:
            return "main"

        def branches(self) -> tuple[BranchInfo, ...]:
            return (
                BranchInfo("main", BASE, False),
                BranchInfo(branch, HEAD, False),
                BranchInfo("coordination/task-registry", REF, False),
                BranchInfo("provenance/branch-retention", "e" * 40, False),
            )

        def open_pulls(self) -> list[object]:
            return []

        def open_issues(self) -> list[object]:
            return [
                {
                    "id": 6142352475,
                    "number": 1067,
                    "node_id": entry.issue_node_id,
                    "repository_url": f"https://api.github.com/repos/{REPOSITORY}",
                    "state": "open",
                    "comments": 1,
                    "updated_at": "2026-11-02T11:50:00Z",
                }
            ]

        def issue_comments(self, issue_number: int) -> list[object]:
            assert issue_number == 1067
            return [
                {
                    "id": 6600000001,
                    "user": {"id": 148511409},
                    "body": render_status_comment(status),
                    "updated_at": "2026-11-02T11:50:00Z",
                }
            ]

        def unfinished_workflow_branch_names(self) -> frozenset[str]:
            return frozenset({branch})

    from tools.github_work_inventory import GitHubWorkApi, build_inventory

    api = InventoryGet()
    typed_api: GitHubWorkApi = api  # type: ignore[assignment]
    inventory = build_inventory(typed_api, include_workflow_refs=True)
    return snapshot, lease, inventory


def test_real_models_and_inventory_still_force_hold() -> None:
    snapshot, lease, inventory = _actual_inputs()
    report = inspect_expired_lease_evidence(
        snapshot,
        lease,
        inventory,
        expected_main_sha=BASE,
        attempts=[],
        artifacts=[],
        now=NOW,
    )
    assert report["task_id"] == TASK
    assert report["registry_ref_oid"] == REF
    assert report["unfinished_workflow_branch_observed"] is True
    assert report["sampled_run_ids"] == []
    assert report["decision"] == "HOLD"
    assert report["lease_reassignment_authorized"] is False
    assert report["cleanup_authorized"] is False
    assert inventory["task_projection_trusted"] is False


def test_real_lease_head_drift_refuses_even_with_stable_inventory() -> None:
    snapshot, lease, inventory = _actual_inputs()
    moved = replace(lease, head_sha="f" * 40)
    with pytest.raises(ValueError, match="identities conflict"):
        inspect_expired_lease_evidence(
            snapshot,
            moved,
            inventory,
            expected_main_sha=BASE,
            attempts=[],
            artifacts=[],
            now=NOW,
        )


def test_native_producer_observations_with_jobs_and_artifacts_still_hold() -> None:
    from copy import deepcopy

    from tools.github_workflow_artifacts import observe_workflow_artifacts
    from tools.github_workflow_attempts import observe_workflow_attempt

    snapshot, lease, inventory = _actual_inputs()
    run_id = 1234
    repo_id = 123
    run = {
        "id": run_id,
        "run_attempt": 2,
        "head_branch": lease.lease_branch,
        "head_sha": HEAD,
        "status": "completed",
        "conclusion": "success",
        "updated_at": "2026-11-02T11:50:00Z",
        "repository": {"id": repo_id},
        "head_repository": {"id": repo_id},
    }
    job = {
        "id": 41,
        "run_id": run_id,
        "run_attempt": 2,
        "head_branch": lease.lease_branch,
        "head_sha": HEAD,
        "name": "core",
        "status": "completed",
        "conclusion": "success",
    }
    artifact = {
        "id": 42,
        "node_id": "A_42",
        "name": "receipt",
        "size_in_bytes": 128,
        "expired": False,
        "digest": "sha256:" + "f" * 64,
        "workflow_run": {
            "id": run_id,
            "repository_id": repo_id,
            "head_repository_id": repo_id,
            "head_branch": lease.lease_branch,
            "head_sha": HEAD,
        },
    }

    class WorkflowGet:
        def _request_json(
            self, method: str, suffix: str, *, query: dict[str, str] | None = None
        ) -> object:
            assert method == "GET"
            if suffix == "":
                return {"id": repo_id, "node_id": "R_123", "full_name": REPOSITORY}
            if suffix == f"actions/runs/{run_id}":
                return deepcopy(run)
            assert query == {"page": "1", "per_page": "100"}
            if suffix == f"actions/runs/{run_id}/attempts/2/jobs":
                return {"total_count": 1, "jobs": [deepcopy(job)]}
            if suffix == f"actions/runs/{run_id}/artifacts":
                return {"total_count": 1, "artifacts": [deepcopy(artifact)]}
            raise AssertionError(f"Unexpected endpoint: {suffix}")

    api = WorkflowGet()
    attempt = observe_workflow_attempt(api, run_id=run_id)
    artifacts = observe_workflow_artifacts(
        api, expected_repository=REPOSITORY, run_id=run_id
    )
    report = inspect_expired_lease_evidence(
        snapshot,
        lease,
        inventory,
        expected_main_sha=BASE,
        attempts=[attempt],
        artifacts=[artifacts],
        now=NOW,
    )
    assert report["sampled_run_ids"] == [run_id]
    assert report["sampled_artifact_ids"] == [42]
    assert report["supplied_evidence_consistent"] is True
    assert report["decision"] == "HOLD"
    for flag in (
        "trusted_roster_complete",
        "authoritative_registry_established",
        "lease_reassignment_authorized",
        "cleanup_authorized",
    ):
        assert report[flag] is False
