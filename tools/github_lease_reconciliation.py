"""Read-only cross-evidence preflight for an expired Coordinator write lease.

This compares *supplied* point-in-time observations only. Their transport and
completeness are not authenticated here, and even perfect consistency NEVER
permits a lease epoch change, Task mutation or branch deletion. In particular,
never construct a LeaseReconciliation/REASSIGN grant from this output.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tools.agent_repo.coordination.leases import LeaseRecord
    from tools.github_coordination_registry_snapshot import RegistryRefSnapshot

_SHA = re.compile(r"^[0-9a-f]{40}$")
_REF = re.compile(r"^refs/heads/coordination/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _rows(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an ordered sequence")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value) or value == "0" * 40:
        raise ValueError(f"{label} must be an exact Git SHA")
    return value


def _id(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive native integer")
    return value


def _false(value: Mapping[str, Any], *keys: str) -> None:
    for key in keys:
        if value.get(key) is not False:
            raise ValueError(f"{key} must remain false")


def _unique_strings(value: object, label: str) -> list[str]:
    names: list[str] = []
    for name in _rows(value, label):
        if not isinstance(name, str) or type(name) is not str or not name.strip():
            raise ValueError(f"{label} is malformed, duplicated or unordered")
        names.append(name)
    if len(set(names)) != len(names) or names != sorted(names):
        raise ValueError(f"{label} is malformed, duplicated or unordered")
    return names


def _run_state(value: Mapping[str, Any]) -> None:
    state = value.get("status")
    if type(state) is not str or state not in {
        "requested",
        "waiting",
        "pending",
        "queued",
        "in_progress",
        "completed",
    }:
        raise ValueError("attempt status is unknown")
    conclusion = value.get("conclusion")
    if state == "completed":
        if type(conclusion) is not str or conclusion not in {
            "success",
            "failure",
            "cancelled",
            "skipped",
            "neutral",
            "timed_out",
            "action_required",
            "stale",
        }:
            raise ValueError("completed attempt conclusion is unknown")
    elif conclusion is not None:
        raise ValueError("unfinished attempt cannot have a conclusion")


def inspect_expired_lease_evidence(
    snapshot: RegistryRefSnapshot,
    lease: LeaseRecord,
    inventory: Mapping[str, Any],
    *,
    expected_main_sha: str,
    attempts: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    """Compare bounded supplied inputs, reject inconsistencies, always HOLD.

    A matching supplied roster does NOT prove that all Tasks, closed Issues,
    PRs, workflow runs/attempts or deleted/expired artifacts were enumerated.
    Caller must independently authenticate evidence and detect changes between
    source reads; no combined atomic snapshot is possible here.
    """
    _sha(expected_main_sha, "expected_main_sha")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if now < lease.expires_at:
        raise ValueError("write lease is not expired")
    if lease.head_sha is None:
        raise ValueError("lease has no exact branch tip")
    _sha(lease.head_sha, "lease head_sha")

    if (
        snapshot.ref_readback_verified is not True
        or snapshot.authoritative_registry_established is not False
        or snapshot.task_registry_complete is not False
        or snapshot.cleanup_authorized is not False
        or snapshot.registry.repository != snapshot.repository
        or snapshot.registry.digest() != snapshot.registry_digest
        or type(snapshot.repository_node_id) is not str
        or not snapshot.repository_node_id.strip()
    ):
        raise ValueError("registry snapshot is malformed or claims authority")
    _sha(snapshot.ref_oid, "registry ref_oid")
    if (
        type(snapshot.ref_name) is not str
        or not _REF.fullmatch(snapshot.ref_name)
        or ".." in snapshot.ref_name
        or snapshot.ref_name.endswith(".lock")
        or any(part in {".", ".."} for part in snapshot.ref_name.split("/"))
    ):
        raise ValueError("registry ref is not a safe coordination ref")
    matches = [
        entry
        for entry in snapshot.registry.entries
        if entry.status.task_id == lease.task_id
    ]
    if len(matches) != 1:
        raise ValueError("Task is missing or ambiguous in registry")
    entry = matches[0]
    status = entry.status
    if (
        entry.repository != snapshot.repository
        or status.task_revision != lease.task_revision
        or status.task_contract_digest != lease.task_contract_digest
        or status.base_sha != lease.base_sha
        or status.owner != lease.owner
        or status.lease_epoch != lease.epoch
        or status.lease_branch != lease.lease_branch
        or status.head_sha != lease.head_sha
    ):
        raise ValueError("registry Task and expired lease identities conflict")
    if lease.base_sha != expected_main_sha:
        raise ValueError("current main differs from lease base")

    observed = _object(inventory, "work inventory")
    if (
        observed.get("schema") != "github_work_inventory_v1"
        or observed.get("repository") != snapshot.repository
        or observed.get("main_sha") != expected_main_sha
        or observed.get("unfinished_workflow_branch_names_checked") is not True
        or observed.get("task_scope") != "open_issue_comments_only"
    ):
        raise ValueError("work inventory scope, main, or repository differs")
    _false(
        observed,
        "task_projection_trusted",
        "task_registry_complete",
        "workflow_activity_checked",
        "cleanup_authorized",
    )

    branches = _rows(observed.get("branches"), "branch roster")
    seen: dict[str, str] = {}
    for raw in branches:
        branch = _object(raw, "branch")
        name = branch.get("name")
        if type(name) is not str or not name.strip() or name in seen:
            raise ValueError("branch roster is malformed or duplicated")
        seen[name] = _sha(branch.get("sha"), "branch tip")
    if seen.get("main") != expected_main_sha:
        raise ValueError("main branch tip is missing or changed")
    if seen.get(lease.lease_branch) != lease.head_sha:
        raise ValueError("lease branch is missing or tip changed")
    registry_branch = snapshot.ref_name.removeprefix("refs/heads/")
    if seen.get(registry_branch) != snapshot.ref_oid:
        raise ValueError("registry ref is missing or tip differs from inventory")
    active = _unique_strings(
        observed.get("unfinished_workflow_branches"), "unfinished workflow branches"
    )
    if not set(active).issubset(seen):
        raise ValueError("unfinished workflow refs contain unknown branches")

    open_prs: list[int] = []
    pr_numbers: set[int] = set()
    for raw in _rows(observed.get("pull_requests"), "open PR roster"):
        pr = _object(raw, "open PR")
        number = _id(pr.get("number"), "PR number")
        if number in pr_numbers:
            raise ValueError("open PR roster contains duplicate numbers")
        pr_numbers.add(number)
        head_ref = pr.get("head_ref")
        if head_ref is not None:
            if type(head_ref) is not str or not head_ref.strip():
                raise ValueError("open PR head ref is malformed")
            if seen.get(head_ref) != _sha(pr.get("head_sha"), "PR head_sha"):
                raise ValueError("open PR head is missing or its tip changed")
        if head_ref == lease.lease_branch:
            open_prs.append(number)
    open_pr_count = observed.get("open_pr_count")
    if (
        type(open_pr_count) is not int
        or open_pr_count < 0
        or open_pr_count != len(pr_numbers)
    ):
        raise ValueError("open_pr_count differs from current open PR roster")

    pairs: dict[int, Mapping[str, Any]] = {}
    for raw in _rows(attempts, "attempt observations"):
        item = _object(raw, "attempt observation")
        run = _id(item.get("run_id"), "workflow run id")
        if run in pairs:
            raise ValueError("duplicate workflow attempt run id")
        if (
            item.get("schema") != "github_workflow_attempt_observation_v1"
            or item.get("scope") != "single_run_attempt"
            or item.get("job_source") != "attempt_scoped_only"
            or item.get("head_branch") != lease.lease_branch
            or item.get("head_sha") != lease.head_sha
        ):
            raise ValueError("attempt observation differs from lease")
        _id(item.get("run_attempt"), "run attempt")
        _run_state(item)
        _false(
            item,
            "workflow_activity_checked",
            "task_registry_complete",
            "cleanup_authorized",
        )
        jobs = _rows(item.get("jobs"), "run attempt jobs")
        if (
            type(item.get("job_count")) is not int
            or item["job_count"] < 1
            or len(jobs) != item["job_count"]
        ):
            raise ValueError("run attempt jobs are absent or incomplete")
        job_ids = [_id(_object(job, "job").get("id"), "job id") for job in jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("run attempt job ids are duplicated")
        pairs[run] = item
    if (
        any(item["status"] != "completed" for item in pairs.values())
        and lease.lease_branch not in active
    ):
        raise ValueError("unfinished workflow branch missing from inventory")

    artifact_runs: dict[int, Mapping[str, Any]] = {}
    artifact_ids: set[int] = set()
    artifact_repository_id: int | None = None
    for raw in _rows(artifacts, "artifact observations"):
        item = _object(raw, "artifact observation")
        run = _id(item.get("run_id"), "artifact run id")
        if run in artifact_runs:
            raise ValueError("duplicate artifact run observation")
        attempt = pairs.get(run)
        if attempt is None:
            raise ValueError("artifact run has no matched attempt observation")
        observed_attempt = _id(item.get("run_attempt_observed"), "artifact run attempt")
        if (
            item.get("schema") != "github_workflow_artifact_observation_v1"
            or item.get("scope") != "single_run_all_attempts_current_artifacts"
            or item.get("repository") != snapshot.repository
            or item.get("head_branch") != attempt["head_branch"]
            or item.get("head_sha") != attempt["head_sha"]
            or observed_attempt != attempt["run_attempt"]
            or item.get("status") != attempt.get("status")
            or item.get("conclusion") != attempt.get("conclusion")
            or item.get("artifact_metadata_double_read") is not True
        ):
            raise ValueError("artifact observation differs from run attempt")
        repo_id = _id(item.get("repository_id"), "artifact repository id")
        if artifact_repository_id is None:
            artifact_repository_id = repo_id
        elif repo_id != artifact_repository_id:
            raise ValueError("artifact reports disagree on repository id")
        _false(
            item,
            "artifact_attempt_binding_verified",
            "artifact_bytes_verified",
            "artifact_history_complete",
            "workflow_activity_checked",
            "task_registry_complete",
            "lease_reassignment_authorized",
            "cleanup_authorized",
        )
        entries = _rows(item.get("artifacts"), "run artifacts")
        if (
            type(item.get("artifact_count")) is not int
            or len(entries) != item["artifact_count"]
        ):
            raise ValueError("artifact count differs from supplied roster")
        for art in entries:
            aid = _id(_object(art, "artifact").get("id"), "artifact id")
            if aid in artifact_ids:
                raise ValueError("duplicate artifact id across runs")
            artifact_ids.add(aid)
        artifact_runs[run] = item
    if set(artifact_runs) != set(pairs):
        raise ValueError("attempt run has no matched artifact observation")

    # Deliberately NEVER instantiate LeaseReconciliation or return a grant.
    return {
        "schema": "github_expired_lease_evidence_hold_v1",
        "repository": snapshot.repository,
        "task_id": lease.task_id,
        "issue_number_unverified": entry.issue_number,
        "issue_node_id_unverified": entry.issue_node_id,
        "lease_epoch": lease.epoch,
        "lease_branch": lease.lease_branch,
        "lease_head_sha": lease.head_sha,
        "registry_ref_oid": snapshot.ref_oid,
        "observed_open_pr_numbers": sorted(open_prs),
        "unfinished_workflow_branch_observed": lease.lease_branch in active,
        "sampled_run_ids": sorted(pairs),
        "sampled_artifact_ids": sorted(artifact_ids),
        "supplied_evidence_consistent": True,
        "decision": "HOLD",
        "trusted_roster_complete": False,
        "authoritative_registry_established": False,
        "lease_reassignment_authorized": False,
        "cleanup_authorized": False,
        "unresolved_requirements": [
            "authenticated_coordinator_and_registry",
            "trusted_complete_task_and_closed_issue_roster",
            "complete_workflow_run_attempt_job_inventory",
            "complete_historical_artifact_inventory_and_bytes",
            "atomic_freshness_or_reconciliation_transaction",
            "real_disposable_github_concurrency_and_interruption_e2e",
        ],
    }


__all__ = ["inspect_expired_lease_evidence"]
