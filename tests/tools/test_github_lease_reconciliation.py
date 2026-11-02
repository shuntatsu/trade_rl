"""Negative and hold-only oracles for Issue #1067 cross-evidence preflight."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tools.github_lease_reconciliation import inspect_expired_lease_evidence

REPO = "shuntatsu/trade_rl"
BASE = "a" * 40
HEAD = "b" * 40
REF = "c" * 40
TASK = "T1067-01"
BRANCH = "agent/T1067-01/e0001-agent-a"
RUN = 1234
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


class FakeRegistry:
    def __init__(self, entries: list[object]) -> None:
        self.repository = REPO
        self.entries = entries

    def digest(self) -> str:
        return "d" * 64


def _evidence() -> dict[str, object]:
    status = SimpleNamespace(
        task_id=TASK,
        task_revision=1,
        task_contract_digest="e" * 64,
        base_sha=BASE,
        owner="agent-a",
        lease_epoch=1,
        lease_branch=BRANCH,
        head_sha=HEAD,
    )
    lease = SimpleNamespace(
        task_id=TASK,
        task_revision=1,
        task_contract_digest="e" * 64,
        base_sha=BASE,
        owner="agent-a",
        epoch=1,
        lease_branch=BRANCH,
        head_sha=HEAD,
        expires_at=NOW - timedelta(minutes=1),
    )
    registry = FakeRegistry(
        [
            SimpleNamespace(
                repository=REPO,
                issue_number=1067,
                issue_node_id="I_1067",
                status=status,
            )
        ]
    )
    snap = SimpleNamespace(
        repository=REPO,
        repository_node_id="R_repository",
        ref_name="refs/heads/coordination/task-registry",
        ref_oid=REF,
        registry=registry,
        registry_digest=registry.digest(),
        ref_readback_verified=True,
        authoritative_registry_established=False,
        task_registry_complete=False,
        cleanup_authorized=False,
    )
    inventory = {
        "schema": "github_work_inventory_v1",
        "repository": REPO,
        "main_sha": BASE,
        "task_scope": "open_issue_comments_only",
        "unfinished_workflow_branch_names_checked": True,
        "task_projection_trusted": False,
        "task_registry_complete": False,
        "workflow_activity_checked": False,
        "cleanup_authorized": False,
        "branches": [
            {"name": "main", "sha": BASE, "reason": "default"},
            {"name": BRANCH, "sha": HEAD, "reason": "open_pr_ref"},
        ],
        "unfinished_workflow_branches": [BRANCH],
        "pull_requests": [{"number": 17, "head_ref": BRANCH, "head_sha": HEAD}],
    }
    attempt = {
        "schema": "github_workflow_attempt_observation_v1",
        "run_id": RUN,
        "run_attempt": 2,
        "head_branch": BRANCH,
        "head_sha": HEAD,
        "status": "completed",
        "conclusion": "failure",
        "job_source": "attempt_scoped_only",
        "scope": "single_run_attempt",
        "job_count": 1,
        "jobs": [{"id": 110, "name": "core", "status": "completed"}],
        "workflow_activity_checked": False,
        "task_registry_complete": False,
        "cleanup_authorized": False,
    }
    artifact = {
        "schema": "github_workflow_artifact_observation_v1",
        "repository": REPO,
        "repository_id": 123,
        "run_id": RUN,
        "run_attempt_observed": 2,
        "head_branch": BRANCH,
        "head_sha": HEAD,
        "status": "completed",
        "conclusion": "failure",
        "scope": "single_run_all_attempts_current_artifacts",
        "artifact_metadata_double_read": True,
        "artifact_count": 1,
        "artifacts": [{"id": 21, "name": "receipt"}],
        "artifact_attempt_binding_verified": False,
        "artifact_bytes_verified": False,
        "artifact_history_complete": False,
        "workflow_activity_checked": False,
        "task_registry_complete": False,
        "lease_reassignment_authorized": False,
        "cleanup_authorized": False,
    }
    return {
        "snapshot": snap,
        "lease": lease,
        "inventory": inventory,
        "expected_main_sha": BASE,
        "attempts": [attempt],
        "artifacts": [artifact],
        "now": NOW,
    }


def _inspect(e: dict[str, object]) -> dict[str, object]:
    return inspect_expired_lease_evidence(**e)  # type: ignore[arg-type]


def test_consistent_observations_still_hold_with_open_pr_and_workflow() -> None:
    result = _inspect(_evidence())
    assert result["schema"] == "github_expired_lease_evidence_hold_v1"
    assert result["decision"] == "HOLD"
    assert result["supplied_evidence_consistent"] is True
    assert result["observed_open_pr_numbers"] == [17]
    assert result["issue_number_unverified"] == 1067
    assert result["issue_node_id_unverified"] == "I_1067"
    assert result["unfinished_workflow_branch_observed"] is True
    assert result["sampled_run_ids"] == [RUN]
    assert result["sampled_artifact_ids"] == [21]
    assert result["trusted_roster_complete"] is False
    assert result["authoritative_registry_established"] is False
    assert result["lease_reassignment_authorized"] is False
    assert result["cleanup_authorized"] is False
    assert (
        "complete_historical_artifact_inventory_and_bytes"
        in result["unresolved_requirements"]
    )


def test_empty_supplied_run_and_artifact_rosters_are_not_proof_of_absence() -> None:
    e = _evidence()
    e["attempts"] = []
    e["artifacts"] = []
    e["inventory"]["unfinished_workflow_branches"] = []  # type: ignore[index]
    result = _inspect(e)
    assert result["sampled_run_ids"] == []
    assert result["sampled_artifact_ids"] == []
    assert result["decision"] == "HOLD"
    assert result["trusted_roster_complete"] is False
    assert result["lease_reassignment_authorized"] is False


def test_lease_must_be_expired_and_branch_tip_exact() -> None:
    e = _evidence()
    e["lease"].expires_at = NOW + timedelta(minutes=1)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="not expired"):
        _inspect(e)
    e = _evidence()
    e["lease"].head_sha = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="no exact branch tip"):
        _inspect(e)


def test_main_advance_or_untrusted_inventory_fails_closed() -> None:
    for label in ("expected", "inventory"):
        e = _evidence()
        if label == "expected":
            e["expected_main_sha"] = "f" * 40
        else:
            e["inventory"]["main_sha"] = "f" * 40  # type: ignore[index]
        with pytest.raises(ValueError, match="main"):
            _inspect(e)
    e = _evidence()
    inv = e["inventory"]
    inv["unfinished_workflow_branch_names_checked"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="inventory"):
        _inspect(e)


@pytest.mark.parametrize(
    "field",
    (
        "task_projection_trusted",
        "task_registry_complete",
        "workflow_activity_checked",
        "cleanup_authorized",
    ),
)
def test_inventory_cannot_claim_authority(field: str) -> None:
    e = _evidence()
    e["inventory"][field] = True  # type: ignore[index]
    with pytest.raises(ValueError, match=field):
        _inspect(e)


def test_registry_digest_and_task_lease_fences_are_required() -> None:
    e = _evidence()
    e["snapshot"].registry_digest = "0" * 64  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="registry snapshot"):
        _inspect(e)
    e = _evidence()
    entry = e["snapshot"].registry.entries[0]  # type: ignore[attr-defined]
    entry.status.lease_epoch = 2
    with pytest.raises(ValueError, match="identities conflict"):
        _inspect(e)
    e = _evidence()
    e["snapshot"].registry.entries.clear()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="missing or ambiguous"):
        _inspect(e)


def test_deleted_or_drifted_lease_branch_and_main_refuse() -> None:
    e = _evidence()
    e["inventory"]["branches"].pop()  # type: ignore[index]
    with pytest.raises(ValueError, match="lease branch"):
        _inspect(e)
    e = _evidence()
    e["inventory"]["branches"][1]["sha"] = "f" * 40  # type: ignore[index]
    with pytest.raises(ValueError, match="lease branch"):
        _inspect(e)
    e = _evidence()
    e["inventory"]["branches"][0]["sha"] = "f" * 40  # type: ignore[index]
    with pytest.raises(ValueError, match="main branch"):
        _inspect(e)


def test_workflow_ref_missing_unknown_or_duplicated_refuses() -> None:
    e = _evidence()
    e["inventory"]["unfinished_workflow_branches"] = ["missing"]  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown branches"):
        _inspect(e)
    e = _evidence()
    inventory = e["inventory"]
    inventory["unfinished_workflow_branches"] = [BRANCH, BRANCH]  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicated"):
        _inspect(e)


def test_open_pr_head_movement_or_duplicate_pr_fails_closed() -> None:
    e = _evidence()
    e["inventory"]["pull_requests"][0]["head_sha"] = "f" * 40  # type: ignore[index]
    with pytest.raises(ValueError, match="PR head"):
        _inspect(e)
    e = _evidence()
    rows = e["inventory"]["pull_requests"]  # type: ignore[index]
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate numbers"):
        _inspect(e)


def test_pairing_requires_run_attempt_and_artifact_for_each_run() -> None:
    e = _evidence()
    e["artifacts"] = []
    with pytest.raises(ValueError, match="no matched artifact"):
        _inspect(e)
    e = _evidence()
    e["attempts"] = []
    with pytest.raises(ValueError, match="no matched attempt"):
        _inspect(e)


def test_run_id_duplicates_and_changed_attempt_refuse() -> None:
    e = _evidence()
    e["attempts"].append(deepcopy(e["attempts"][0]))  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate workflow attempt"):
        _inspect(e)
    e = _evidence()
    e["attempts"][0]["head_sha"] = "f" * 40  # type: ignore[index]
    with pytest.raises(ValueError, match="differs from lease"):
        _inspect(e)
    e = _evidence()
    e["artifacts"][0]["run_attempt_observed"] = 3  # type: ignore[index]
    with pytest.raises(ValueError, match="differs from run attempt"):
        _inspect(e)


def test_attempt_job_count_missing_or_duplicate_ids_refuses() -> None:
    e = _evidence()
    e["attempts"][0]["job_count"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="absent or incomplete"):
        _inspect(e)
    e = _evidence()
    jobs = e["attempts"][0]["jobs"]  # type: ignore[index]
    jobs.append(deepcopy(jobs[0]))
    e["attempts"][0]["job_count"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicated"):
        _inspect(e)


@pytest.mark.parametrize(
    "field",
    (
        "artifact_attempt_binding_verified",
        "artifact_bytes_verified",
        "artifact_history_complete",
        "workflow_activity_checked",
        "task_registry_complete",
        "lease_reassignment_authorized",
        "cleanup_authorized",
    ),
)
def test_artifact_reports_cannot_promote_unverified_authority(field: str) -> None:
    e = _evidence()
    e["artifacts"][0][field] = True  # type: ignore[index]
    with pytest.raises(ValueError, match=field):
        _inspect(e)


def test_artifact_count_and_duplicate_id_across_runs_refuse() -> None:
    e = _evidence()
    e["artifacts"][0]["artifact_count"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="count differs"):
        _inspect(e)
    e = _evidence()
    # Same artifact id must not be claimed by different run-scoped observations.
    second_run = deepcopy(e["attempts"][0])  # type: ignore[index]
    second_run["run_id"] = RUN + 1
    e["attempts"].append(second_run)  # type: ignore[index]
    second_artifact = deepcopy(e["artifacts"][0])  # type: ignore[index]
    second_artifact["run_id"] = RUN + 1
    e["artifacts"].append(second_artifact)  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate artifact id"):
        _inspect(e)


def test_wrong_type_and_missing_sources_are_not_successful_absence() -> None:
    e = _evidence()
    e["inventory"]["branches"] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="branch roster"):
        _inspect(e)
    e = _evidence()
    e["attempts"] = None
    with pytest.raises(ValueError, match="attempt observations"):
        _inspect(e)
    e = _evidence()
    e["artifacts"] = None
    with pytest.raises(ValueError, match="artifact observations"):
        _inspect(e)


def test_artifact_run_attempt_requires_native_integer_and_repository_identity() -> None:
    e = _evidence()
    e["artifacts"][0]["run_attempt_observed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="artifact run attempt"):
        _inspect(e)
    e = _evidence()
    second = deepcopy(e["attempts"][0])  # type: ignore[index]
    second["run_id"] = RUN + 1
    e["attempts"].append(second)  # type: ignore[index]
    art = deepcopy(e["artifacts"][0])  # type: ignore[index]
    art["run_id"] = RUN + 1
    art["repository_id"] = 999
    art["artifacts"] = [{"id": 22}]
    e["artifacts"].append(art)  # type: ignore[index]
    with pytest.raises(ValueError, match="disagree on repository"):
        _inspect(e)


def test_invalid_run_state_and_unsettled_conclusion_fail_closed() -> None:
    e = _evidence()
    e["attempts"][0]["status"] = "unknown"  # type: ignore[index]
    with pytest.raises(ValueError, match="attempt status"):
        _inspect(e)
    e = _evidence()
    e["attempts"][0]["status"] = "in_progress"  # type: ignore[index]
    with pytest.raises(ValueError, match="unfinished attempt"):
        _inspect(e)
    e = _evidence()
    e["attempts"][0]["conclusion"] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="completed attempt"):
        _inspect(e)


def test_zero_oid_registry_ref_and_forged_readback_refuse() -> None:
    e = _evidence()
    e["snapshot"].ref_oid = "0" * 40  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="ref_oid"):
        _inspect(e)
    e = _evidence()
    e["snapshot"].ref_readback_verified = False  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="registry snapshot"):
        _inspect(e)


def test_unhashable_run_state_and_conclusion_fail_with_value_error() -> None:
    e = _evidence()
    e["attempts"][0]["status"] = {}  # type: ignore[index]
    with pytest.raises(ValueError, match="status"):
        _inspect(e)
    e = _evidence()
    e["attempts"][0]["conclusion"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="conclusion"):
        _inspect(e)


@pytest.mark.parametrize(
    "bad",
    [
        "refs/heads/main",
        "refs/heads/coordination/../main",
        "refs/heads/coordination/a.lock",
        "refs/heads/coordination/a//b",
    ],
)
def test_malformed_registry_ref_fails_closed(bad: str) -> None:
    e = _evidence()
    e["snapshot"].ref_name = bad  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="safe coordination"):
        _inspect(e)


def test_blank_repository_node_identity_fails_closed() -> None:
    e = _evidence()
    e["snapshot"].repository_node_id = " \t "  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="registry snapshot"):
        _inspect(e)
