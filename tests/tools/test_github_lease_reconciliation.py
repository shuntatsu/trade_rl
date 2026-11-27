"""Negative and hold-only oracles for Issue #1067 cross-evidence preflight."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

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
        "open_pr_count": 1,
        "branches": [
            {"name": "main", "sha": BASE, "reason": "default"},
            {"name": BRANCH, "sha": HEAD, "reason": "open_pr_ref"},
            {
                "name": "coordination/task-registry",
                "sha": REF,
                "reason": "unclassified",
            },
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
        "jobs": [
            {"id": 110, "name": "core", "status": "completed", "conclusion": "failure"}
        ],
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
    e["inventory"]["branches"].pop(1)  # type: ignore[index]
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


def test_registry_ref_must_exist_in_inventory_with_exact_oid() -> None:
    e = _evidence()
    e["inventory"]["branches"][-1]["sha"] = "f" * 40  # type: ignore[index]
    with pytest.raises(ValueError, match="registry ref"):
        _inspect(e)

    e = _evidence()
    e["inventory"]["branches"].pop()  # type: ignore[index]
    with pytest.raises(ValueError, match="registry ref"):
        _inspect(e)


@pytest.mark.parametrize("count", [None, -1, True, 2, "1"])
def test_open_pr_count_must_be_native_and_equal_to_roster(count: object) -> None:
    e = _evidence()
    if count is None:
        e["inventory"].pop("open_pr_count")  # type: ignore[attr-defined]
    else:
        e["inventory"]["open_pr_count"] = count  # type: ignore[index]
    with pytest.raises(ValueError, match="open_pr_count"):
        _inspect(e)


@pytest.mark.parametrize(
    "state", ["requested", "waiting", "pending", "queued", "in_progress"]
)
def test_unfinished_supplied_attempt_requires_workflow_branch_in_inventory(
    state: str,
) -> None:
    e = _evidence()
    e["inventory"]["unfinished_workflow_branches"] = []  # type: ignore[index]
    e["attempts"][0]["status"] = state  # type: ignore[index]
    e["attempts"][0]["conclusion"] = None  # type: ignore[index]
    e["artifacts"][0]["status"] = state  # type: ignore[index]
    e["artifacts"][0]["conclusion"] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="unfinished workflow"):
        _inspect(e)


class _StrSubclass(str):
    """Reject a non-native string even though it supports str operations."""


@pytest.mark.parametrize(
    "invalid",
    [False, 7, 1.0, None, {}, [], _StrSubclass("branch"), ""],
)
def test_unfinished_workflow_refs_require_native_nonempty_strings(
    invalid: object,
) -> None:
    e = _evidence()
    e["inventory"]["unfinished_workflow_branches"] = [invalid]  # type: ignore[index]
    with pytest.raises(ValueError, match="malformed, duplicated or unordered"):
        _inspect(e)


def test_unfinished_workflow_refs_reject_unsorted_values() -> None:
    e = _evidence()
    values = ["zzz", "aaa"]
    e["inventory"]["unfinished_workflow_branches"] = values  # type: ignore[index]
    with pytest.raises(ValueError, match="malformed, duplicated or unordered"):
        _inspect(e)


@pytest.mark.parametrize(
    "state", ["requested", "waiting", "pending", "queued", "in_progress"]
)
@pytest.mark.parametrize("missing", ["attempts", "artifacts", "both", "neither"])
def test_unfinished_observations_require_explicit_conclusion(
    state: str, missing: str
) -> None:
    e = _evidence()
    for source in ("attempts", "artifacts"):
        row = e[source][0]  # type: ignore[index]
        row["status"] = state
        row["conclusion"] = None
        if missing in (source, "both"):
            del row["conclusion"]
    if missing == "neither":
        result = _inspect(e)
        assert result["decision"] == "HOLD"
        assert result["supplied_evidence_consistent"] is True
        assert result["lease_reassignment_authorized"] is False
        assert result["cleanup_authorized"] is False
    else:
        with pytest.raises(ValueError, match="conclusion is missing"):
            _inspect(e)


@pytest.mark.parametrize(
    ("state", "conclusion", "missing", "pattern"),
    [
        ("completed", "success", True, "job conclusion is missing"),
        ("unknown", "success", False, "job status"),
        ("completed", None, False, "completed job conclusion"),
        ("queued", "success", False, "unfinished job"),
        ("completed", [], False, "job conclusion"),
        ("completed", "unknown", False, "job conclusion"),
        ([], None, False, "job status"),
    ],
)
def test_supplied_job_state_refuses_invalid_observation(
    state: object, conclusion: object, missing: bool, pattern: str
) -> None:
    e = _evidence()
    attempt = e["attempts"][0]  # type: ignore[index]
    artifact = e["artifacts"][0]  # type: ignore[index]
    attempt.update(status="in_progress", conclusion=None)
    artifact.update(status="in_progress", conclusion=None)
    job = attempt["jobs"][0]
    job.update(status=state, conclusion=conclusion)
    if missing:
        del job["conclusion"]
    with pytest.raises(ValueError, match=pattern):
        _inspect(e)


@pytest.mark.parametrize(
    ("run_status", "run_conclusion", "job_status", "job_conclusion", "pattern"),
    [
        (
            "completed",
            "failure",
            "in_progress",
            None,
            "completed run contains unfinished",
        ),
        (
            "completed",
            "success",
            "completed",
            "failure",
            "successful run contains failed",
        ),
        (
            "completed",
            "success",
            "completed",
            "cancelled",
            "successful run contains failed",
        ),
    ],
)
def test_supplied_run_and_job_contradictions_refuse(
    run_status: str,
    run_conclusion: str,
    job_status: str,
    job_conclusion: str | None,
    pattern: str,
) -> None:
    e = _evidence()
    attempt = e["attempts"][0]  # type: ignore[index]
    artifact = e["artifacts"][0]  # type: ignore[index]
    attempt.update(status=run_status, conclusion=run_conclusion)
    artifact.update(status=run_status, conclusion=run_conclusion)
    attempt["jobs"][0].update(status=job_status, conclusion=job_conclusion)
    with pytest.raises(ValueError, match=pattern):
        _inspect(e)


@pytest.mark.parametrize("bad_name", [None, "", " ", 42])
def test_supplied_job_name_is_required(bad_name: object) -> None:
    e = _evidence()
    job = e["attempts"][0]["jobs"][0]  # type: ignore[index]
    job.update(name=bad_name, conclusion="failure")
    with pytest.raises(ValueError, match="job name"):
        _inspect(e)


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        ("completed", "success"),
        ("completed", "skipped"),
        ("completed", "neutral"),
        ("in_progress", None),
        ("queued", None),
    ],
)
def test_valid_supplied_job_status_remains_hold(
    status: str, conclusion: str | None
) -> None:
    e = _evidence()
    attempt = e["attempts"][0]  # type: ignore[index]
    artifact = e["artifacts"][0]  # type: ignore[index]
    attempt.update(
        status=status, conclusion="success" if status == "completed" else None
    )
    artifact.update(status=attempt["status"], conclusion=attempt["conclusion"])
    attempt["jobs"][0].update(status=status, conclusion=conclusion)
    result = _inspect(e)
    assert result["supplied_evidence_consistent"] is True
    assert result["decision"] == "HOLD"
    assert result["lease_reassignment_authorized"] is False
    assert result["cleanup_authorized"] is False


@pytest.mark.parametrize(
    "suffix",
    [".hidden/registry", "group.lock/registry", "registry.", "group/.hidden"],
)
def test_invalid_registry_ref_refuses_even_if_inventory_matches(suffix: str) -> None:
    e = _evidence()
    branch = f"coordination/{suffix}"
    e["snapshot"].ref_name = f"refs/heads/{branch}"  # type: ignore[attr-defined]
    e["inventory"]["branches"][-1]["name"] = branch  # type: ignore[index]
    e["attempts"][0]["jobs"][0]["conclusion"] = "failure"  # type: ignore[index]
    with pytest.raises(ValueError, match="safe coordination"):
        _inspect(e)


@pytest.mark.parametrize(
    "suffix", ["group./registry", "nested/group/registry", "group.LOCK/registry"]
)
def test_valid_registry_ref_is_still_hold(suffix: str) -> None:
    e = _evidence()
    branch = f"coordination/{suffix}"
    e["snapshot"].ref_name = f"refs/heads/{branch}"  # type: ignore[attr-defined]
    e["inventory"]["branches"][-1]["name"] = branch  # type: ignore[index]
    e["attempts"][0]["jobs"][0]["conclusion"] = "failure"  # type: ignore[index]
    result = _inspect(e)
    assert result["decision"] == "HOLD"
    assert result["cleanup_authorized"] is False


def test_expiry_preflight_distinguishes_repeated_wall_clock_hour() -> None:
    zone = ZoneInfo("America/New_York")
    e = _evidence()
    e["lease"].expires_at = datetime(2026, 11, 1, 1, 5, tzinfo=zone, fold=1)  # type: ignore[attr-defined]
    e["now"] = datetime(2026, 11, 1, 1, 50, tzinfo=zone, fold=0)
    with pytest.raises(ValueError, match="not expired"):
        _inspect(e)
    e["now"] = datetime(2026, 11, 1, 1, 5, tzinfo=zone, fold=1)
    assert _inspect(e)["decision"] == "HOLD"


@pytest.mark.parametrize(
    "location",
    ["expected", "inventory", "branch", "pr", "attempt", "artifact"],
)
def test_foreign_sha_never_reaches_comparison(location: str) -> None:
    calls: list[str] = []

    class ForeignSHA(str):
        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            return True

        def __ne__(self, other: object) -> bool:
            calls.append("ne")
            return False

    e = _evidence()
    value = ForeignSHA("f" * 40)
    if location == "expected":
        e["expected_main_sha"] = value
    elif location == "inventory":
        e["inventory"]["main_sha"] = value  # type: ignore[index]
    elif location == "branch":
        e["inventory"]["branches"][1]["sha"] = value  # type: ignore[index]
    elif location == "pr":
        e["inventory"]["pull_requests"][0]["head_sha"] = value  # type: ignore[index]
    elif location == "attempt":
        e["attempts"][0]["head_sha"] = value  # type: ignore[index]
    else:
        e["artifacts"][0]["head_sha"] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="exact Git SHA"):
        _inspect(e)
    assert calls == []


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("inventory", "schema"),
        ("inventory", "repository"),
        ("inventory", "task_scope"),
        ("attempts", "schema"),
        ("attempts", "scope"),
        ("attempts", "job_source"),
        ("attempts", "head_branch"),
        ("artifacts", "schema"),
        ("artifacts", "scope"),
        ("artifacts", "repository"),
        ("artifacts", "head_branch"),
    ],
)
@pytest.mark.parametrize("same_text", [False, True])
def test_reconciliation_rejects_foreign_identity_text(
    section: str, field: str, same_text: bool
) -> None:
    calls: list[str] = []

    class ForeignText(str):
        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            return True

        def __ne__(self, other: object) -> bool:
            calls.append("ne")
            return False

    e = _evidence()
    row = e[section] if section == "inventory" else e[section][0]  # type: ignore[index]
    row[field] = ForeignText(row[field] if same_text else "foreign")  # type: ignore[index]
    with pytest.raises(ValueError):
        _inspect(e)
    assert calls == []


@pytest.mark.parametrize(
    ("path", "label"),
    [
        (("inventory",), "work inventory"),
        (("inventory", "branches", 0), "branch"),
        (("inventory", "pull_requests", 0), "open PR"),
        (("attempts", 0), "attempt observation"),
        (("attempts", 0, "jobs", 0), "job"),
        (("artifacts", 0), "artifact observation"),
        (("artifacts", 0, "artifacts", 0), "artifact"),
    ],
)
@pytest.mark.parametrize("kind", ["dict", "key"])
def test_foreign_reconciliation_objects_rejected_before_hooks(
    path: tuple[str | int, ...], label: str, kind: str
) -> None:
    calls: list[str] = []

    class ForeignDict(dict):
        def get(self, *args):
            calls.append("get")
            return super().get(*args)

        def __getitem__(self, key):
            calls.append("getitem")
            return super().__getitem__(key)

        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

        def __contains__(self, key):
            calls.append("contains")
            return super().__contains__(key)

        def keys(self):
            calls.append("keys")
            return super().keys()

        def items(self):
            calls.append("items")
            return super().items()

        def values(self):
            calls.append("values")
            return super().values()

        def __len__(self):
            calls.append("len")
            return super().__len__()

        def __bool__(self):
            calls.append("bool")
            return super().__len__() != 0

    class ForeignKey(str):
        def __hash__(self):
            calls.append("hash")
            return super().__hash__()

        def __eq__(self, other):
            calls.append("eq")
            return super().__eq__(other)

    e = _evidence()
    parent = e
    for part in path[:-1]:
        parent = parent[part]  # type: ignore[index]
    original = parent[path[-1]]  # type: ignore[index]
    if kind == "dict":
        replacement = ForeignDict(original)
    else:
        replacement = dict(original)
        key = next(iter(replacement))
        replacement[ForeignKey(key)] = replacement.pop(key)
    parent[path[-1]] = replacement  # type: ignore[index]
    calls.clear()
    with pytest.raises(ValueError, match=label + " must be a native"):
        _inspect(e)
    assert calls == []


@pytest.mark.parametrize(
    ("path", "label"),
    [
        (("inventory", "branches"), "branch roster"),
        (("inventory", "pull_requests"), "open PR roster"),
        (("inventory", "unfinished_workflow_branches"), "unfinished workflow branches"),
        (("attempts",), "attempt observations"),
        (("artifacts",), "artifact observations"),
        (("attempts", 0, "jobs"), "run attempt jobs"),
        (("artifacts", 0, "artifacts"), "run artifacts"),
    ],
)
@pytest.mark.parametrize("kind", ["list", "tuple"])
def test_foreign_reconciliation_sequences_rejected_before_hooks(
    path: tuple[str | int, ...], label: str, kind: str
) -> None:
    calls: list[str] = []

    class ForeignList(list):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

        def __len__(self):
            calls.append("len")
            return super().__len__()

        def __getitem__(self, key):
            calls.append("getitem")
            return super().__getitem__(key)

        def __bool__(self):
            calls.append("bool")
            return super().__len__() != 0

    class ForeignTuple(tuple):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

        def __len__(self):
            calls.append("len")
            return super().__len__()

        def __getitem__(self, key):
            calls.append("getitem")
            return super().__getitem__(key)

        def __bool__(self):
            calls.append("bool")
            return super().__len__() != 0

    e = _evidence()
    parent = e
    for part in path[:-1]:
        parent = parent[part]  # type: ignore[index]
    original = parent[path[-1]]  # type: ignore[index]
    parent[path[-1]] = (  # type: ignore[index]
        ForeignList(original) if kind == "list" else ForeignTuple(original)
    )
    calls.clear()
    with pytest.raises(ValueError, match=label + " must be a native"):
        _inspect(e)
    assert calls == []


@pytest.mark.parametrize(
    "path",
    [
        ("inventory", "branches"),
        ("inventory", "pull_requests"),
        ("inventory", "unfinished_workflow_branches"),
        ("attempts",),
        ("artifacts",),
        ("attempts", 0, "jobs"),
        ("artifacts", 0, "artifacts"),
    ],
)
def test_native_tuple_reconciliation_rosters_remain_supported(
    path: tuple[str | int, ...],
) -> None:
    e = _evidence()
    expected = _inspect(e)
    parent = e
    for part in path[:-1]:
        parent = parent[part]  # type: ignore[index]
    parent[path[-1]] = tuple(parent[path[-1]])  # type: ignore[index, arg-type]
    assert _inspect(e) == expected


def test_foreign_workflow_branch_class_hook_never_runs() -> None:
    calls: list[str] = []

    class ForeignBranch:
        @property
        def __class__(self):
            calls.append("__class__")
            raise RuntimeError("foreign class hook executed")

    e = _evidence()
    e["inventory"]["unfinished_workflow_branches"] = [ForeignBranch()]  # type: ignore[index]
    with pytest.raises(ValueError, match="unfinished workflow branches"):
        _inspect(e)
    assert calls == []
