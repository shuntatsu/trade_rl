from __future__ import annotations

import pytest

from tools.agent_repo.coordination.enrollment import EnrollmentProjection
from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import lease_branch_name
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase
from tools.agent_repo.coordination.registry import (
    CoordinatorRegistry,
    RegistryTransitionPlan,
    plan_registry_transition,
)

REPOSITORY = "shuntatsu/trade_rl"
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
DIGEST = "0" * 64


def _projection(
    task_id: str,
    *,
    phase: TaskPhase = TaskPhase.EXECUTING,
    repository: str = REPOSITORY,
    issue_number: int = 1067,
    task_revision: int = 1,
    task_digest: str = DIGEST,
    head_sha: str = HEAD_SHA,
    owner: str = "agent-a",
    lease_epoch: int = 1,
) -> EnrollmentProjection:
    return EnrollmentProjection(
        repository=repository,
        issue_number=issue_number,
        issue_node_id=f"I_{task_id}",
        status=TaskStatusRecord(
            task_id=task_id,
            task_revision=task_revision,
            task_contract_digest=task_digest,
            phase=phase,
            condition=TaskCondition.HEALTHY,
            base_sha=BASE_SHA,
            owner=owner,
            lease_epoch=lease_epoch,
            lease_branch=lease_branch_name(task_id, lease_epoch, owner),
            head_sha=head_sha,
        ),
    )


def _plan(
    registry: CoordinatorRegistry,
    candidate: EnrollmentProjection,
    *,
    expected_before_digest: str | None = None,
) -> RegistryTransitionPlan:
    return plan_registry_transition(
        registry,
        candidate,
        expected_before_digest=(
            registry.digest()
            if expected_before_digest is None
            else expected_before_digest
        ),
    )


def test_plan_creates_new_task_and_binds_exact_payload_digests() -> None:
    registry = CoordinatorRegistry(REPOSITORY)
    candidate = _projection("T1067-01")

    plan = _plan(registry, candidate)

    assert plan.operation == "create"
    assert plan.task_id == "T1067-01"
    assert plan.before_digest == registry.digest()
    assert plan.after_registry.entries == (candidate,)
    assert plan.after_digest == plan.after_registry.digest()
    assert plan.after_digest != plan.before_digest
    assert plan.authoritative_registry_established is False
    assert plan.task_registry_complete is False
    assert plan.cleanup_authorized is False


def test_plan_updates_one_task_and_preserves_other_registry_entries() -> None:
    target = _projection("T1067-01", phase=TaskPhase.EXECUTING)
    other = _projection("T1067-02", issue_number=1068)
    registry = CoordinatorRegistry(REPOSITORY, (other, target))
    candidate = _projection("T1067-01", phase=TaskPhase.REVIEW)

    plan = _plan(registry, candidate)

    assert plan.operation == "update"
    assert [entry.status.task_id for entry in plan.after_registry.entries] == [
        "T1067-01",
        "T1067-02",
    ]
    assert plan.after_registry.entries[0] == candidate
    assert plan.after_registry.entries[1] == other
    assert plan.after_digest != plan.before_digest


def test_plan_rejects_stale_digest_noop_and_cross_repository_candidate() -> None:
    candidate = _projection("T1067-01")
    registry = CoordinatorRegistry(REPOSITORY, (candidate,))

    with pytest.raises(ValueError, match="expected registry digest"):
        _plan(
            registry,
            _projection("T1067-01", phase=TaskPhase.REVIEW),
            expected_before_digest="f" * 64,
        )

    with pytest.raises(ValueError, match="no-op"):
        _plan(registry, candidate)

    with pytest.raises(ValueError, match="repository"):
        _plan(
            CoordinatorRegistry(REPOSITORY),
            _projection("T1067-03", repository="other/repository"),
        )


@pytest.mark.parametrize(
    "candidate,match",
    [
        (
            _projection("T1067-01", phase=TaskPhase.INTEGRATION),
            "transition is not allowed",
        ),
        (
            _projection("T1067-01", task_revision=2, task_digest="1" * 64),
            "revision bump requires",
        ),
        (
            _projection("T1067-01", head_sha="c" * 40),
            "branch advance requires",
        ),
        (
            _projection("T1067-01", owner="agent-b"),
            "owner or fenced branch changed",
        ),
    ],
)
def test_plan_reuses_existing_fail_closed_enrollment_transition_rules(
    candidate: EnrollmentProjection,
    match: str,
) -> None:
    registry = CoordinatorRegistry(REPOSITORY, (_projection("T1067-01"),))

    with pytest.raises(ValueError, match=match):
        _plan(registry, candidate)
