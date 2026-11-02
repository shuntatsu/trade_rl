"""Test-only HTTPS transport and guarded CAS driver with negative oracles.

The fake GraphQL implementation simulates competing requests only; these tests
are NOT evidence of actual GitHub CAS linearizability, remote authentication,
branch protection, or interruption recovery in a disposable repository.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

import pytest

from tools.agent_repo.coordination.enrollment import EnrollmentProjection
from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import lease_branch_name
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase
from tools.agent_repo.coordination.registry import CoordinatorRegistry
from tools.github_coordination_disposable_e2e import (
    execute_disposable_registry_cas,
    inspect_disposable_interrupted_cas,
    prepare_disposable_registry_plan,
)
from tools.github_coordination_ref_cas import ZERO_OID

REPOSITORY = "shuntatsu/trade-rl-coordination-disposable"
PRODUCTION = "shuntatsu/trade_rl"
REF = "refs/heads/coordination/task-registry"
A = "a" * 40
B = "b" * 40
C = "c" * 40
COORDINATOR = "coordinator-bot"


def _entry(phase: TaskPhase, *, checkpoint: str | None = None) -> EnrollmentProjection:
    task_id = "T1067-E2E"
    return EnrollmentProjection(
        repository=REPOSITORY,
        issue_number=1067,
        issue_node_id="I_1067",
        status=TaskStatusRecord(
            task_id=task_id,
            task_revision=1,
            task_contract_digest="d" * 64,
            phase=phase,
            condition=TaskCondition.HEALTHY,
            base_sha="e" * 40,
            owner=COORDINATOR,
            lease_epoch=1,
            lease_branch=lease_branch_name(task_id, 1, COORDINATOR),
            head_sha="f" * 40,
            checkpoint=checkpoint,
        ),
    )


class FakeRegistryObjects:
    def __init__(self, before: CoordinatorRegistry, after: CoordinatorRegistry) -> None:
        self.objects = {A: before.canonical_bytes(), B: after.canonical_bytes()}
        self.calls: list[tuple[str, str]] = []

    def read_registry_bytes(self, repository: str, commit_oid: str) -> bytes:
        self.calls.append((repository, commit_oid))
        assert repository == REPOSITORY
        return self.objects[commit_oid]


class ConcurrentGraphQL:
    def __init__(self, *, oid: str | None = A, viewer: str = COORDINATOR) -> None:
        self.oid = oid
        self.viewer = viewer
        self.repository_id = "R_disposable"
        self.mutation_count = 0
        self.calls: list[str] = []
        self.fail_after_commit = False
        self.lock = threading.Lock()

    def execute(self, document: str, variables: dict[str, object]) -> dict[str, object]:
        with self.lock:
            if "mutation UpdateCoordinationRef" in document:
                self.calls.append("mutation")
                value = variables["input"]
                assert isinstance(value, dict)
                updates = value["refUpdates"]
                assert isinstance(updates, list) and len(updates) == 1
                update = updates[0]
                assert isinstance(update, dict)
                assert update["force"] is False
                if update["beforeOid"] != (self.oid or ZERO_OID):
                    raise RuntimeError("beforeOid CAS conflict")
                self.mutation_count += 1
                self.oid = update["afterOid"]  # type: ignore[assignment]
                if self.fail_after_commit:
                    raise RuntimeError("network interrupted after remote mutation")
                return {
                    "data": {
                        "updateRefs": {"clientMutationId": value["clientMutationId"]}
                    }
                }
            self.calls.append("read")
            return {
                "data": {
                    "viewer": {"login": self.viewer},
                    "repository": {
                        "id": self.repository_id,
                        "ref": (
                            None if self.oid is None else {"target": {"oid": self.oid}}
                        ),
                    },
                }
            }


def _fixture(*, initial: bool = False) -> tuple[ConcurrentGraphQL, FakeRegistryObjects]:
    before = CoordinatorRegistry(
        REPOSITORY, () if initial else (_entry(TaskPhase.EXECUTING),)
    )
    after = CoordinatorRegistry(
        REPOSITORY, (_entry(TaskPhase.EXECUTING if initial else TaskPhase.REVIEW),)
    )
    return ConcurrentGraphQL(oid=None if initial else A), FakeRegistryObjects(
        before, after
    )


def _run(
    api: ConcurrentGraphQL,
    objects: FakeRegistryObjects,
    **kwargs: object,
) -> dict[str, object]:
    args: dict[str, Any] = {
        "repository": REPOSITORY,
        "ref_name": REF,
        "expected_oid": A,
        "after_oid": B,
        "expected_coordinator_login": COORDINATOR,
        "allow_unprotected_test_mutation": True,
    }
    args.update(kwargs)
    return execute_disposable_registry_cas(api, objects, **args)


def test_test_only_real_model_plan_is_single_task_and_payload_bound() -> None:
    api, objects = _fixture()
    plan = prepare_disposable_registry_plan(
        objects, repository=REPOSITORY, expected_oid=A, after_oid=B
    )
    assert plan.operation == "update"
    assert plan.task_id == "T1067-E2E"
    result = _run(api, objects)
    assert api.oid == B
    assert api.calls == ["read", "mutation", "read"]
    assert api.mutation_count == 1
    assert result["readback_verified"] is True
    assert result["payload_readback_verified"] is True
    assert result["test_only_unprotected_mutation"] is True
    for flag in (
        "authoritative_registry_established",
        "task_registry_complete",
        "cleanup_authorized",
    ):
        assert result[flag] is False
    # Caller plan, executor preflight, and CAS validation independently
    # re-read both commits before the final post-mutation payload check.
    assert objects.calls == [
        (REPOSITORY, A),
        (REPOSITORY, B),
        (REPOSITORY, A),
        (REPOSITORY, B),
        (REPOSITORY, A),
        (REPOSITORY, B),
        (REPOSITORY, B),
    ]


def test_initial_disposable_enrollment_only_uses_absent_ref_precondition() -> None:
    api, objects = _fixture(initial=True)
    result = _run(api, objects, expected_oid=ZERO_OID)
    assert api.mutation_count == 1
    assert api.oid == B
    assert result["before_oid"] == ZERO_OID
    assert result["authoritative_registry_established"] is False
    assert all(oid == B for _, oid in objects.calls)


@pytest.mark.parametrize(
    ("repository", "message"),
    [
        (PRODUCTION, "production repository"),
        ("SHUNTATSU/TRADE_RL", "production repository"),
        ("shuntatsu/not-a-sandbox", "explicitly disposable"),
        ("not-a-repository", "repository"),
    ],
)
def test_unsafe_repository_fails_before_any_io(repository: str, message: str) -> None:
    api, objects = _fixture()
    with pytest.raises(ValueError, match=message):
        _run(api, objects, repository=repository)
    assert not api.calls and not objects.calls


def test_flag_is_required_even_for_disposable_repository() -> None:
    api, objects = _fixture()
    with pytest.raises(ValueError, match="test-only mutation is disabled"):
        _run(api, objects, allow_unprotected_test_mutation=False)
    assert not api.calls and not objects.calls


@pytest.mark.parametrize("bad", ["refs/heads/main", "refs/tags/coordination/x"])
def test_invalid_namespace_is_rejected_before_io(bad: str) -> None:
    api, objects = _fixture()
    with pytest.raises(ValueError, match="coordination branch namespace"):
        _run(api, objects, ref_name=bad)
    assert not api.calls and not objects.calls


def test_tampered_or_incomplete_after_registry_fails_before_mutation() -> None:
    api, objects = _fixture()
    objects.objects[B] = b"not valid JSON"
    with pytest.raises(ValueError):
        _run(api, objects)
    assert api.mutation_count == 0
    assert not api.calls

    api, objects = _fixture()
    objects.objects[B] = objects.objects[A]
    with pytest.raises(ValueError, match="exactly one Task transition"):
        _run(api, objects)
    assert api.mutation_count == 0
    assert not api.calls


def test_two_task_mutation_or_deletion_cannot_pass_preflight() -> None:
    api, objects = _fixture()
    before = CoordinatorRegistry(REPOSITORY, (_entry(TaskPhase.EXECUTING),))
    added = replace(
        _entry(TaskPhase.REVIEW),
        issue_number=1068,
        issue_node_id="I_1068",
        status=replace(
            _entry(TaskPhase.REVIEW).status,
            task_id="T1067-OTHER",
            lease_branch=lease_branch_name("T1067-OTHER", 1, COORDINATOR),
        ),
    )
    after = CoordinatorRegistry(REPOSITORY, (_entry(TaskPhase.REVIEW), added))
    objects.objects[B] = after.canonical_bytes()
    with pytest.raises(ValueError, match="exactly one Task transition"):
        _run(api, objects)
    assert api.mutation_count == 0

    api, objects = _fixture()
    objects.objects[A] = before.canonical_bytes()
    objects.objects[B] = CoordinatorRegistry(REPOSITORY).canonical_bytes()
    with pytest.raises(ValueError, match="never remove"):
        _run(api, objects)
    assert api.mutation_count == 0


def test_stale_before_oid_blocks_competing_writer_without_retry() -> None:
    api, objects = _fixture()
    api.oid = C
    with pytest.raises(ValueError, match="expected OID"):
        _run(api, objects)
    assert api.mutation_count == 0
    assert api.calls == ["read"]


def test_unauthenticated_viewer_blocks_mutation() -> None:
    api, objects = _fixture()
    api.viewer = "another-user"
    with pytest.raises(ValueError, match="authenticated Coordinator"):
        _run(api, objects)
    assert api.mutation_count == 0


def test_parallel_competing_clients_only_one_succeeds_in_fake_store() -> None:
    api, objects = _fixture()
    alternate = CoordinatorRegistry(
        REPOSITORY, (_entry(TaskPhase.REVIEW, checkpoint="alternative"),)
    )
    objects.objects[C] = alternate.canonical_bytes()
    winners: list[str] = []
    failures: list[str] = []
    lock = threading.Lock()

    def writer(after_oid: str) -> None:
        try:
            receipt = _run(api, objects, after_oid=after_oid)
            assert receipt["cleanup_authorized"] is False
            with lock:
                winners.append(after_oid)
        except (ValueError, RuntimeError) as error:
            with lock:
                failures.append(type(error).__name__)

    threads = [threading.Thread(target=writer, args=(v,)) for v in (B, C)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(winners) == 1
    assert len(failures) == 1
    assert api.oid == winners[0]
    assert api.mutation_count == 1


def test_interruption_after_remote_mutation_never_reports_success_or_retries() -> None:
    api, objects = _fixture()
    api.fail_after_commit = True
    with pytest.raises(RuntimeError, match="interrupted after remote mutation"):
        _run(api, objects)
    assert api.oid == B
    assert api.mutation_count == 1
    assert api.calls == ["read", "mutation"]


def _inspect(
    api: ConcurrentGraphQL,
    objects: FakeRegistryObjects,
    **kwargs: object,
) -> dict[str, object]:
    args: dict[str, Any] = {
        "repository": REPOSITORY,
        "ref_name": REF,
        "expected_oid": A,
        "after_oid": B,
        "expected_coordinator_login": COORDINATOR,
    }
    args.update(kwargs)
    return inspect_disposable_interrupted_cas(api, objects, **args)


@pytest.mark.parametrize(
    ("oid", "expected_state"),
    [
        (A, "BEFORE_OBSERVED_NOT_PROOF_UNAPPLIED"),
        (B, "AFTER_OBSERVED_UNATTRIBUTED"),
        (C, "CONFLICTING_OID_OBSERVED"),
    ],
)
def test_interruption_inspection_is_nonmutating_and_never_authorizes_recovery(
    oid: str, expected_state: str
) -> None:
    api, objects = _fixture()
    api.oid = oid
    report = _inspect(api, objects)
    assert report["state"] == expected_state
    assert report["observed_oid"] == oid
    assert api.calls == ["read", "read"]
    assert api.mutation_count == 0
    assert report["mutation_attempted"] is False
    for key in (
        "writer_attribution_verified",
        "transaction_outcome_proven",
        "retry_authorized",
        "rollback_authorized",
        "authoritative_registry_established",
        "task_registry_complete",
        "cleanup_authorized",
    ):
        assert report[key] is False


def test_interruption_inspection_rejects_actor_ref_or_repository_drift() -> None:
    api, objects = _fixture()
    api.viewer = "other-user"
    with pytest.raises(ValueError, match="authenticated Coordinator"):
        _inspect(api, objects)
    assert api.calls == ["read"]

    class DriftingGraphQL(ConcurrentGraphQL):
        def execute(
            self, document: str, variables: dict[str, object]
        ) -> dict[str, object]:
            result = super().execute(document, variables)
            if len(self.calls) == 1:
                self.oid = C
            return result

    api = DriftingGraphQL()
    with pytest.raises(ValueError, match="changed during inspection"):
        _inspect(api, objects)
    assert api.mutation_count == 0


def test_interruption_inspection_refuses_non_disposable_target_before_io() -> None:
    api, objects = _fixture()
    with pytest.raises(ValueError, match="production repository"):
        _inspect(api, objects, repository=PRODUCTION)
    assert api.calls == [] and objects.calls == []
