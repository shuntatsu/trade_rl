from __future__ import annotations

from collections import Counter

import pytest

from tools.agent_repo.coordination.enrollment import EnrollmentProjection
from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import lease_branch_name
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase
from tools.agent_repo.coordination.registry import (
    CoordinatorRegistry,
    plan_registry_transition,
)
from tools.github_coordination_ref_cas import (
    ZERO_OID,
    compare_and_swap_registry_plan,
)

REPOSITORY = "shuntatsu/trade-rl-coordination-disposable"
REF = "refs/heads/coordination/task-registry"
A = "a" * 40
B = "b" * 40
C = "c" * 40


def _entry(phase: TaskPhase) -> EnrollmentProjection:
    return EnrollmentProjection(
        repository=REPOSITORY,
        issue_number=1067,
        issue_node_id="I_1067",
        status=TaskStatusRecord(
            task_id="T1067-01",
            task_revision=1,
            task_contract_digest="d" * 64,
            phase=phase,
            condition=TaskCondition.HEALTHY,
            base_sha="e" * 40,
            owner="coordinator-bot",
            lease_epoch=1,
            lease_branch=lease_branch_name("T1067-01", 1, "coordinator-bot"),
            head_sha="f" * 40,
        ),
    )


class FakeGraphQL:
    def __init__(self, oid: str | None = A, viewer: str = "coordinator-bot") -> None:
        self.oid = oid
        self.viewer = viewer
        self.calls: list[str] = []
        self.fail_mutation = False

    def execute(self, document: str, variables: dict[str, object]) -> dict[str, object]:
        if "mutation UpdateCoordinationRef" in document:
            self.calls.append("mutation")
            if self.fail_mutation:
                raise RuntimeError("mutation failed")
            value = variables["input"]
            assert isinstance(value, dict)
            updates = value["refUpdates"]
            assert isinstance(updates, list) and len(updates) == 1
            update = updates[0]
            assert isinstance(update, dict)
            assert update["beforeOid"] == (self.oid or ZERO_OID)
            self.oid = update["afterOid"]  # type: ignore[assignment]
            return {"data": {"updateRefs": {"clientMutationId": value["clientMutationId"]}}}
        self.calls.append("read")
        return {
            "data": {
                "viewer": {"login": self.viewer},
                "repository": {
                    "id": "R_test",
                    "ref": None if self.oid is None else {"target": {"oid": self.oid}},
                },
            }
        }


class FakeObjects:
    def __init__(self, bytes_by_oid: dict[str, bytes]) -> None:
        self.bytes_by_oid = bytes_by_oid
        self.calls: list[tuple[str, str]] = []
        self.drift_after_readback: bytes | None = None
        self.read_counts: Counter[str] = Counter()

    def read_registry_bytes(self, repository: str, commit_oid: str) -> bytes:
        self.calls.append((repository, commit_oid))
        self.read_counts[commit_oid] += 1
        if self.drift_after_readback is not None and self.read_counts[commit_oid] >= 2:
            return self.drift_after_readback
        return self.bytes_by_oid[commit_oid]


def _fixture(*, initial: bool = False) -> tuple[object, FakeGraphQL, FakeObjects]:
    before = CoordinatorRegistry(REPOSITORY, () if initial else (_entry(TaskPhase.EXECUTING),))
    after = _entry(TaskPhase.EXECUTING if initial else TaskPhase.REVIEW)
    plan = plan_registry_transition(before, after, expected_before_digest=before.digest())
    objects = FakeObjects(
        {B: plan.after_registry.canonical_bytes()}
        if initial
        else {A: before.canonical_bytes(), B: plan.after_registry.canonical_bytes()}
    )
    return plan, FakeGraphQL(None if initial else A), objects


def _cas(plan: object, api: FakeGraphQL, objects: FakeObjects, **kwargs: object) -> dict[str, object]:
    args: dict[str, object] = {
        "plan": plan,
        "repository": REPOSITORY,
        "ref_name": REF,
        "expected_oid": A,
        "after_oid": B,
        "expected_coordinator_login": "coordinator-bot",
        "allow_unprotected_test_mutation": True,
    }
    args.update(kwargs)
    return compare_and_swap_registry_plan(api, objects, **args)  # type: ignore[arg-type]


def test_payload_bound_test_only_cas_checks_before_after_and_independent_readback() -> None:
    plan, api, objects = _fixture()
    result = _cas(plan, api, objects)
    assert api.oid == B
    assert api.calls == ["read", "mutation", "read"]
    assert objects.calls == [(REPOSITORY, A), (REPOSITORY, B), (REPOSITORY, B)]
    assert result["before_registry_digest"] == plan.before_digest
    assert result["after_registry_digest"] == plan.after_digest
    assert result["payload_readback_verified"] is True
    assert result["authoritative_registry_established"] is False
    assert result["task_registry_complete"] is False
    assert result["cleanup_authorized"] is False


def test_zero_oid_initial_creation_requires_empty_registry() -> None:
    plan, api, objects = _fixture(initial=True)
    result = _cas(plan, api, objects, expected_oid=ZERO_OID)
    assert result["before_oid"] == ZERO_OID
    assert objects.calls == [(REPOSITORY, B), (REPOSITORY, B)]
    assert api.oid == B


@pytest.mark.parametrize("bad_before", [b"{}", b"\n", b"not-json"])
def test_tampered_before_registry_never_mutates(bad_before: bytes) -> None:
    plan, api, objects = _fixture()
    objects.bytes_by_oid[A] = bad_before
    with pytest.raises(ValueError):
        _cas(plan, api, objects)
    assert "mutation" not in api.calls
    assert api.oid == A


def test_wrong_after_registry_canonical_bytes_never_mutate() -> None:
    plan, api, objects = _fixture()
    objects.bytes_by_oid[B] = CoordinatorRegistry(REPOSITORY, ()).canonical_bytes()
    with pytest.raises(ValueError, match="after registry"):
        _cas(plan, api, objects)
    assert "mutation" not in api.calls
    assert api.oid == A


def test_noncanonical_after_registry_bytes_are_rejected() -> None:
    plan, api, objects = _fixture()
    objects.bytes_by_oid[B] += b"\n"
    with pytest.raises(ValueError):
        _cas(plan, api, objects)
    assert "mutation" not in api.calls


def test_missing_object_fails_before_mutation() -> None:
    plan, api, objects = _fixture()
    del objects.bytes_by_oid[B]
    with pytest.raises(KeyError):
        _cas(plan, api, objects)
    assert "mutation" not in api.calls


def test_post_cas_payload_readback_drift_fails_closed() -> None:
    plan, api, objects = _fixture()
    objects.drift_after_readback = CoordinatorRegistry(REPOSITORY, ()).canonical_bytes()
    with pytest.raises(ValueError, match="after registry"):
        _cas(plan, api, objects)
    assert api.oid == B  # Ref may already have changed: never claim rollback or success.


def test_stale_remote_oid_fails_even_if_payloads_match() -> None:
    plan, api, objects = _fixture()
    api.oid = C
    with pytest.raises(ValueError, match="expected OID"):
        _cas(plan, api, objects)
    assert "mutation" not in api.calls


def test_disposable_guard_and_optin_remain_mandatory() -> None:
    plan, api, objects = _fixture()
    with pytest.raises(ValueError, match="production repository"):
        _cas(plan, api, objects, repository="shuntatsu/trade_rl")
    with pytest.raises(ValueError, match="test-only mutation is disabled"):
        _cas(plan, api, objects, allow_unprotected_test_mutation=False)
    assert api.calls == []
    assert objects.calls == []


def test_mismatched_plan_repository_never_mutates() -> None:
    plan, api, objects = _fixture()
    with pytest.raises(ValueError, match="plan repository"):
        _cas(plan, api, objects, repository="shuntatsu/other-coordination-disposable")
    assert api.calls == []
    assert objects.calls == []
