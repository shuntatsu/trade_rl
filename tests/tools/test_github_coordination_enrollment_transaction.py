"""Source-bound disposable-only enrollment transaction tests.

These are fake transports; never claim a live GitHub single-writer proof.
"""

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
from tools.github_coordination_enrollment_transaction import (
    execute_source_bound_disposable_enrollment,
)

REPOSITORY = "shuntatsu/trade-rl-coordination-disposable"
REGISTRY_REF = "refs/heads/coordination/task-registry"
PATH = "coordination/registry.json"
MAIN = "a" * 40
BRANCH = "b" * 40
BEFORE = "c" * 40
AFTER = "d" * 40
BEFORE_TREE = "e" * 40
AFTER_TREE = "f" * 40
NOW = datetime(2026, 11, 23, tzinfo=timezone.utc)
ISSUE = 1067
TASK = "T1067-P0"


def packet_and_lease() -> tuple[TaskPacket, LeaseRecord, TaskStatusRecord]:
    packet = TaskPacket(
        task_id=TASK,
        task_revision=1,
        parent_issue=ISSUE,
        title="enrollment",
        objective="proof-bound disposable CAS",
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
        lease_epoch=lease.epoch,
        lease_branch=lease.lease_branch,
        head_sha=BRANCH,
    )
    return packet, lease, status


def blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


class GraphQL:
    def __init__(self) -> None:
        self.oid = BEFORE
        self.viewer = "coordinator-bot"
        self.repository_id = "R_disposable"
        self.calls: list[str] = []
        self.mutations = 0
        self.interrupt = False

    def execute(self, document: str, variables: dict[str, object]) -> dict[str, object]:
        if "mutation UpdateCoordinationRef" in document:
            self.calls.append("mutation")
            value = variables["input"]
            assert isinstance(value, dict)
            updates = value["refUpdates"]
            assert isinstance(updates, list) and len(updates) == 1
            update = updates[0]
            assert isinstance(update, dict)
            if self.oid != update["beforeOid"]:
                raise ValueError("stale beforeOid")
            self.oid = update["afterOid"]  # type: ignore[assignment]
            self.mutations += 1
            if self.interrupt:
                raise RuntimeError("interrupted after remote mutation")
            return {
                "data": {"updateRefs": {"clientMutationId": value["clientMutationId"]}}
            }
        self.calls.append("read")
        return {
            "data": {
                "viewer": {"login": self.viewer},
                "repository": {
                    "id": self.repository_id,
                    "ref": {"target": {"oid": self.oid}},
                },
            }
        }


class SourceRest:
    def __init__(self, graphql: GraphQL, after: CoordinatorRegistry) -> None:
        self.graphql = graphql
        self.before = CoordinatorRegistry(REPOSITORY)
        self.after = after
        self.calls: list[tuple[str, str]] = []
        self.counts: Counter[str] = Counter()
        self.drift: dict[str, tuple[int, dict[str, Any]]] = {}
        self.corrupt_after = False

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        self.calls.append((method, suffix))
        self.counts[suffix] += 1
        if method != "GET":
            raise AssertionError("Source transport is GET-only")
        if suffix == "":
            data: dict[str, Any] = {
                "id": 123,
                "node_id": "R_disposable",
                "full_name": REPOSITORY,
                "default_branch": "main",
            }
        elif suffix == f"issues/{ISSUE}":
            data = {
                "id": 456,
                "number": ISSUE,
                "node_id": "I_issue",
                "state": "open",
                "repository_url": f"https://api.github.com/repos/{REPOSITORY}",
                "updated_at": "2026-11-23T00:00:00Z",
            }
        elif suffix.startswith("git/ref/"):
            name = "refs/" + suffix.removeprefix("git/ref/")
            oid = {
                "refs/heads/main": MAIN,
                "refs/heads/" + packet_and_lease()[1].lease_branch: BRANCH,
                REGISTRY_REF: self.graphql.oid,
            }[name]
            data = {"ref": name, "object": {"type": "commit", "sha": oid}}
        else:
            oid = self.graphql.oid
            registry = self.after if oid == AFTER else self.before
            raw = registry.canonical_bytes()
            if self.corrupt_after and oid == AFTER:
                raw += b"\n"
            tree_oid = AFTER_TREE if oid == AFTER else BEFORE_TREE
            if suffix == f"git/commits/{oid}":
                data = {"sha": oid, "tree": {"sha": tree_oid}}
            elif suffix == f"git/trees/{tree_oid}":
                assert query == {"recursive": "1"}
                data = {
                    "sha": tree_oid,
                    "truncated": False,
                    "tree": [
                        {
                            "path": PATH,
                            "type": "blob",
                            "mode": "100644",
                            "sha": blob_oid(raw),
                            "size": len(raw),
                        }
                    ],
                }
            elif suffix == f"git/blobs/{blob_oid(raw)}":
                data = {
                    "sha": blob_oid(raw),
                    "size": len(raw),
                    "encoding": "base64",
                    "content": base64.b64encode(raw).decode(),
                }
            else:
                raise AssertionError(f"Unexpected read: {suffix}")
        if suffix in self.drift:
            nth, change = self.drift[suffix]
            if self.counts[suffix] >= nth:
                data.update(change)
        return data


class Objects:
    def __init__(self, after: CoordinatorRegistry) -> None:
        self.before = CoordinatorRegistry(REPOSITORY)
        self.after = after
        self.calls: list[str] = []
        self.corrupt = False

    def read_registry_bytes(self, repository: str, commit_oid: str) -> bytes:
        assert repository == REPOSITORY
        self.calls.append(commit_oid)
        if commit_oid == BEFORE:
            return self.before.canonical_bytes()
        if commit_oid == AFTER:
            return self.after.canonical_bytes() + (b"\n" if self.corrupt else b"")
        raise AssertionError("unexpected object read")


def fixture() -> tuple[GraphQL, SourceRest, Objects]:
    _, _, status = packet_and_lease()
    after = CoordinatorRegistry(
        REPOSITORY, (EnrollmentProjection(REPOSITORY, ISSUE, "I_issue", status),)
    )
    graphql = GraphQL()
    return graphql, SourceRest(graphql, after), Objects(after)


def run(
    graphql: GraphQL, rest: SourceRest, objects: Objects, **overrides: Any
) -> dict[str, object]:
    packet, lease, status = packet_and_lease()
    kwargs: dict[str, Any] = dict(
        packet=packet,
        lease=lease,
        status=status,
        repository=REPOSITORY,
        ref_name=REGISTRY_REF,
        registry_path=PATH,
        expected_issue_node_id="I_issue",
        expected_oid=BEFORE,
        after_oid=AFTER,
        expected_coordinator_login="coordinator-bot",
        now=NOW,
        allow_unprotected_test_mutation=True,
    )
    kwargs.update(overrides)
    return execute_source_bound_disposable_enrollment(graphql, rest, objects, **kwargs)


def test_source_bound_disposable_cas_has_independent_full_readback() -> None:
    graphql, rest, objects = fixture()
    result = run(graphql, rest, objects)
    assert graphql.mutations == 1 and graphql.oid == AFTER
    assert result["source_readback_verified"] is True
    assert result["registry_snapshot_verified"] is True
    assert result["payload_readback_verified"] is True
    for name in (
        "enrollment_persisted",
        "authoritative_registry_established",
        "task_registry_complete",
        "lease_reassignment_authorized",
        "cleanup_authorized",
    ):
        assert result[name] is False
    assert all(method == "GET" for method, _ in rest.calls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repository": "shuntatsu/trade_rl"},
        {"allow_unprotected_test_mutation": False},
        {"ref_name": "refs/heads/main"},
        {"expected_oid": AFTER},
    ],
)
def test_unsafe_target_or_missing_optin_never_touches_any_transport(
    kwargs: dict[str, Any],
) -> None:
    graphql, rest, objects = fixture()
    with pytest.raises(ValueError):
        run(graphql, rest, objects, **kwargs)
    assert not rest.calls and not objects.calls and not graphql.calls


def test_after_payload_must_match_exact_issue_and_task_before_cas() -> None:
    graphql, rest, objects = fixture()
    objects.after = CoordinatorRegistry(REPOSITORY)
    with pytest.raises(ValueError):
        run(graphql, rest, objects)
    assert graphql.mutations == 0


def test_concurrent_stale_remote_oid_never_mutates() -> None:
    graphql, rest, objects = fixture()
    graphql.oid = "9" * 40
    with pytest.raises((ValueError, AssertionError)):
        run(graphql, rest, objects)
    assert graphql.mutations == 0


def test_modified_issue_during_preflight_prevents_mutation() -> None:
    graphql, rest, objects = fixture()
    rest.drift[f"issues/{ISSUE}"] = (2, {"updated_at": "drifted"})
    with pytest.raises(ValueError, match="changed"):
        run(graphql, rest, objects)
    assert graphql.mutations == 0


def test_modified_source_after_preflight_before_cas_prevents_mutation() -> None:
    graphql, rest, objects = fixture()
    rest.drift[f"issues/{ISSUE}"] = (4, {"updated_at": "drifted"})
    with pytest.raises(ValueError, match="changed"):
        run(graphql, rest, objects)
    assert graphql.mutations == 0


def test_interrupted_mutation_is_not_retried_or_acknowledged() -> None:
    graphql, rest, objects = fixture()
    graphql.interrupt = True
    with pytest.raises(RuntimeError, match="interrupted"):
        run(graphql, rest, objects)
    assert graphql.mutations == 1 and graphql.calls.count("mutation") == 1


def test_post_cas_source_edit_fails_closed_without_claiming_rollback() -> None:
    graphql, rest, objects = fixture()
    rest.drift[f"issues/{ISSUE}"] = (6, {"updated_at": "drifted"})
    with pytest.raises(ValueError, match="changed"):
        run(graphql, rest, objects)
    assert graphql.mutations == 1 and graphql.oid == AFTER


def test_post_cas_registry_bytes_must_match_independent_rest_observation() -> None:
    graphql, rest, objects = fixture()
    rest.corrupt_after = True
    with pytest.raises(ValueError):
        run(graphql, rest, objects)
    assert graphql.mutations == 1


def test_unauthorized_viewer_refused_by_graphql_cas() -> None:
    graphql, rest, objects = fixture()
    graphql.viewer = "worker-a"
    with pytest.raises(ValueError, match="GraphQL coordinator"):
        run(graphql, rest, objects)
    assert graphql.mutations == 0


def test_invalid_native_inputs_are_rejected_before_source_get() -> None:
    graphql, rest, objects = fixture()
    with pytest.raises(ValueError, match="native Task contracts"):
        run(graphql, rest, objects, packet=None)
    assert not rest.calls and not objects.calls and not graphql.calls


def test_naive_clock_does_not_authorize_mutation() -> None:
    graphql, rest, objects = fixture()
    with pytest.raises(ValueError, match="timezone-aware"):
        run(graphql, rest, objects, now=NOW.replace(tzinfo=None))
    assert graphql.mutations == 0


def test_replay_of_wrong_contract_digest_cannot_mutate() -> None:
    graphql, rest, objects = fixture()
    _, _, status = packet_and_lease()
    with pytest.raises(ValueError):
        run(
            graphql,
            rest,
            objects,
            status=replace(status, task_contract_digest="f" * 64),
        )
    assert graphql.mutations == 0


def test_cross_task_entry_remains_identical_across_enrollment_cas() -> None:
    graphql, rest, objects = fixture()
    _, _, status = packet_and_lease()
    extra_status = replace(
        status,
        task_id="T1067-OTHER",
        task_contract_digest="1" * 64,
        lease_branch=lease_branch_name("T1067-OTHER", 1, "worker-a"),
    )
    other = EnrollmentProjection(REPOSITORY, 1068, "I_other", extra_status)
    before = CoordinatorRegistry(REPOSITORY, (other,))
    after = CoordinatorRegistry(
        REPOSITORY,
        (
            other,
            EnrollmentProjection(REPOSITORY, ISSUE, "I_issue", status),
        ),
    )
    rest.before, rest.after = before, after
    objects.before, objects.after = before, after

    receipt = run(graphql, rest, objects)
    assert receipt["source_readback_verified"] is True
    assert other in rest.after.entries
    assert graphql.mutations == 1


def test_forged_extra_task_in_after_registry_is_rejected_before_mutation() -> None:
    graphql, rest, objects = fixture()
    _, _, status = packet_and_lease()
    extra_status = replace(
        status,
        task_id="T1067-FORGED",
        lease_branch=lease_branch_name("T1067-FORGED", 1, "worker-a"),
    )
    objects.after = CoordinatorRegistry(
        REPOSITORY,
        (
            *objects.after.entries,
            EnrollmentProjection(REPOSITORY, 1069, "I_forged", extra_status),
        ),
    )
    with pytest.raises(ValueError):
        run(graphql, rest, objects)
    assert graphql.mutations == 0


def test_main_branch_moved_during_pre_cas_source_readback_fails_closed() -> None:
    graphql, rest, objects = fixture()
    rest.drift["git/ref/heads/main"] = (
        4,
        {"object": {"type": "commit", "sha": "9" * 40}},
    )
    with pytest.raises(ValueError, match="changed"):
        run(graphql, rest, objects)
    assert graphql.mutations == 0


def test_post_cas_actor_mismatch_cannot_acknowledge_registration() -> None:
    class ActorSwitch(GraphQL):
        def execute(
            self, document: str, variables: dict[str, object]
        ) -> dict[str, object]:
            result = super().execute(document, variables)
            if "mutation UpdateCoordinationRef" in document:
                self.viewer = "worker-a"
            return result

    graphql, rest, objects = fixture()
    switched = ActorSwitch()
    rest.graphql = switched
    with pytest.raises(ValueError, match="authenticated Coordinator changed"):
        run(switched, rest, objects)
    assert switched.mutations == 1


def test_graphql_rest_repository_identity_mismatch_is_rejected_without_mutation() -> (
    None
):
    graphql, rest, objects = fixture()
    graphql.repository_id = "R_wrong"
    with pytest.raises(ValueError, match="repository identity"):
        run(graphql, rest, objects)
    assert graphql.mutations == 0
    assert not objects.calls


def test_graphql_actor_cannot_self_declare_worker_as_coordinator() -> None:
    graphql, rest, objects = fixture()
    graphql.viewer = "worker-a"
    with pytest.raises(ValueError, match="coordinator"):
        run(graphql, rest, objects)
    assert graphql.mutations == 0
    assert not objects.calls
