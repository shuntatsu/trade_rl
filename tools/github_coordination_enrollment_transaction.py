"""Source-bound, disposable-only Task enrollment CAS verification.

The source Task audit and registry CAS share a single proposed transition.
All outputs remain TEST-ONLY. No production writer, complete roster or branch
cleanup authority is created, even when these bounded reads succeed.
"""

from __future__ import annotations

from datetime import datetime

from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import LeaseRecord
from tools.agent_repo.coordination.model import TaskPacket
from tools.github_coordination_enrollment_audit import (
    _commit_ref,
    _read_state,
    audit_task_enrollment,
)
from tools.github_coordination_ref_cas import (
    GraphQLApi,
    RegistryObjectReader,
    _read_snapshot,
    _validated_test_only_update,
    compare_and_swap_registry_plan,
)
from tools.github_coordination_registry_reader import GitObjectReadApi
from tools.github_coordination_registry_snapshot import observe_registry_snapshot


def execute_source_bound_disposable_enrollment(
    graphql: GraphQLApi,
    rest: GitObjectReadApi,
    objects: RegistryObjectReader,
    *,
    packet: TaskPacket,
    lease: LeaseRecord,
    status: TaskStatusRecord,
    repository: str,
    ref_name: str,
    registry_path: str,
    expected_issue_node_id: str,
    expected_oid: str,
    after_oid: str,
    expected_coordinator_login: str,
    now: datetime,
    allow_unprotected_test_mutation: bool = False,
) -> dict[str, object]:
    """Attempt one test-only CAS bound to a live source-derived Task proposal.

    This is not atomic across REST reads. A source change after updateRefs may
    leave an already-mutated disposable ref; exceptions must not trigger an
    automatic retry, compensation or success acknowledgment.
    """

    # Reject real repositories, non-coordination refs and missing opt-in BEFORE
    # any external read, including the otherwise read-only enrollment audit.
    _validated_test_only_update(
        repository=repository,
        ref_name=ref_name,
        expected_oid=expected_oid,
        after_oid=after_oid,
        expected_coordinator_login=expected_coordinator_login,
        allow_unprotected_test_mutation=allow_unprotected_test_mutation,
    )
    # The first GET must not interpolate fields from an unvalidated caller
    # object. The audit separately verifies all contract/digest/lease values.
    if (
        type(packet) is not TaskPacket
        or type(lease) is not LeaseRecord
        or type(status) is not TaskStatusRecord
        or type(packet.parent_issue) is not int
        or packet.parent_issue < 1
    ):
        raise ValueError("Task, lease and status require native Task contracts")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if (
        not isinstance(expected_issue_node_id, str)
        or not expected_issue_node_id.strip()
    ):
        raise ValueError("expected Issue node identity is required")

    source = _read_state(
        rest,
        repository=repository,
        issue_number=packet.parent_issue,
        lease_branch=lease.lease_branch,
    )
    # Pin the same immutable repository node and authenticated CAS actor
    # across the two separate readers; identical names alone are insufficient.
    actor_ref = _read_snapshot(graphql, repository=repository, ref_name=ref_name)
    if (
        actor_ref.repository_id != source.repository[1]
        or actor_ref.viewer_login.casefold() != expected_coordinator_login.casefold()
        or actor_ref.oid != expected_oid
    ):
        raise ValueError("GraphQL coordinator or repository identity differs from REST")
    audit = audit_task_enrollment(
        rest,
        packet=packet,
        lease=lease,
        status=status,
        repository=repository,
        registry_ref=ref_name,
        registry_path=registry_path,
        expected_issue_node_id=expected_issue_node_id,
        now=now,
    )
    if audit.registry_oid != expected_oid:
        raise ValueError("audited registry OID differs from expected CAS version")
    if source.main_sha != audit.main_sha or source.branch_sha != audit.branch_sha:
        raise ValueError("GitHub source changed during enrollment audit")

    # These checks intentionally occur after the expensive immutable-object
    # audit and immediately before the atomic ref CAS.
    if (
        _read_state(
            rest,
            repository=repository,
            issue_number=packet.parent_issue,
            lease_branch=lease.lease_branch,
        )
        != source
        or _commit_ref(rest, ref_name) != expected_oid
    ):
        raise ValueError("GitHub source changed before enrollment CAS")

    receipt = compare_and_swap_registry_plan(
        graphql,
        objects,
        plan=audit.plan,
        repository=repository,
        ref_name=ref_name,
        expected_oid=expected_oid,
        after_oid=after_oid,
        expected_coordinator_login=expected_coordinator_login,
        allow_unprotected_test_mutation=allow_unprotected_test_mutation,
    )

    # Check the *independent REST Git-object source*, not merely the object
    # reader that was supplied to GraphQL CAS. A failed check cannot undo CAS.
    if (
        _read_state(
            rest,
            repository=repository,
            issue_number=packet.parent_issue,
            lease_branch=lease.lease_branch,
        )
        != source
    ):
        raise ValueError("GitHub source changed after enrollment CAS")
    after_snapshot = observe_registry_snapshot(
        rest,
        expected_repository=repository,
        ref_name=ref_name,
        registry_path=registry_path,
    )
    if (
        after_snapshot.ref_oid != after_oid
        or after_snapshot.registry != audit.plan.after_registry
        or after_snapshot.registry_digest != audit.plan.after_digest
    ):
        raise ValueError("GitHub registry does not match the enrollment CAS plan")
    if (
        _read_state(
            rest,
            repository=repository,
            issue_number=packet.parent_issue,
            lease_branch=lease.lease_branch,
        )
        != source
        or _commit_ref(rest, ref_name) != after_oid
    ):
        raise ValueError("GitHub source changed during final enrollment readback")

    return {
        **receipt,
        "schema": "source_bound_disposable_enrollment_cas_observation_v1",
        "issue_node_id": audit.projection.issue_node_id,
        "task_id": audit.plan.task_id,
        "task_contract_digest": audit.projection.status.task_contract_digest,
        "source_readback_verified": True,
        "registry_snapshot_verified": True,
        "enrollment_persisted": False,
        "lease_reassignment_authorized": False,
    }


__all__ = ["execute_source_bound_disposable_enrollment"]
