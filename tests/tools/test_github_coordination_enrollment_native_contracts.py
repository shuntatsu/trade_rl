"""Reject overridden enrollment contracts before observing or mutating remote state."""

from dataclasses import replace

import pytest

from tests.tools import test_github_coordination_enrollment_audit as audit_tests
from tests.tools import (
    test_github_coordination_enrollment_transaction as transaction_tests,
)
from tools.agent_repo.coordination.enrollment import prepare_enrollment_projection
from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import LeaseRecord
from tools.agent_repo.coordination.model import TaskPacket


def _forged_packet(original: TaskPacket) -> TaskPacket:
    class ForgedPacket(TaskPacket):
        def contract_digest(self) -> str:
            return original.contract_digest()

    packet = ForgedPacket(**vars(replace(original, objective="different contract")))
    assert TaskPacket.contract_digest(packet) != original.contract_digest()
    return packet


class _ForeignLease(LeaseRecord):
    pass


class _ForeignStatus(TaskStatusRecord):
    pass


def test_projection_rejects_overridden_contract_digest() -> None:
    packet, lease, status = audit_tests.inputs()
    with pytest.raises(ValueError, match="TaskPacket"):
        prepare_enrollment_projection(
            _forged_packet(packet),
            lease,
            status,
            repository=audit_tests.REPO,
            issue_number=audit_tests.ISSUE,
            issue_node_id="I_issue",
            current_main_sha=audit_tests.MAIN,
            observed_branch_sha=audit_tests.BRANCH,
            issue_is_open=True,
            now=audit_tests.NOW,
        )


def test_projection_rejects_foreign_lease() -> None:
    packet, lease, status = audit_tests.inputs()
    with pytest.raises(ValueError, match="LeaseRecord"):
        prepare_enrollment_projection(
            packet,
            _ForeignLease(**vars(lease)),
            status,
            repository=audit_tests.REPO,
            issue_number=audit_tests.ISSUE,
            issue_node_id="I_issue",
            current_main_sha=audit_tests.MAIN,
            observed_branch_sha=audit_tests.BRANCH,
            issue_is_open=True,
            now=audit_tests.NOW,
        )


@pytest.mark.parametrize("field", ["packet", "lease", "status"])
def test_audit_rejects_foreign_contract_before_get(field: str) -> None:
    packet, lease, status = audit_tests.inputs()
    foreign = {
        "packet": _forged_packet(packet),
        "lease": _ForeignLease(**vars(lease)),
        "status": _ForeignStatus(**vars(status)),
    }
    api = audit_tests.FakeGitHub()
    with pytest.raises(ValueError, match="native contracts"):
        audit_tests.audit(api, **{field: foreign[field]})
    assert not api.calls


@pytest.mark.parametrize("field", ["packet", "lease", "status"])
def test_disposable_cas_rejects_foreign_contract_before_io(field: str) -> None:
    packet, lease, status = transaction_tests.packet_and_lease()
    foreign = {
        "packet": _forged_packet(packet),
        "lease": _ForeignLease(**vars(lease)),
        "status": _ForeignStatus(**vars(status)),
    }
    graphql, rest, objects = transaction_tests.fixture()
    with pytest.raises(ValueError, match="native Task contracts"):
        transaction_tests.run(graphql, rest, objects, **{field: foreign[field]})
    assert not graphql.calls
    assert not rest.calls
    assert graphql.mutations == 0
