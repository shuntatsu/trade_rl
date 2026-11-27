"""Reject foreign identity/serialization hooks at canonical registry boundaries."""

from dataclasses import replace
from datetime import timedelta

import pytest

from tests.agent_repo.test_coordination_enrollment import NOW, _snapshot
from tests.agent_repo.test_coordination_registry_transition import (
    REPOSITORY,
    _projection,
)
from tools.agent_repo.coordination.enrollment import (
    EnrollmentProjection,
    prepare_enrollment_projection,
    require_enrollment_readback,
    validate_enrollment_transition,
)
from tools.agent_repo.coordination.github_state import (
    TaskStatusRecord,
    render_status_comment,
)
from tools.agent_repo.coordination.registry import (
    CoordinatorRegistry,
    parse_registry_bytes,
    plan_registry_transition,
)


class _ForgedIdentity(str):
    def __ne__(self, other: object) -> bool:
        return False


class _TrapIdentity(str):
    def __ne__(self, other: object) -> bool:
        raise AssertionError("foreign inequality must not be invoked")

    def strip(self, chars: str | None = None) -> str:
        raise AssertionError("foreign strip must not be invoked")


class _ForeignProjection(EnrollmentProjection):
    def canonical_payload(self) -> dict[str, object]:
        payload = super().canonical_payload()
        payload["repository"] = "other/repository"
        return payload


class _ForeignStatus(TaskStatusRecord):
    def canonical_payload(self) -> dict[str, object]:
        raise AssertionError("foreign status serializer must not be invoked")


class _ForeignRegistry(CoordinatorRegistry):
    def digest(self) -> str:
        raise AssertionError("foreign digest must not be invoked")


@pytest.mark.parametrize("identity", [_ForgedIdentity, _TrapIdentity])
def test_projection_rejects_foreign_repository(identity: type[str]) -> None:
    with pytest.raises(ValueError, match="repository"):
        _projection("T1067-01", repository=identity("other/repository"))


def test_projection_rejects_foreign_issue_node_before_strip() -> None:
    with pytest.raises(ValueError, match="issue reference"):
        replace(_projection("T1067-01"), issue_node_id=_TrapIdentity("I_123"))


def test_registry_rejects_foreign_repository_even_when_empty() -> None:
    with pytest.raises(ValueError, match="repository"):
        CoordinatorRegistry(_ForgedIdentity(REPOSITORY))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("task_id", _ForgedIdentity("T1067-01"), "task_id"),
        ("task_contract_digest", _ForgedIdentity("0" * 64), "task_contract_digest"),
        ("base_sha", _ForgedIdentity("a" * 40), "base_sha"),
        ("head_sha", _ForgedIdentity("b" * 40), "head_sha"),
        ("owner", _TrapIdentity("agent-a"), "owner"),
        (
            "lease_branch",
            _ForgedIdentity("agent/T1067-01/e9999-other"),
            "fenced branch",
        ),
    ],
)
def test_status_rejects_foreign_identity(field: str, value: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_projection("T1067-01").status, **{field: value})


def _foreign_projection() -> EnrollmentProjection:
    normal = _projection("T1067-01")
    return _ForeignProjection(
        normal.repository, normal.issue_number, normal.issue_node_id, normal.status
    )


def test_registry_rejects_foreign_entry_serializer() -> None:
    with pytest.raises(ValueError, match="EnrollmentProjection"):
        CoordinatorRegistry(REPOSITORY, (_foreign_projection(),))


def test_projection_rejects_foreign_status_serializer() -> None:
    normal = _projection("T1067-01")
    foreign = _ForeignStatus(**vars(normal.status))
    with pytest.raises(ValueError, match="TaskStatusRecord"):
        replace(normal, status=foreign)


def test_transition_rejects_foreign_registry_before_digest() -> None:
    with pytest.raises(ValueError, match="CoordinatorRegistry"):
        plan_registry_transition(
            _ForeignRegistry(REPOSITORY),
            _projection("T1067-01"),
            expected_before_digest="0" * 64,
        )


def test_transition_rejects_foreign_candidate() -> None:
    registry = CoordinatorRegistry(REPOSITORY)
    with pytest.raises(ValueError, match="EnrollmentProjection"):
        plan_registry_transition(
            registry, _foreign_projection(), expected_before_digest=registry.digest()
        )


@pytest.mark.parametrize("foreign_previous", [False, True])
def test_direct_transition_rejects_foreign_projection(foreign_previous: bool) -> None:
    native = _projection("T1067-01")
    foreign = _foreign_projection()
    previous, candidate = (foreign, native) if foreign_previous else (None, foreign)
    roster = ("T1067-01",) if foreign_previous else ()
    with pytest.raises(ValueError, match="EnrollmentProjection"):
        validate_enrollment_transition(previous, candidate, enumerated_task_ids=roster)


def test_readback_rejects_foreign_projection_even_if_same_object() -> None:
    foreign = _foreign_projection()
    with pytest.raises(ValueError, match="EnrollmentProjection"):
        require_enrollment_readback(
            foreign,
            foreign,
            expected_registry_oid="c" * 40,
            observed_registry_oid="c" * 40,
            observed_branch_sha="b" * 40,
        )


def test_native_registry_roundtrip_and_plan_preserve_bytes() -> None:
    registry = CoordinatorRegistry(REPOSITORY)
    plan = plan_registry_transition(
        registry,
        _projection("T1067-01"),
        expected_before_digest=registry.digest(),
    )
    parsed = parse_registry_bytes(plan.after_registry.canonical_bytes())
    assert parsed == plan.after_registry
    assert parsed.digest() == plan.after_digest
    assert not plan.authoritative_registry_established
    assert not plan.task_registry_complete
    assert not plan.cleanup_authorized


class _AttributeTrapStatus(TaskStatusRecord):
    def __getattribute__(self, name: str) -> object:
        if name == "phase":
            raise AssertionError("foreign phase must not be accessed")
        return super().__getattribute__(name)


def test_preflight_rejects_foreign_status_before_access() -> None:
    packet, lease, _ = _snapshot()
    status = object.__new__(_AttributeTrapStatus)
    with pytest.raises(ValueError, match="TaskStatusRecord"):
        prepare_enrollment_projection(
            packet,
            lease,
            status,
            repository=REPOSITORY,
            issue_number=1067,
            issue_node_id="I_123",
            current_main_sha=packet.base_sha,
            observed_branch_sha="b" * 40,
            issue_is_open=True,
            now=NOW + timedelta(minutes=2),
        )


def test_status_renderer_rejects_foreign_record() -> None:
    normal = _projection("T1067-01").status
    with pytest.raises(ValueError, match="TaskStatusRecord"):
        render_status_comment(_ForeignStatus(**vars(normal)))


@pytest.mark.parametrize(
    "field",
    ["phase", "condition"],
)
def test_status_rejects_foreign_enum_text_before_lookup(field: str) -> None:
    normal = _projection("T1067-01").status
    foreign = _ForgedIdentity(getattr(normal, field).value)
    with pytest.raises(ValueError, match=field):
        replace(normal, **{field: foreign})
