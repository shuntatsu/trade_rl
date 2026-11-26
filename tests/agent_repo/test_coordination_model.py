from __future__ import annotations

from dataclasses import replace

import pytest

from tools.agent_repo.coordination.model import (
    Capability,
    DependencyKind,
    EvidenceKind,
    EvidenceRecord,
    EvidenceResult,
    ExecutionMode,
    RiskLevel,
    TaskCondition,
    TaskDependency,
    TaskPacket,
    TaskPhase,
    TaskStatus,
    WriteScope,
)
from tools.agent_repo.coordination.state import (
    evidence_is_current,
    evidence_is_current_after_peer_integration,
    validate_transition,
)

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
OTHER_HEAD_SHA = "c" * 40
MAIN_SHA = "d" * 40
OTHER_MAIN_SHA = "e" * 40


def _packet(
    *,
    revision: int = 1,
    objective: str = "Implement coordination state",
    title: str = "Coordination state",
    execution_mode: ExecutionMode = ExecutionMode.WRITE,
) -> TaskPacket:
    return TaskPacket(
        task_id="T500-01",
        task_revision=revision,
        parent_issue=500,
        title=title,
        objective=objective,
        execution_mode=execution_mode,
        non_goals=("no distributed lock service",),
        dependencies=(
            TaskDependency(
                task_id="T500-00",
                kind=DependencyKind.HARD,
                requires_phase=TaskPhase.COMPLETE,
            ),
        ),
        write_scope=(
            WriteScope(
                allow=("tools/agent_repo/**", "tests/agent_repo/**"),
                deny=("trade_rl/**",),
            )
            if execution_mode is ExecutionMode.WRITE
            else WriteScope()
        ),
        resource_keys=(
            "authority:agent-coordination-task-state",
            "workflow:agent-task-claim",
        ),
        capabilities=(
            Capability.REPOSITORY_READ,
            *(
                (Capability.LEASE_BRANCH_WRITE, Capability.DRAFT_PR_WRITE)
                if execution_mode is ExecutionMode.WRITE
                else ()
            ),
        ),
        acceptance_criteria=("duplicate claim is rejected",),
        test_oracle=("deterministic transition",),
        base_sha=BASE_SHA,
        risk=RiskLevel.MEDIUM,
        deliverable="pull_request",
    )


def _evidence(
    packet: TaskPacket,
    *,
    kind: EvidenceKind = EvidenceKind.REVIEW,
    head_sha: str = HEAD_SHA,
    main_sha: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_kind=kind,
        task_id=packet.task_id,
        task_revision=packet.task_revision,
        task_contract_digest=packet.contract_digest(),
        base_sha=packet.base_sha,
        head_sha=head_sha,
        identifier="evidence-1",
        result=EvidenceResult.PASS,
        main_sha=main_sha,
    )


def test_task_contract_digest_is_deterministic_and_semantic() -> None:
    left = _packet()
    right = _packet()

    assert left.canonical_payload() == right.canonical_payload()
    assert left.contract_digest() == right.contract_digest()
    assert len(left.contract_digest()) == 64

    changed = _packet(objective="Changed semantic objective")
    assert changed.contract_digest() != left.contract_digest()

    presentation_only = _packet(title="Different display title")
    assert presentation_only.contract_digest() == left.contract_digest()


def test_task_packet_rejects_invalid_identity_and_scope() -> None:
    with pytest.raises(ValueError, match="task_id"):
        TaskPacket(
            task_id="",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.READ_ONLY,
            base_sha=BASE_SHA,
        )

    with pytest.raises(ValueError, match="task_revision"):
        _packet(revision=0)

    with pytest.raises(ValueError, match="base_sha"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.READ_ONLY,
            base_sha="not-a-sha",
        )

    with pytest.raises(ValueError, match="read_only"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.READ_ONLY,
            write_scope=WriteScope(allow=("tools/**",)),
            base_sha=BASE_SHA,
        )

    with pytest.raises(ValueError, match="write.*allow"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.WRITE,
            base_sha=BASE_SHA,
        )


def test_task_packet_rejects_duplicate_contract_entries() -> None:
    with pytest.raises(ValueError, match="duplicate dependency"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.WRITE,
            dependencies=(
                TaskDependency("T500-00", DependencyKind.HARD),
                TaskDependency("T500-00", DependencyKind.EVIDENCE),
            ),
            write_scope=WriteScope(allow=("tools/**",)),
            base_sha=BASE_SHA,
        )

    with pytest.raises(ValueError, match="duplicate resource key"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.WRITE,
            write_scope=WriteScope(allow=("tools/**",)),
            resource_keys=("identity:x", "identity:x"),
            base_sha=BASE_SHA,
        )


def test_phase_transition_contract_is_explicit() -> None:
    for current, target in (
        (TaskPhase.PLANNED, TaskPhase.READY),
        (TaskPhase.READY, TaskPhase.EXECUTING),
        (TaskPhase.EXECUTING, TaskPhase.REVIEW),
        (TaskPhase.REVIEW, TaskPhase.VERIFICATION),
        (TaskPhase.VERIFICATION, TaskPhase.INTEGRATION),
        (TaskPhase.INTEGRATION, TaskPhase.COMPLETE),
        (TaskPhase.REVIEW, TaskPhase.EXECUTING),
        (TaskPhase.VERIFICATION, TaskPhase.EXECUTING),
        (TaskPhase.INTEGRATION, TaskPhase.VERIFICATION),
        (TaskPhase.INTEGRATION, TaskPhase.EXECUTING),
    ):
        validate_transition(current, target)

    with pytest.raises(ValueError, match="transition"):
        validate_transition(TaskPhase.READY, TaskPhase.COMPLETE)
    with pytest.raises(ValueError, match="terminal"):
        validate_transition(TaskPhase.COMPLETE, TaskPhase.EXECUTING)


def test_task_status_keeps_phase_and_condition_independent() -> None:
    status = TaskStatus(
        phase=TaskPhase.REVIEW,
        condition=TaskCondition.STALE,
        reason="head_changed",
    )
    assert status.phase is TaskPhase.REVIEW
    assert status.condition is TaskCondition.STALE
    assert status.reason == "head_changed"


def test_review_evidence_is_bound_to_exact_task_contract_and_head() -> None:
    packet = _packet()
    evidence = _evidence(packet)

    assert evidence_is_current(evidence, packet, head_sha=HEAD_SHA)
    assert not evidence_is_current(evidence, packet, head_sha=OTHER_HEAD_SHA)
    assert not evidence_is_current(evidence, _packet(revision=2), head_sha=HEAD_SHA)
    assert not evidence_is_current(
        evidence,
        _packet(objective="new objective"),
        head_sha=HEAD_SHA,
    )


def test_integration_evidence_is_also_bound_to_current_main() -> None:
    packet = _packet()
    evidence = _evidence(
        packet,
        kind=EvidenceKind.INTEGRATION,
        main_sha=MAIN_SHA,
    )

    assert evidence_is_current(
        evidence,
        packet,
        head_sha=HEAD_SHA,
        current_main_sha=MAIN_SHA,
    )
    assert not evidence_is_current(
        evidence,
        packet,
        head_sha=HEAD_SHA,
        current_main_sha=OTHER_MAIN_SHA,
    )
    assert not evidence_is_current(evidence, packet, head_sha=HEAD_SHA)


def test_evidence_rejects_malformed_sha_and_integration_without_main() -> None:
    packet = _packet()
    with pytest.raises(ValueError, match="head_sha"):
        EvidenceRecord(
            evidence_kind=EvidenceKind.REVIEW,
            task_id=packet.task_id,
            task_revision=packet.task_revision,
            task_contract_digest=packet.contract_digest(),
            base_sha=packet.base_sha,
            head_sha="bad",
            identifier="evidence-1",
            result=EvidenceResult.PASS,
        )

    with pytest.raises(ValueError, match="main_sha"):
        _evidence(packet, kind=EvidenceKind.INTEGRATION)


@pytest.mark.parametrize(
    ("field", "value"),
    [("task_revision", 1.0), ("parent_issue", 500.0)],
)
def test_task_packet_rejects_integer_shaped_floats(field: str, value: object) -> None:
    kwargs = {
        "task_id": "T500-01",
        "task_revision": 1,
        "parent_issue": 500,
        "title": "x",
        "objective": "x",
        "execution_mode": ExecutionMode.READ_ONLY,
        "base_sha": BASE_SHA,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="integer"):
        TaskPacket(**kwargs)


def test_task_packet_rejects_non_string_identity_and_sha() -> None:
    with pytest.raises(ValueError, match="task_id"):
        TaskPacket(
            task_id=123,
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.READ_ONLY,
            base_sha=BASE_SHA,
        )
    with pytest.raises(ValueError, match="base_sha"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.READ_ONLY,
            base_sha=123,
        )


def test_evidence_identity_rejects_integer_shaped_float_and_non_string_id() -> None:
    packet = _packet()
    with pytest.raises(ValueError, match="task_revision"):
        EvidenceRecord(
            evidence_kind=EvidenceKind.REVIEW,
            task_id=packet.task_id,
            task_revision=1.0,
            task_contract_digest=packet.contract_digest(),
            base_sha=packet.base_sha,
            head_sha=HEAD_SHA,
            identifier="evidence-1",
            result=EvidenceResult.PASS,
        )
    with pytest.raises(ValueError, match="identifier"):
        EvidenceRecord(
            evidence_kind=EvidenceKind.REVIEW,
            task_id=packet.task_id,
            task_revision=packet.task_revision,
            task_contract_digest=packet.contract_digest(),
            base_sha=packet.base_sha,
            head_sha=HEAD_SHA,
            identifier=123,
            result=EvidenceResult.PASS,
        )


def test_task_status_reason_requires_text() -> None:
    with pytest.raises(ValueError, match="status reason"):
        TaskStatus(phase=TaskPhase.REVIEW, reason=123)


class _TextLookalike:
    """Legacy str() coercion accepted this unrelated caller object."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def _invalid_text_collection(kind: str, *, resource: bool = False) -> object:
    if kind == "integer":
        return (123,)
    if kind == "lookalike":
        return (_TextLookalike("identity:spoof" if resource else "not-typed"),)
    if kind == "null":
        return (None,)
    if kind == "boolean":
        return (True,)
    if kind == "mixed":
        return (
            ("identity:valid" if resource else "valid"),
            _TextLookalike("identity:spoof" if resource else "invalid"),
        )
    if kind == "bare_string":
        return "abc"
    if kind == "mapping":
        return {"abc": 1}
    if kind == "unordered_set":
        return {"alpha", "beta"}
    if kind == "generator":
        return (name for name in ("alpha", "beta"))
    raise AssertionError(kind)


@pytest.mark.parametrize("field", ("allow", "deny"))
@pytest.mark.parametrize(
    "kind",
    (
        "integer",
        "lookalike",
        "null",
        "boolean",
        "mixed",
        "bare_string",
        "mapping",
        "unordered_set",
        "generator",
    ),
)
def test_write_scope_rejects_untyped_direct_text_collections(
    field: str, kind: str
) -> None:
    with pytest.raises(ValueError, match="ordered native string"):
        WriteScope(**{field: _invalid_text_collection(kind)})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ("non_goals", "resource_keys", "acceptance_criteria", "test_oracle")
)
@pytest.mark.parametrize(
    "kind",
    (
        "integer",
        "lookalike",
        "null",
        "boolean",
        "mixed",
        "bare_string",
        "mapping",
        "unordered_set",
        "generator",
    ),
)
def test_task_packet_rejects_untyped_direct_text_collections(
    field: str, kind: str
) -> None:
    original = _packet()
    kwargs = {name: getattr(original, name) for name in TaskPacket.__dataclass_fields__}
    kwargs[field] = _invalid_text_collection(kind, resource=field == "resource_keys")
    with pytest.raises(ValueError, match="ordered native string"):
        TaskPacket(**kwargs)  # type: ignore[arg-type]


def test_direct_text_lists_are_frozen_without_changing_canonical_identity() -> None:
    original = _packet()
    allow = ["tools/**"]
    scope = WriteScope(allow=allow)
    packet_lists = {
        name: list(getattr(original, name))
        for name in ("non_goals", "resource_keys", "acceptance_criteria", "test_oracle")
    }
    kwargs = {name: getattr(original, name) for name in TaskPacket.__dataclass_fields__}
    kwargs.update(packet_lists)
    packet = TaskPacket(**kwargs)
    assert packet.canonical_payload() == original.canonical_payload()
    assert packet.contract_digest() == original.contract_digest()
    assert scope.allow == ("tools/**",)
    allow.append("later")
    for source in packet_lists.values():
        source.append("later")
    assert scope.allow == ("tools/**",)
    assert packet.contract_digest() == original.contract_digest()


class _ForgedObservedSha(str):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


class _ComparisonTrapSha(str):
    def __eq__(self, other: object) -> bool:
        raise AssertionError("unvalidated identity compared")

    def __ne__(self, other: object) -> bool:
        raise AssertionError("unvalidated identity compared")


@pytest.mark.parametrize("sha_type", [_ForgedObservedSha, _ComparisonTrapSha])
@pytest.mark.parametrize("kind", [EvidenceKind.INTEGRATION, EvidenceKind.POST_MERGE])
@pytest.mark.parametrize("field", ["head", "main"])
def test_evidence_rejects_non_native_observed_sha(
    sha_type: type[str], field: str, kind: EvidenceKind
) -> None:
    packet = _packet()
    evidence = _evidence(packet, kind=kind, main_sha=MAIN_SHA)
    assert not evidence_is_current(
        evidence,
        packet,
        head_sha=sha_type(OTHER_HEAD_SHA) if field == "head" else HEAD_SHA,
        current_main_sha=sha_type(OTHER_MAIN_SHA) if field == "main" else MAIN_SHA,
    )


class _ForgedRecordedSha(str):
    def __eq__(self, other: object) -> bool:
        return not (type(other) is str and other == "0" * 40)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class _RecordedComparisonTrapSha(_ComparisonTrapSha):
    # The legacy constructor checks that an input SHA is not all zeroes.
    # Allow that check, but fail on any freshness comparison.
    def __eq__(self, other: object) -> bool:
        if type(other) is str and other == "0" * 40:
            return False
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        if type(other) is str and other == "0" * 40:
            return True
        return super().__ne__(other)


@pytest.mark.parametrize("sha_type", [_ForgedRecordedSha, _RecordedComparisonTrapSha])
@pytest.mark.parametrize("kind", [EvidenceKind.INTEGRATION, EvidenceKind.POST_MERGE])
@pytest.mark.parametrize("field", ["head", "main"])
def test_evidence_rejects_non_native_recorded_sha(
    field: str, kind: EvidenceKind, sha_type: type[str]
) -> None:
    packet = _packet()
    evidence = _evidence(
        packet,
        kind=kind,
        head_sha=sha_type(OTHER_HEAD_SHA) if field == "head" else HEAD_SHA,
        main_sha=sha_type(OTHER_MAIN_SHA) if field == "main" else MAIN_SHA,
    )
    assert not evidence_is_current(
        evidence, packet, head_sha=HEAD_SHA, current_main_sha=MAIN_SHA
    )


@pytest.mark.parametrize("sha_type", [_ForgedRecordedSha, _RecordedComparisonTrapSha])
@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", "T500-02"),
        ("task_contract_digest", "f" * 64),
        ("base_sha", OTHER_HEAD_SHA),
    ],
)
def test_evidence_rejects_non_native_recorded_identity(
    field: str, value: str, sha_type: type[str]
) -> None:
    packet = _packet()
    evidence = replace(_evidence(packet), **{field: sha_type(value)})
    assert not evidence_is_current(evidence, packet, head_sha=HEAD_SHA)


@pytest.mark.parametrize("sha_type", [_ForgedRecordedSha, _RecordedComparisonTrapSha])
@pytest.mark.parametrize(
    "field,value", [("task_id", "T500-02"), ("base_sha", OTHER_HEAD_SHA)]
)
def test_evidence_rejects_non_native_packet_identity(
    field: str, value: str, sha_type: type[str]
) -> None:
    # Empty dependencies avoid invoking task-id equality in legacy admission.
    packet = replace(_packet(), dependencies=(), **{field: sha_type(value)})
    evidence = replace(
        _evidence(replace(_packet(), dependencies=())),
        task_contract_digest=packet.contract_digest(),
    )
    assert not evidence_is_current(evidence, packet, head_sha=HEAD_SHA)


@pytest.mark.parametrize("kind", list(EvidenceKind))
def test_evidence_kind_freshness_matrix(kind: EvidenceKind) -> None:
    packet = _packet()
    requires_main = kind in {EvidenceKind.INTEGRATION, EvidenceKind.POST_MERGE}
    evidence = _evidence(
        packet, kind=kind, main_sha=MAIN_SHA if requires_main else None
    )
    assert evidence_is_current(
        evidence, packet, head_sha=HEAD_SHA, current_main_sha=MAIN_SHA
    )
    assert not evidence_is_current(
        evidence, packet, head_sha=OTHER_HEAD_SHA, current_main_sha=MAIN_SHA
    )
    assert not evidence_is_current(
        evidence,
        packet,
        head_sha=_ComparisonTrapSha(HEAD_SHA),
        current_main_sha=MAIN_SHA,
    )
    assert evidence_is_current(evidence, packet, head_sha=HEAD_SHA) is not requires_main
    assert (
        evidence_is_current(
            evidence, packet, head_sha=HEAD_SHA, current_main_sha=OTHER_MAIN_SHA
        )
        is not requires_main
    )
    assert (
        evidence_is_current(
            evidence,
            packet,
            head_sha=HEAD_SHA,
            current_main_sha=_ComparisonTrapSha(OTHER_MAIN_SHA),
        )
        is not requires_main
    )


class _ForgedDigestPacket(TaskPacket):
    def contract_digest(self) -> str:
        return _ForgedRecordedSha("f" * 64)


class _TrapDigestPacket(TaskPacket):
    def contract_digest(self) -> str:
        return _ComparisonTrapSha("f" * 64)


@pytest.mark.parametrize("packet_type", [_ForgedDigestPacket, _TrapDigestPacket])
def test_evidence_rejects_non_native_expected_digest(
    packet_type: type[TaskPacket],
) -> None:
    native = _packet()
    packet = packet_type(**vars(native))
    assert not evidence_is_current(_evidence(native), packet, head_sha=HEAD_SHA)


@pytest.mark.parametrize("sha_type", [_ForgedRecordedSha, _RecordedComparisonTrapSha])
@pytest.mark.parametrize("side", ["current", "peer"])
def test_peer_integration_rejects_non_native_task_id(
    sha_type: type[str], side: str
) -> None:
    native = replace(_packet(), dependencies=())
    peer = replace(native, task_id="T500-02", resource_keys=())
    evidence = _evidence(native)
    packet = native
    if side == "current":
        packet = replace(native, task_id=sha_type(native.task_id))
    else:
        peer = replace(peer, task_id=sha_type(peer.task_id))
    assert not evidence_is_current_after_peer_integration(
        evidence, packet, peer, head_sha=HEAD_SHA
    )


class _RevisionSubclass(int):
    pass


@pytest.mark.parametrize("revision", [True, 1.0, _RevisionSubclass(1)])
def test_evidence_revision_requires_native_int(revision: int) -> None:
    packet = _packet()
    evidence = _evidence(packet)
    with pytest.raises(ValueError, match="task_revision"):
        replace(packet, task_revision=revision)
    with pytest.raises(ValueError, match="task_revision"):
        replace(evidence, task_revision=revision)


@pytest.mark.parametrize("kind", list(EvidenceKind))
def test_evidence_kind_constructor_normalizes_native_string(kind: EvidenceKind) -> None:
    packet = _packet()
    requires_main = kind in {EvidenceKind.INTEGRATION, EvidenceKind.POST_MERGE}
    evidence = replace(
        _evidence(packet, kind=kind, main_sha=MAIN_SHA if requires_main else None),
        evidence_kind=kind.value,
    )
    assert evidence.evidence_kind is kind
    assert evidence_is_current(
        evidence, packet, head_sha=HEAD_SHA, current_main_sha=MAIN_SHA
    )
    with pytest.raises(ValueError):
        replace(evidence, evidence_kind="not-a-kind")


def test_peer_integration_native_task_id_controls() -> None:
    packet = _packet()
    evidence = _evidence(packet)
    peer = replace(packet, task_id="T500-02", resource_keys=())
    assert evidence_is_current_after_peer_integration(
        evidence, packet, peer, head_sha=HEAD_SHA
    )
    assert not evidence_is_current_after_peer_integration(
        evidence,
        packet,
        replace(peer, resource_keys=packet.resource_keys),
        head_sha=HEAD_SHA,
    )
    with pytest.raises(ValueError, match="distinct task"):
        evidence_is_current_after_peer_integration(
            evidence, packet, packet, head_sha=HEAD_SHA
        )


class _NeverEqualTaskId(str):
    def __eq__(self, other: object) -> bool:
        return False

    def __ne__(self, other: object) -> bool:
        return True


def test_peer_integration_cannot_disguise_same_task() -> None:
    packet = _packet()
    evidence = _evidence(packet)
    peer = replace(
        packet,
        task_id=_NeverEqualTaskId(packet.task_id),
        dependencies=(),
        resource_keys=(),
    )
    assert not evidence_is_current_after_peer_integration(
        evidence, packet, peer, head_sha=HEAD_SHA
    )


@pytest.mark.parametrize("method", ["contract_digest", "canonical_payload"])
def test_evidence_rejects_subclass_forging_native_contract(method: str) -> None:
    previous = _packet()
    changed = _packet(objective="Changed objective")
    assert changed.contract_digest() != previous.contract_digest()

    class ForgedDigestPacket(TaskPacket):
        def contract_digest(self) -> str:
            if method == "contract_digest":
                return previous.contract_digest()
            return super().contract_digest()

        def canonical_payload(self):
            if method == "canonical_payload":
                return previous.canonical_payload()
            return super().canonical_payload()

    packet = ForgedDigestPacket(**vars(changed))
    assert not evidence_is_current(_evidence(previous), packet, head_sha=HEAD_SHA)


class _AlwaysEqualRevision(int):
    def __ne__(self, other: object) -> bool:
        return False


@pytest.mark.parametrize("side", ["packet", "evidence"])
def test_evidence_rejects_revision_override(side: str) -> None:
    previous = _packet()
    evidence = _evidence(previous)

    class ForgedRevisionPacket(TaskPacket):
        def __getattribute__(self, name: str):
            if name == "task_revision" and vars(self).get("_armed", False):
                return _AlwaysEqualRevision(2)
            return super().__getattribute__(name)

    class ForgedRevisionEvidence(EvidenceRecord):
        def __getattribute__(self, name: str):
            if name == "task_revision" and vars(self).get("_armed", False):
                return _AlwaysEqualRevision(2)
            return super().__getattribute__(name)

    packet = previous
    if side == "packet":
        packet = ForgedRevisionPacket(**vars(previous))
        object.__setattr__(packet, "_armed", True)
    else:
        evidence = ForgedRevisionEvidence(**vars(evidence))
        object.__setattr__(evidence, "_armed", True)
    assert not evidence_is_current(evidence, packet, head_sha=HEAD_SHA)


@pytest.mark.parametrize("kind", [EvidenceKind.INTEGRATION, EvidenceKind.POST_MERGE])
def test_evidence_rejects_kind_override(kind: EvidenceKind) -> None:
    packet = _packet()
    original = _evidence(packet, kind=kind, main_sha=MAIN_SHA)

    class ForgedKindEvidence(EvidenceRecord):
        def __getattribute__(self, name: str):
            if name == "evidence_kind" and vars(self).get("_armed", False):
                return EvidenceKind.REVIEW
            return super().__getattribute__(name)

    evidence = ForgedKindEvidence(**vars(original))
    object.__setattr__(evidence, "_armed", True)
    assert not evidence_is_current(
        evidence, packet, head_sha=HEAD_SHA, current_main_sha=OTHER_MAIN_SHA
    )


@pytest.mark.parametrize("side", ["packet", "evidence", "peer"])
def test_freshness_rejects_subclass_before_attribute_access(side: str) -> None:
    class TrapPacket(TaskPacket):
        def __getattribute__(self, name: str):
            raise AssertionError(f"unvalidated packet attribute: {name}")

    class TrapEvidence(EvidenceRecord):
        def __getattribute__(self, name: str):
            raise AssertionError(f"unvalidated evidence attribute: {name}")

    packet = _packet()
    evidence = _evidence(packet)
    peer = replace(packet, task_id="T500-02", resource_keys=())
    if side == "packet":
        packet = object.__new__(TrapPacket)
    elif side == "evidence":
        evidence = object.__new__(TrapEvidence)
    else:
        peer = object.__new__(TrapPacket)
    if side != "peer":
        assert not evidence_is_current(evidence, packet, head_sha=HEAD_SHA)
    assert not evidence_is_current_after_peer_integration(
        evidence, packet, peer, head_sha=HEAD_SHA
    )


def test_peer_freshness_rejects_even_nonconflicting_packet_subclass() -> None:
    class PeerSubclass(TaskPacket):
        pass

    packet = _packet()
    peer = replace(packet, task_id="T500-02", resource_keys=())
    evidence = _evidence(packet)
    assert evidence_is_current_after_peer_integration(
        evidence, packet, peer, head_sha=HEAD_SHA
    )
    assert not evidence_is_current_after_peer_integration(
        evidence, packet, PeerSubclass(**vars(peer)), head_sha=HEAD_SHA
    )
