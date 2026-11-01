from __future__ import annotations

import json

import pytest

from tools.agent_repo.coordination.enrollment import EnrollmentProjection
from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import lease_branch_name
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase
from tools.agent_repo.coordination.registry import (
    CoordinatorRegistry,
    parse_registry_bytes,
)

REPOSITORY = "shuntatsu/trade_rl"
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
DIGEST = "0" * 64


def _projection(
    task_id: str = "T1067-01",
    *,
    repository: str = REPOSITORY,
    issue_number: int = 1067,
) -> EnrollmentProjection:
    status = TaskStatusRecord(
        task_id=task_id,
        task_revision=1,
        task_contract_digest=DIGEST,
        phase=TaskPhase.EXECUTING,
        condition=TaskCondition.HEALTHY,
        base_sha=BASE_SHA,
        owner="agent-a",
        lease_epoch=1,
        lease_branch=lease_branch_name(task_id, 1, "agent-a"),
        head_sha=HEAD_SHA,
    )
    return EnrollmentProjection(
        repository=repository,
        issue_number=issue_number,
        issue_node_id=f"I_{task_id}",
        status=status,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def test_registry_order_bytes_and_digest_are_deterministic() -> None:
    first = _projection("T1067-02")
    second = _projection("T1067-01")
    left = CoordinatorRegistry(REPOSITORY, (first, second))
    right = CoordinatorRegistry(REPOSITORY, (second, first))

    assert left == right
    assert left.entries == right.entries
    assert left.canonical_bytes() == right.canonical_bytes()
    assert left.digest() == right.digest()
    assert [
        entry["status"]["task_id"] for entry in left.canonical_payload()["entries"]
    ] == [
        "T1067-01",
        "T1067-02",
    ]
    assert parse_registry_bytes(left.canonical_bytes()) == left


def test_registry_round_trip_requires_exact_canonical_bytes() -> None:
    registry = CoordinatorRegistry(REPOSITORY, (_projection(),))
    encoded = registry.canonical_bytes()

    assert parse_registry_bytes(encoded) == registry
    assert parse_registry_bytes(encoded).canonical_bytes() == encoded

    with pytest.raises(ValueError, match="not canonical"):
        parse_registry_bytes(b" " + encoded)


def test_empty_registry_is_valid_schema_state_without_authority_flags() -> None:
    registry = CoordinatorRegistry(REPOSITORY)
    payload = registry.canonical_payload()

    assert payload == {
        "schema": "coordinator_registry_v1",
        "repository": REPOSITORY,
        "entries": [],
    }
    assert "task_registry_complete" not in payload
    assert "cleanup_authorized" not in payload
    assert parse_registry_bytes(registry.canonical_bytes()) == registry


def test_duplicate_task_id_and_cross_repository_entry_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate Task IDs"):
        CoordinatorRegistry(
            REPOSITORY,
            (_projection("T1067-01"), _projection("T1067-01", issue_number=1068)),
        )

    with pytest.raises(ValueError, match="repository differs"):
        CoordinatorRegistry(
            REPOSITORY,
            (_projection(repository="other/repository"),),
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda payload: payload.update({"schema": "coordinator_registry_v2"}),
            "schema",
        ),
        (
            lambda payload: payload.update({"unknown": True}),
            "unknown field",
        ),
        (
            lambda payload: payload.pop("repository"),
            "missing field",
        ),
        (
            lambda payload: payload["entries"][0].update({"unknown": True}),
            "unknown field",
        ),
        (
            lambda payload: payload["entries"][0].pop("issue_node_id"),
            "missing field",
        ),
        (
            lambda payload: payload["entries"][0]["status"].update({"unknown": True}),
            "unknown field",
        ),
        (
            lambda payload: payload["entries"][0]["status"].pop("task_revision"),
            "missing field",
        ),
        (
            lambda payload: payload["entries"][0]["status"].update(
                {"task_revision": 1.0}
            ),
            "registry entry is invalid",
        ),
    ],
)
def test_registry_rejects_schema_shape_and_nested_identity_drift(
    mutate: object, match: str
) -> None:
    payload = CoordinatorRegistry(REPOSITORY, (_projection(),)).canonical_payload()
    mutate(payload)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        parse_registry_bytes(_canonical_json(payload))


def test_duplicate_json_fields_are_rejected_at_top_and_nested_levels() -> None:
    encoded = (
        CoordinatorRegistry(REPOSITORY, (_projection(),)).canonical_bytes().decode()
    )

    top_duplicate = encoded.replace(
        '"repository":"shuntatsu/trade_rl"',
        '"repository":"shuntatsu/trade_rl","repository":"shuntatsu/trade_rl"',
        1,
    )
    with pytest.raises(ValueError, match="duplicate field"):
        parse_registry_bytes(top_duplicate.encode())

    nested_duplicate = encoded.replace(
        '"task_revision":1',
        '"task_revision":1,"task_revision":1',
        1,
    )
    with pytest.raises(ValueError, match="duplicate field"):
        parse_registry_bytes(nested_duplicate.encode())


def test_registry_rejects_nonlist_entries_invalid_utf8_and_nonbytes() -> None:
    payload = CoordinatorRegistry(REPOSITORY).canonical_payload()
    payload["entries"] = {}
    with pytest.raises(ValueError, match="must be a list"):
        parse_registry_bytes(_canonical_json(payload))

    with pytest.raises(ValueError, match="UTF-8"):
        parse_registry_bytes(b"\xff")

    with pytest.raises(ValueError, match="must be bytes"):
        parse_registry_bytes("{}".encode().decode())  # type: ignore[arg-type]
