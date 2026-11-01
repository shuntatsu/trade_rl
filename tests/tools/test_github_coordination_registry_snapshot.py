"""Read-only registry snapshot regression tests."""

from __future__ import annotations

import base64
import hashlib

import pytest

from tools.agent_repo.coordination.registry import CoordinatorRegistry
from tools.github_coordination_registry_snapshot import observe_registry_snapshot

REPOSITORY = "shuntatsu/trade-rl-coordination-disposable"
REF = "refs/heads/coordination/task-registry"
COMMIT = "a" * 40
TREE = "b" * 40
PATH = "coordination/registry.json"


class FakeGet:
    def __init__(self) -> None:
        self.data = CoordinatorRegistry(REPOSITORY).canonical_bytes()
        self.blob = hashlib.sha1(
            b"blob " + str(len(self.data)).encode() + b"\x00" + self.data
        ).hexdigest()
        self.ref_sha = COMMIT
        self.repository_node_id = "R_123"
        self.calls: list[str] = []

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        assert method == "GET"
        self.calls.append(suffix)
        if suffix == "":
            return {
                "full_name": REPOSITORY,
                "node_id": self.repository_node_id,
                "id": 123,
            }
        if suffix == "git/ref/heads/coordination/task-registry":
            return {"ref": REF, "object": {"type": "commit", "sha": self.ref_sha}}
        if suffix == f"git/commits/{COMMIT}":
            return {"sha": COMMIT, "tree": {"sha": TREE}}
        if suffix == f"git/trees/{TREE}":
            assert query == {"recursive": "1"}
            return {
                "sha": TREE,
                "truncated": False,
                "tree": [
                    {
                        "path": PATH,
                        "mode": "100644",
                        "type": "blob",
                        "sha": self.blob,
                        "size": len(self.data),
                    }
                ],
            }
        if suffix == f"git/blobs/{self.blob}":
            return {
                "sha": self.blob,
                "size": len(self.data),
                "encoding": "base64",
                "content": base64.b64encode(self.data).decode(),
            }
        raise AssertionError(f"unexpected read: {suffix}")


def _observe(api: FakeGet):
    return observe_registry_snapshot(
        api,
        expected_repository=REPOSITORY,
        ref_name=REF,
        registry_path=PATH,
    )


def test_stable_snapshot_binds_ref_oid_and_canonical_digest() -> None:
    api = FakeGet()
    result = _observe(api)
    assert result.ref_oid == COMMIT
    assert result.registry_digest == CoordinatorRegistry(REPOSITORY).digest()
    assert result.ref_readback_verified is True
    assert result.authoritative_registry_established is False
    assert result.task_registry_complete is False
    assert result.cleanup_authorized is False
    assert api.calls == [
        "",
        "git/ref/heads/coordination/task-registry",
        f"git/commits/{COMMIT}",
        f"git/trees/{TREE}",
        f"git/blobs/{api.blob}",
        "git/ref/heads/coordination/task-registry",
        "",
    ]


def test_invalid_ref_fails_before_any_get() -> None:
    api = FakeGet()
    with pytest.raises(ValueError, match="coordination"):
        observe_registry_snapshot(
            api,
            expected_repository=REPOSITORY,
            ref_name="refs/heads/main",
            registry_path=PATH,
        )
    assert api.calls == []


def test_missing_ref_is_not_treated_as_empty_registry() -> None:
    api = FakeGet()
    api.ref_sha = "0" * 40
    with pytest.raises(ValueError, match="ref"):
        _observe(api)
    assert len(api.calls) == 2
