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
        self.next_repository_node_id: str | None = None
        self.next_ref_sha: str | None = None
        self.ref_type = "commit"
        self.repo_reads = 0
        self.ref_reads = 0
        self.fail_suffix: str | None = None
        self.calls: list[str] = []

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        assert method == "GET"
        self.calls.append(suffix)
        if suffix == self.fail_suffix:
            raise RuntimeError("GitHub GET unavailable")
        if suffix == "":
            self.repo_reads += 1
            node_id = self.repository_node_id
            if self.repo_reads > 1 and self.next_repository_node_id is not None:
                node_id = self.next_repository_node_id
            return {"full_name": REPOSITORY, "node_id": node_id, "id": 123}
        if suffix == "git/ref/heads/coordination/task-registry":
            self.ref_reads += 1
            oid = self.ref_sha
            if self.ref_reads > 1 and self.next_ref_sha is not None:
                oid = self.next_ref_sha
            return {"ref": REF, "object": {"type": self.ref_type, "sha": oid}}
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


def test_ref_drift_after_object_read_fails_without_success() -> None:
    api = FakeGet()
    api.next_ref_sha = "c" * 40
    with pytest.raises(ValueError, match="ref changed"):
        _observe(api)
    assert api.ref_reads == 2
    assert api.repo_reads == 1


def test_repository_identity_change_after_object_read_fails() -> None:
    api = FakeGet()
    api.next_repository_node_id = "R_replaced"
    with pytest.raises(ValueError, match="repository identity changed"):
        _observe(api)
    assert api.ref_reads == 2
    assert api.repo_reads == 2


def test_invalid_target_type_rejected_before_blob_access() -> None:
    api = FakeGet()
    api.ref_type = "blob"
    with pytest.raises(ValueError, match="target must be a commit"):
        _observe(api)
    assert api.calls == ["", "git/ref/heads/coordination/task-registry"]


def test_corrupt_blob_contents_fail_before_ref_readback() -> None:
    api = FakeGet()
    api.data = b"corrupted"
    with pytest.raises(ValueError, match="Git blob SHA"):
        _observe(api)
    assert api.ref_reads == 1


def test_api_failure_propagates_and_does_not_treat_unknown_as_empty() -> None:
    api = FakeGet()
    api.fail_suffix = "git/trees/" + TREE
    with pytest.raises(RuntimeError, match="GET unavailable"):
        _observe(api)
    assert api.ref_reads == 1


def test_bad_path_fails_before_any_get() -> None:
    api = FakeGet()
    with pytest.raises(ValueError, match="registry path"):
        observe_registry_snapshot(
            api,
            expected_repository=REPOSITORY,
            ref_name=REF,
            registry_path="../registry.json",
        )
    assert api.calls == []


def test_observation_is_not_enrollment_permission_even_with_empty_registry() -> None:
    result = _observe(FakeGet())
    assert result.registry.entries == ()
    assert result.authoritative_registry_established is False
    assert result.task_registry_complete is False
    assert result.cleanup_authorized is False
