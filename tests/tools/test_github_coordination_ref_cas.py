from __future__ import annotations

from copy import deepcopy

import pytest

from tools.github_coordination_ref_cas import (
    ZERO_OID,
    compare_and_swap_coordination_ref,
)

REPO = "shuntatsu/trade-rl-coordination-disposable"
REF = "refs/heads/coordination/task-registry"
A = "a" * 40
B = "b" * 40
_UNSET = object()


class FakeGraphQL:
    def __init__(self, *, oid: str | None = A, viewer: str = "coordinator-bot") -> None:
        self.oid = oid
        self.viewer = viewer
        self.repository_id = "R_repo"
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail_mutation = False
        self.readback_oid: str | None | object = _UNSET
        self.readback_viewer: str | None = None
        self.readback_repository_id: str | None = None
        self.reads = 0

    def execute(self, document: str, variables: dict[str, object]) -> dict[str, object]:
        self.calls.append((document, deepcopy(variables)))
        if "mutation UpdateCoordinationRef" in document:
            if self.fail_mutation:
                raise RuntimeError("GraphQL mutation failed")
            update = variables["input"]
            assert isinstance(update, dict)
            ref_updates = update["refUpdates"]
            assert isinstance(ref_updates, list)
            item = ref_updates[0]
            assert isinstance(item, dict)
            before = item["beforeOid"]
            current = self.oid if self.oid is not None else ZERO_OID
            if before != current:
                raise RuntimeError("beforeOid rejected")
            self.oid = item["afterOid"]  # type: ignore[assignment]
            return {
                "data": {"updateRefs": {"clientMutationId": update["clientMutationId"]}}
            }

        self.reads += 1
        oid = self.oid
        viewer = self.viewer
        repo_id = self.repository_id
        if self.reads >= 2 and self.readback_oid is not _UNSET:
            oid = self.readback_oid  # type: ignore[assignment]
        if self.reads >= 2 and self.readback_viewer is not None:
            viewer = self.readback_viewer
        if self.reads >= 2 and self.readback_repository_id is not None:
            repo_id = self.readback_repository_id
        ref = None if oid is None else {"target": {"oid": oid}}
        return {
            "data": {
                "viewer": {"login": viewer},
                "repository": {"id": repo_id, "ref": ref},
            }
        }


def _api(*, oid: str | None = A, viewer: str = "coordinator-bot") -> FakeGraphQL:
    return FakeGraphQL(oid=oid, viewer=viewer)


def _cas(api: FakeGraphQL, **changes: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "repository": REPO,
        "ref_name": REF,
        "expected_oid": A,
        "after_oid": B,
        "expected_coordinator_login": "coordinator-bot",
        "allow_unprotected_test_mutation": True,
    }
    kwargs.update(changes)
    return compare_and_swap_coordination_ref(api, **kwargs)  # type: ignore[arg-type]


def test_production_repository_is_never_a_mutation_target() -> None:
    api = _api()
    with pytest.raises(ValueError, match="production repository"):
        _cas(api, repository="shuntatsu/trade_rl")
    assert api.calls == []


def test_mutation_is_disabled_by_default() -> None:
    api = _api()
    with pytest.raises(ValueError, match="test-only mutation is disabled"):
        compare_and_swap_coordination_ref(
            api,
            repository=REPO,
            ref_name=REF,
            expected_oid=A,
            after_oid=B,
            expected_coordinator_login="coordinator-bot",
        )
    assert api.calls == []


def test_test_only_cas_binds_actor_expected_oid_and_readback() -> None:
    api = _api()
    result = _cas(api)
    assert result["before_oid"] == A
    assert result["after_oid"] == B
    assert result["readback_verified"] is True
    assert result["authoritative_registry_established"] is False
    assert result["cleanup_authorized"] is False
    assert len(api.calls) == 3
    _, variables = api.calls[1]
    mutation_input = variables["input"]
    assert isinstance(mutation_input, dict)
    assert mutation_input["repositoryId"] == "R_repo"
    assert mutation_input["refUpdates"] == [
        {
            "name": REF,
            "beforeOid": A,
            "afterOid": B,
            "force": False,
        }
    ]


def test_initial_creation_uses_zero_oid_absence_precondition() -> None:
    api = _api(oid=None)
    result = _cas(api, expected_oid=ZERO_OID)
    assert result["before_oid"] == ZERO_OID
    assert api.oid == B


def test_stale_expected_oid_is_rejected_before_mutation() -> None:
    api = _api(oid=A)
    with pytest.raises(ValueError, match="expected OID"):
        _cas(api, expected_oid="c" * 40)
    assert len(api.calls) == 1


def test_wrong_authenticated_actor_is_rejected_before_mutation() -> None:
    api = _api(viewer="another-user")
    with pytest.raises(ValueError, match="authenticated Coordinator"):
        _cas(api)
    assert len(api.calls) == 1


def test_readback_oid_actor_and_repository_drift_are_rejected() -> None:
    api = _api()
    api.readback_oid = "c" * 40
    with pytest.raises(ValueError, match="readback OID"):
        _cas(api)

    api = _api()
    api.readback_viewer = "other-user"
    with pytest.raises(ValueError, match="authenticated Coordinator changed"):
        _cas(api)

    api = _api()
    api.readback_repository_id = "R_other"
    with pytest.raises(ValueError, match="repository identity changed"):
        _cas(api)


def test_mutation_error_propagates_without_retry_or_success_claim() -> None:
    api = _api()
    api.fail_mutation = True
    with pytest.raises(RuntimeError, match="mutation failed"):
        _cas(api)
    assert len(api.calls) == 2
    assert api.oid == A


def test_graphql_errors_and_bad_mutation_readback_fail_closed() -> None:
    class ErrorApi(FakeGraphQL):
        def execute(
            self, document: str, variables: dict[str, object]
        ) -> dict[str, object]:
            if "mutation UpdateCoordinationRef" in document:
                return {"errors": [{"message": "denied"}], "data": None}
            return super().execute(document, variables)

    with pytest.raises(RuntimeError, match="GraphQL errors"):
        _cas(ErrorApi())

    class BadMutationApi(FakeGraphQL):
        def execute(
            self, document: str, variables: dict[str, object]
        ) -> dict[str, object]:
            if "mutation UpdateCoordinationRef" in document:
                return {"data": {"updateRefs": {"clientMutationId": "other"}}}
            return super().execute(document, variables)

    with pytest.raises(ValueError, match="clientMutationId"):
        _cas(BadMutationApi())


@pytest.mark.parametrize(
    "ref_name",
    ["refs/heads/main", "refs/tags/coordination/task-registry", "coordination/x"],
)
def test_only_coordination_branch_namespace_is_allowed(ref_name: str) -> None:
    api = _api()
    with pytest.raises(ValueError, match="coordination branch namespace"):
        _cas(api, ref_name=ref_name)
    assert api.calls == []


def test_deletion_noop_and_malformed_identity_are_rejected() -> None:
    api = _api()
    with pytest.raises(ValueError, match="zero OID"):
        _cas(api, after_oid=ZERO_OID)
    with pytest.raises(ValueError, match="different"):
        _cas(api, after_oid=A)
    with pytest.raises(ValueError, match="repository"):
        _cas(api, repository="bad")
    with pytest.raises(ValueError, match="canonical GitHub login"):
        _cas(api, expected_coordinator_login="bad login")
    assert api.calls == []
