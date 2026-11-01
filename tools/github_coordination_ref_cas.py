"""Test-only GitHub GraphQL ref CAS primitive for Coordinator integration E2E.

This module deliberately does not expose a CLI and refuses mutation unless an
explicit test-only flag is supplied. It validates the exact authenticated viewer,
repository identity, expected ref OID, atomic updateRefs request and independent
readback. A successful call is NOT authoritative Task enrollment, protection
proof, repository-wide roster completeness, or cleanup authorization.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple, Protocol

ZERO_OID = "0" * 40
_REF_PREFIX = "refs/heads/coordination/"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}[A-Za-z0-9])?$")

_READ_REF_QUERY = """
query CoordinationRefState($owner: String!, $name: String!, $ref: String!) {
  viewer { login }
  repository(owner: $owner, name: $name) {
    id
    ref(qualifiedName: $ref) { target { oid } }
  }
}
""".strip()

_UPDATE_REF_MUTATION = """
mutation UpdateCoordinationRef($input: UpdateRefsInput!) {
  updateRefs(input: $input) { clientMutationId }
}
""".strip()


class GraphQLApi(Protocol):
    def execute(
        self, document: str, variables: dict[str, object]
    ) -> dict[str, object]: ...


class _Snapshot(NamedTuple):
    viewer_login: str
    repository_id: str
    oid: str


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _sha(value: Any, name: str, *, allow_zero: bool = True) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 40-hex Git OID")
    if not allow_zero and value == ZERO_OID:
        raise ValueError(f"{name} must not be the zero OID")
    return value


def _repository(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, str) or not _REPOSITORY_RE.fullmatch(value):
        raise ValueError("repository must use owner/name form")
    owner, name = value.split("/", 1)
    return value, owner, name


def _ref_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(_REF_PREFIX)
        or len(value) <= len(_REF_PREFIX)
        or value.endswith("/")
        or ".." in value
        or "//" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError("ref must be inside the coordination branch namespace")
    return value


def _login(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _LOGIN_RE.fullmatch(value):
        raise ValueError(f"{name} must be a canonical GitHub login")
    return value


def _graphql_data(payload: Any, operation: str) -> dict[str, Any]:
    response = _mapping(payload, f"{operation} response")
    errors = response.get("errors")
    if errors not in (None, []):
        raise RuntimeError(f"{operation} returned GraphQL errors")
    return _mapping(response.get("data"), f"{operation} data")


def _read_snapshot(
    api: GraphQLApi,
    *,
    repository: str,
    ref_name: str,
) -> _Snapshot:
    _, owner, name = _repository(repository)
    ref_name = _ref_name(ref_name)
    data = _graphql_data(
        api.execute(
            _READ_REF_QUERY,
            {"owner": owner, "name": name, "ref": ref_name},
        ),
        "coordination ref read",
    )
    viewer = _mapping(data.get("viewer"), "viewer")
    viewer_login = _login(viewer.get("login"), "authenticated viewer login")
    repo = _mapping(data.get("repository"), "repository")
    repository_id = _text(repo.get("id"), "repository node id")
    raw_ref = repo.get("ref")
    if raw_ref is None:
        oid = ZERO_OID
    else:
        ref = _mapping(raw_ref, "repository ref")
        target = _mapping(ref.get("target"), "repository ref target")
        oid = _sha(target.get("oid"), "repository ref OID", allow_zero=False)
    return _Snapshot(viewer_login, repository_id, oid)


def compare_and_swap_coordination_ref(
    api: GraphQLApi,
    *,
    repository: str,
    ref_name: str,
    expected_oid: str,
    after_oid: str,
    expected_coordinator_login: str,
    allow_unprotected_test_mutation: bool = False,
) -> dict[str, object]:
    """Perform one exact beforeOid/afterOid CAS for disposable integration tests.

    The default is a hard refusal. Setting ``allow_unprotected_test_mutation``
    explicitly acknowledges that external ref protection and single-writer policy
    have NOT been established. This primitive must not be used as cleanup or Task
    enrollment authority in the production repository.
    """

    repository, _, _ = _repository(repository)
    ref_name = _ref_name(ref_name)
    expected_oid = _sha(expected_oid, "expected OID")
    after_oid = _sha(after_oid, "after OID", allow_zero=False)
    expected_login = _login(expected_coordinator_login, "expected Coordinator login")
    if expected_oid == after_oid:
        raise ValueError("expected and after OIDs must be different")
    if after_oid == ZERO_OID:
        raise ValueError("ref deletion is not allowed by this CAS primitive")
    if allow_unprotected_test_mutation is not True:
        raise ValueError("test-only mutation is disabled without explicit opt-in")

    before = _read_snapshot(api, repository=repository, ref_name=ref_name)
    if before.viewer_login.casefold() != expected_login.casefold():
        raise ValueError("authenticated Coordinator does not match expected login")
    if before.oid != expected_oid:
        raise ValueError("coordination ref does not match expected OID")

    mutation_id = f"coordination-cas:{expected_oid[:12]}:{after_oid[:12]}"
    mutation_data = _graphql_data(
        api.execute(
            _UPDATE_REF_MUTATION,
            {
                "input": {
                    "repositoryId": before.repository_id,
                    "clientMutationId": mutation_id,
                    "refUpdates": [
                        {
                            "name": ref_name,
                            "beforeOid": expected_oid,
                            "afterOid": after_oid,
                            "force": False,
                        }
                    ],
                }
            },
        ),
        "coordination ref update",
    )
    update = _mapping(mutation_data.get("updateRefs"), "updateRefs result")
    if update.get("clientMutationId") != mutation_id:
        raise ValueError("updateRefs clientMutationId readback differs")

    after = _read_snapshot(api, repository=repository, ref_name=ref_name)
    if after.viewer_login.casefold() != expected_login.casefold():
        raise ValueError("authenticated Coordinator changed during transaction")
    if after.repository_id != before.repository_id:
        raise ValueError("repository identity changed during transaction")
    if after.oid != after_oid:
        raise ValueError("coordination ref readback OID differs from requested OID")

    return {
        "schema": "coordination_ref_cas_test_observation_v1",
        "repository": repository,
        "ref_name": ref_name,
        "coordinator_login": before.viewer_login,
        "before_oid": expected_oid,
        "after_oid": after_oid,
        "readback_verified": True,
        "test_only_unprotected_mutation": True,
        "authoritative_registry_established": False,
        "task_registry_complete": False,
        "cleanup_authorized": False,
    }


__all__ = ["ZERO_OID", "compare_and_swap_coordination_ref"]
