"""Explicit, test-only GitHub HTTPS driver for Coordinator registry CAS.

This is not a production Coordinator or registry authority. It cannot create a
sandbox repository. It never mutates trade_rl; the target must explicitly end
in -coordination-disposable, and mutation requires an opt-in CLI flag. The
caller prepares two real immutable Git commits containing canonical registry
bytes, then this driver reuses the existing exact-OID updateRefs CAS.

Ambiguous transport failures MUST NOT be retried or assumed rolled back.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Protocol

from tools.github_coordination_ref_cas import (
    ZERO_OID,
    _READ_REF_QUERY,
    _UPDATE_REF_MUTATION,
    _graphql_data,
    _login,
    _mapping,
    _repository,
    _sha,
    _text,
    _validated_test_only_update,
)

if TYPE_CHECKING:
    from tools.agent_repo.coordination.registry import RegistryTransitionPlan

_GRAPHQL_URL = "https://api.github.com/graphql"
_API_URL = "https://api.github.com"
_MAX_RESPONSE_BYTES = 1024 * 1024
_MAX_REQUEST_BYTES = 64 * 1024
_GIT_READ_SUFFIX = re.compile(r"git/(?:commits|trees|blobs)/[0-9a-f]{40}")


class RegistryObjectReader(Protocol):
    def read_registry_bytes(self, repository: str, commit_oid: str) -> bytes: ...


class GraphQLApi(Protocol):
    def execute(
        self, document: str, variables: dict[str, object]
    ) -> dict[str, object]: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent forwarding a bearer token to any redirect target."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del request, fp, code, msg, headers, newurl
        return None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError("GitHub JSON response has duplicate JSON fields")
        data[key] = value
    return data


def _reject_nonfinite_json_constant(_constant: str) -> None:
    raise ValueError("GitHub JSON response has non-finite JSON constants")


class GitHubGraphQLHttpApi:
    """Bound, test-only GraphQL ref transport; no arbitrary GraphQL capability.

    Reads are scoped to a disposable repository/ref and expected actor. One
    explicitly opted-in exact-OID mutation requires a validated prior read.
    After a mutation attempt (including a transport failure) there is no retry.
    """

    def __init__(
        self,
        token: str,
        *,
        repository: str,
        ref_name: str,
        expected_oid: str,
        after_oid: str,
        expected_coordinator_login: str,
        allow_disposable_mutation: bool = False,
    ) -> None:
        if not isinstance(token, str) or not token or any(c.isspace() for c in token):
            raise ValueError("GitHub token must be a non-empty single-line secret")
        if type(allow_disposable_mutation) is not bool:
            raise ValueError("disposable mutation mode must be an explicit boolean")
        (
            repository,
            ref_name,
            expected_oid,
            after_oid,
            expected_login,
        ) = _validated_test_only_update(
            repository=repository,
            ref_name=ref_name,
            expected_oid=expected_oid,
            after_oid=after_oid,
            expected_coordinator_login=expected_coordinator_login,
            allow_unprotected_test_mutation=True,  # target validation, not a grant
        )
        _, owner, name = _repository(repository)
        self._read_variables: dict[str, object] = {
            "owner": owner,
            "name": name,
            "ref": ref_name,
        }
        self._ref_name = ref_name
        self._expected_oid = expected_oid
        self._after_oid = after_oid
        self._expected_login = expected_login
        self._mutation_enabled = allow_disposable_mutation
        self._repository_id: str | None = None
        self._mutation_attempted = False
        self._lock = threading.Lock()
        self._token = token

    def _request(self, request: urllib.request.Request) -> bytes:
        try:
            opener = urllib.request.build_opener(_NoRedirect())
            with opener.open(request, timeout=30) as response:
                return response.read(_MAX_RESPONSE_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RuntimeError(
                "GitHub GraphQL transport failed; mutation outcome may be unknown"
            ) from None

    def _require_readback(self, decoded: dict[str, Any]) -> None:
        data = _graphql_data(decoded, "coordination ref read")
        viewer = _mapping(data.get("viewer"), "viewer")
        login = _login(viewer.get("login"), "authenticated viewer login")
        if login.casefold() != self._expected_login.casefold():
            raise ValueError("authenticated Coordinator differs from expected login")
        repo = _mapping(data.get("repository"), "repository")
        repository_id = _text(repo.get("id"), "repository node id")
        ref = repo.get("ref")
        if ref is None:
            oid = ZERO_OID
        else:
            ref_object = _mapping(ref, "repository ref")
            target = _mapping(ref_object.get("target"), "ref target")
            oid = _sha(target.get("oid"), "repository ref OID", allow_zero=False)
        with self._lock:
            if self._repository_id is not None and repository_id != self._repository_id:
                raise ValueError("repository identity changed during GraphQL transport")
            if self._mutation_enabled:
                wanted = (
                    self._after_oid if self._mutation_attempted else self._expected_oid
                )
                if oid != wanted:
                    raise ValueError("coordination ref does not match bound OID")
            self._repository_id = repository_id

    def _require_mutation(self, variables: dict[str, object]) -> None:
        with self._lock:
            if not self._mutation_enabled:
                raise ValueError("disposable GraphQL mutation is disabled")
            if self._mutation_attempted:
                raise ValueError("disposable GraphQL mutation cannot be retried")
            if self._repository_id is None:
                raise ValueError("disposable GraphQL mutation requires prior ref read")
            mutation_id = (
                f"coordination-cas:{self._expected_oid[:12]}:{self._after_oid[:12]}"
            )
            expected: dict[str, object] = {
                "input": {
                    "repositoryId": self._repository_id,
                    "clientMutationId": mutation_id,
                    "refUpdates": [
                        {
                            "name": self._ref_name,
                            "beforeOid": self._expected_oid,
                            "afterOid": self._after_oid,
                            "force": False,
                        }
                    ],
                }
            }
            input_value = variables.get("input")
            updates = (
                input_value.get("refUpdates")
                if type(input_value) is dict
                else None
            )
            update = (
                updates[0]
                if type(updates) is list
                and len(updates) == 1
                and type(updates[0]) is dict
                else None
            )
            if (
                variables != expected
                or update is None
                or update.get("force") is not False
            ):
                raise ValueError("disposable GraphQL mutation is not an exact CAS")
            # Set before HTTP I/O: a timeout/HTTP error may follow a remote commit.
            self._mutation_attempted = True

    def execute(self, document: str, variables: dict[str, object]) -> dict[str, object]:
        if not isinstance(document, str) or not document.strip():
            raise ValueError("GraphQL document must be non-empty")
        if not isinstance(variables, dict):
            raise ValueError("GraphQL variables must be a mapping")
        data = json.dumps(
            {"query": document, "variables": variables},
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(data) > _MAX_REQUEST_BYTES:
            raise ValueError("GraphQL request exceeds the bounded payload limit")
        if document == _READ_REF_QUERY:
            if variables != self._read_variables:
                raise ValueError("coordination GraphQL read is not bound to target")
        elif document == _UPDATE_REF_MUTATION:
            self._require_mutation(variables)
        else:
            raise ValueError("coordination GraphQL operation is not allowed")
        request = urllib.request.Request(
            _GRAPHQL_URL,
            data=data,
            method="POST",
            headers={
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "trade-rl-disposable-coordination-cas",
            },
        )
        response = self._request(request)
        if not response or len(response) > _MAX_RESPONSE_BYTES:
            raise ValueError("GitHub GraphQL response is empty or oversized")
        try:
            decoded = json.loads(
                response.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("GitHub GraphQL response is not strict JSON") from None
        if not isinstance(decoded, dict):
            raise ValueError("GitHub GraphQL response must be an object")
        if document == _READ_REF_QUERY:
            self._require_readback(decoded)
        return decoded


class GitHubGitObjectHttpApi:
    """Disposable-only, bounded, GET-only Git object transport.

    Never reuse the generic branch-hygiene client here: urllib's default
    redirect handler forwards Authorization on redirected GET requests.
    """

    def __init__(self, *, repository: str, token: str) -> None:
        from tools.github_coordination_ref_cas import _repository

        _, owner, name = _repository(repository)
        if (
            not owner
            or not name
            or not name.casefold().endswith("-coordination-disposable")
        ):
            raise ValueError("REST reader requires a disposable repository")
        if (
            not isinstance(token, str)
            or not token
            or any(character.isspace() for character in token)
        ):
            raise ValueError("GitHub token must be a non-empty single-line secret")
        self._repository = repository
        self._token = token
        self._url = (
            f"{_API_URL}/repos/"
            f"{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(name, safe='')}"
        )

    def _request_json(
        self,
        method: str,
        suffix: str,
        *,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if (
            method != "GET"
            or not isinstance(suffix, str)
            or not _GIT_READ_SUFFIX.fullmatch(suffix)
        ):
            raise ValueError("REST reader permits only GET exact commit/tree/blob OIDs")
        is_tree = suffix.startswith("git/trees/")
        if (is_tree and query != {"recursive": "1"}) or (
            not is_tree and query is not None
        ):
            raise ValueError("REST reader requires exact recursive-tree query only")
        url = f"{self._url}/{suffix}"
        if is_tree:
            url += "?recursive=1"
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "trade-rl-disposable-coordination-cas",
            },
        )
        try:
            opener = urllib.request.build_opener(_NoRedirect())
            with opener.open(request, timeout=30) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RuntimeError(
                "GitHub REST transport failed; read outcome unknown"
            ) from None
        if not raw or len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("GitHub REST response is empty or oversized")
        try:
            decoded = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("GitHub REST response is not strict JSON") from None
        if not isinstance(decoded, dict):
            raise ValueError("GitHub REST response must be an object")
        return decoded


def prepare_disposable_registry_plan(
    objects: RegistryObjectReader,
    *,
    repository: str,
    expected_oid: str,
    after_oid: str,
) -> RegistryTransitionPlan:
    """Derive one nondeleting Task transition from two canonical Git objects.

    The caller must first have passed the test-only target/mutation guard.
    Object provenance is the supplied reader's responsibility; production
    callers use GitHubRegistryObjectReader over fixed GitHub HTTPS GETs.
    """
    from tools.agent_repo.coordination.registry import (
        CoordinatorRegistry,
        parse_registry_bytes,
        plan_registry_transition,
    )
    from tools.github_coordination_ref_cas import ZERO_OID

    before = (
        CoordinatorRegistry(repository)
        if expected_oid == ZERO_OID
        else parse_registry_bytes(objects.read_registry_bytes(repository, expected_oid))
    )
    after = parse_registry_bytes(objects.read_registry_bytes(repository, after_oid))
    if before.repository != repository or after.repository != repository:
        raise ValueError("registry repository differs from the disposable target")
    prior = {entry.status.task_id: entry for entry in before.entries}
    later = {entry.status.task_id: entry for entry in after.entries}
    if prior.keys() - later.keys():
        raise ValueError("registry CAS test must never remove Task enrollment")
    modified = [entry for key, entry in later.items() if prior.get(key) != entry]
    if len(modified) != 1:
        raise ValueError("registry CAS test requires exactly one Task transition")
    plan = plan_registry_transition(
        before, modified[0], expected_before_digest=before.digest()
    )
    if plan.after_registry != after or plan.after_digest != after.digest():
        raise ValueError("registry CAS test would change other Task entries")
    return plan


def execute_disposable_registry_cas(
    api: GraphQLApi,
    objects: RegistryObjectReader,
    *,
    repository: str,
    ref_name: str,
    expected_oid: str,
    after_oid: str,
    expected_coordinator_login: str,
    allow_unprotected_test_mutation: bool = False,
) -> dict[str, object]:
    """Exactly one guarded CAS attempt; no retry, rollback or cleanup grant."""
    from tools.github_coordination_ref_cas import (
        _validated_test_only_update,
        compare_and_swap_registry_plan,
    )

    # MUST precede *all* reader/network calls, including GETs.
    _validated_test_only_update(
        repository=repository,
        ref_name=ref_name,
        expected_oid=expected_oid,
        after_oid=after_oid,
        expected_coordinator_login=expected_coordinator_login,
        allow_unprotected_test_mutation=allow_unprotected_test_mutation,
    )
    plan = prepare_disposable_registry_plan(
        objects, repository=repository, expected_oid=expected_oid, after_oid=after_oid
    )
    return compare_and_swap_registry_plan(
        api,
        objects,
        plan=plan,
        repository=repository,
        ref_name=ref_name,
        expected_oid=expected_oid,
        after_oid=after_oid,
        expected_coordinator_login=expected_coordinator_login,
        allow_unprotected_test_mutation=allow_unprotected_test_mutation,
    )


def inspect_disposable_interrupted_cas(
    api: GraphQLApi,
    objects: RegistryObjectReader,
    *,
    repository: str,
    ref_name: str,
    expected_oid: str,
    after_oid: str,
    expected_coordinator_login: str,
) -> dict[str, object]:
    """Read-only after-interruption inspection, NEVER proof of own mutation.

    Observing the desired OID cannot identify the writer; even observing the
    original OID does not prove it was never changed and later restored.
    Neither status authorizes retry, rollback, new leases or cleanup.
    """
    from tools.github_coordination_ref_cas import (
        _read_snapshot,
        _validated_test_only_update,
    )

    _validated_test_only_update(
        repository=repository,
        ref_name=ref_name,
        expected_oid=expected_oid,
        after_oid=after_oid,
        expected_coordinator_login=expected_coordinator_login,
        allow_unprotected_test_mutation=True,  # validate target only; no mutation
    )
    before = _read_snapshot(api, repository=repository, ref_name=ref_name)
    if before.viewer_login.casefold() != expected_coordinator_login.casefold():
        raise ValueError(
            "authenticated Coordinator differs during read-only inspection"
        )
    plan = prepare_disposable_registry_plan(
        objects, repository=repository, expected_oid=expected_oid, after_oid=after_oid
    )
    after = _read_snapshot(api, repository=repository, ref_name=ref_name)
    if before != after:
        raise ValueError(
            "disposable ref, actor or repository changed during inspection"
        )
    if after.oid == after_oid:
        state = "AFTER_OBSERVED_UNATTRIBUTED"
    elif after.oid == expected_oid:
        state = "BEFORE_OBSERVED_NOT_PROOF_UNAPPLIED"
    else:
        state = "CONFLICTING_OID_OBSERVED"
    return {
        "schema": "disposable_registry_cas_interruption_observation_v1",
        "repository": repository,
        "ref_name": ref_name,
        "observed_oid": after.oid,
        "expected_oid": expected_oid,
        "after_oid": after_oid,
        "candidate_registry_digest": plan.after_digest,
        "state": state,
        "mutation_attempted": False,
        "writer_attribution_verified": False,
        "transaction_outcome_proven": False,
        "retry_authorized": False,
        "rollback_authorized": False,
        "authoritative_registry_established": False,
        "task_registry_complete": False,
        "cleanup_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Run only in a separately provisioned, explicitly disposable repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--ref-name", default="refs/heads/coordination/task-registry")
    parser.add_argument("--registry-path", default="coordination/registry.json")
    parser.add_argument("--expected-oid", required=True)
    parser.add_argument("--after-oid", required=True)
    parser.add_argument("--expected-coordinator-login", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allow-disposable-mutation", action="store_true")
    mode.add_argument("--inspect-interrupted-mutation", action="store_true")
    args = parser.parse_args(argv)

    from tools.github_coordination_ref_cas import _validated_test_only_update

    try:
        _validated_test_only_update(
            repository=args.repository,
            ref_name=args.ref_name,
            expected_oid=args.expected_oid,
            after_oid=args.after_oid,
            expected_coordinator_login=args.expected_coordinator_login,
            allow_unprotected_test_mutation=True,  # explicit mode gate above
        )
    except ValueError as error:
        parser.error(str(error))

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("GH_TOKEN or GITHUB_TOKEN is required for disposable CAS")

    from tools.github_coordination_registry_reader import GitHubRegistryObjectReader

    reader = GitHubRegistryObjectReader(
        GitHubGitObjectHttpApi(repository=args.repository, token=token),
        expected_repository=args.repository,
        registry_path=args.registry_path,
    )
    graphql = GitHubGraphQLHttpApi(
        token,
        repository=args.repository,
        ref_name=args.ref_name,
        expected_oid=args.expected_oid,
        after_oid=args.after_oid,
        expected_coordinator_login=args.expected_coordinator_login,
        allow_disposable_mutation=args.allow_disposable_mutation,
    )
    if args.inspect_interrupted_mutation:
        result = inspect_disposable_interrupted_cas(
            graphql,
            reader,
            repository=args.repository,
            ref_name=args.ref_name,
            expected_oid=args.expected_oid,
            after_oid=args.after_oid,
            expected_coordinator_login=args.expected_coordinator_login,
        )
    else:
        result = execute_disposable_registry_cas(
            graphql,
            reader,
            repository=args.repository,
            ref_name=args.ref_name,
            expected_oid=args.expected_oid,
            after_oid=args.after_oid,
            expected_coordinator_login=args.expected_coordinator_login,
            allow_unprotected_test_mutation=True,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


__all__ = [
    "GitHubGitObjectHttpApi",
    "GitHubGraphQLHttpApi",
    "execute_disposable_registry_cas",
    "inspect_disposable_interrupted_cas",
    "prepare_disposable_registry_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
