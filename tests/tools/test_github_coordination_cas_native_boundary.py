"""Native identity admission for the disposable CAS and enrollment adapters."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from tests.tools import test_github_coordination_enrollment_audit as audit_tests
from tests.tools import (
    test_github_coordination_enrollment_transaction as transaction_tests,
)
from tests.tools import test_github_coordination_ref_cas as cas_tests
from tests.tools import test_github_coordination_registry_cas_payload as payload_tests
from tools.agent_repo.coordination.registry import (
    CoordinatorRegistry,
    RegistryTransitionPlan,
)
from tools.github_coordination_disposable_e2e import (
    GitHubEnrollmentReadHttpApi,
    GitHubGitObjectHttpApi,
    GitHubGraphQLHttpApi,
)


class _Text(str):
    pass


class _TextTrap(str):
    def __eq__(self, other: object) -> bool:
        raise AssertionError("foreign equality invoked")

    def __ne__(self, other: object) -> bool:
        raise AssertionError("foreign inequality invoked")

    __hash__ = str.__hash__

    def casefold(self) -> str:
        raise AssertionError("foreign casefold invoked")

    def strip(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("foreign strip invoked")

    def split(self, *args: Any, **kwargs: Any) -> list[str]:
        raise AssertionError("foreign split invoked")

    def startswith(self, *args: Any, **kwargs: Any) -> bool:
        raise AssertionError("foreign prefix invoked")


class _ObjectTrap(dict):
    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("foreign lookup invoked")

    def __iter__(self):
        raise AssertionError("foreign mapping iteration invoked")


class _KeyTrap(str):
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        raise AssertionError("foreign key equality invoked")


class _ErrorRows(list):
    def __eq__(self, other: object) -> bool:
        raise AssertionError("foreign errors comparison invoked")

    def __ne__(self, other: object) -> bool:
        raise AssertionError("foreign errors comparison invoked")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", cas_tests.REPO),
        ("ref_name", cas_tests.REF),
        ("expected_oid", cas_tests.A),
        ("after_oid", cas_tests.B),
        ("expected_coordinator_login", "coordinator-bot"),
    ],
)
@pytest.mark.parametrize("factory", [_Text, _TextTrap])
def test_foreign_cas_identity_rejected_before_transport(field, value, factory) -> None:
    api = cas_tests.FakeGraphQL()
    with pytest.raises(ValueError):
        cas_tests._cas(api, **{field: factory(value)})
    assert api.calls == []


@pytest.mark.parametrize(
    ("path", "factory"),
    [
        ((), _ObjectTrap),
        (("data",), _ObjectTrap),
        (("data", "viewer"), _ObjectTrap),
        (("data", "repository"), _ObjectTrap),
        (("data", "repository", "ref"), _ObjectTrap),
        (("data", "repository", "ref", "target"), _ObjectTrap),
        (("data", "viewer", "login"), _TextTrap),
        (("data", "repository", "id"), _TextTrap),
        (("data", "repository", "ref", "target", "oid"), _TextTrap),
    ],
)
def test_foreign_graphql_read_refused_before_mutation(path, factory) -> None:
    class Api(cas_tests.FakeGraphQL):
        def execute(self, document, variables):
            row = super().execute(document, variables)
            if not path:
                return factory(row)
            cursor = row
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = factory(cursor[path[-1]])
            return row

    api = Api()
    with pytest.raises(ValueError):
        cas_tests._cas(api)
    assert api.reads == 1
    assert api.oid == cas_tests.A
    assert len(api.calls) == 1


def test_foreign_graphql_key_refused_before_lookup() -> None:
    class Api(cas_tests.FakeGraphQL):
        def execute(self, document, variables):
            row = super().execute(document, variables)
            return {_KeyTrap("data"): row["data"]}

    api = Api()
    with pytest.raises(ValueError, match="native string keys"):
        cas_tests._cas(api)
    assert api.oid == cas_tests.A
    assert len(api.calls) == 1


def test_foreign_graphql_error_collection_refused_without_comparison() -> None:
    class Api(cas_tests.FakeGraphQL):
        def execute(self, document, variables):
            return {**super().execute(document, variables), "errors": _ErrorRows()}

    api = Api()
    with pytest.raises(RuntimeError, match="GraphQL errors"):
        cas_tests._cas(api)
    assert api.oid == cas_tests.A
    assert len(api.calls) == 1


def test_foreign_graphql_mutation_id_is_not_successful_readback() -> None:
    class Api(cas_tests.FakeGraphQL):
        def execute(self, document, variables):
            row = super().execute(document, variables)
            if "mutation UpdateCoordinationRef" in document:
                update = row["data"]["updateRefs"]
                update["clientMutationId"] = _TextTrap(update["clientMutationId"])
            return row

    api = Api()
    with pytest.raises(ValueError, match="clientMutationId"):
        cas_tests._cas(api)
    # A failed readback cannot be called a rollback or retried.
    assert api.oid == cas_tests.B
    assert len(api.calls) == 2


def test_foreign_plan_rejected_before_object_reads() -> None:
    plan, api, objects = payload_tests._fixture()

    class ForeignPlan(RegistryTransitionPlan):
        pass

    with pytest.raises(ValueError, match="RegistryTransitionPlan"):
        payload_tests._cas(
            ForeignPlan(
                operation=plan.operation,
                task_id=plan.task_id,
                before_digest=plan.before_digest,
                after_digest=plan.after_digest,
                after_registry=plan.after_registry,
            ),
            api,
            objects,
        )
    assert not objects.calls
    assert not api.calls


@pytest.mark.parametrize(
    "field", ["operation", "task_id", "before_digest", "after_digest"]
)
def test_foreign_plan_identity_refused_before_object_reads(field) -> None:
    plan, api, objects = payload_tests._fixture()
    plan = replace(plan, **{field: _TextTrap(getattr(plan, field))})
    with pytest.raises(ValueError, match="fields must be native"):
        payload_tests._cas(plan, api, objects)
    assert not objects.calls
    assert not api.calls


def test_foreign_plan_registry_refused_before_object_reads() -> None:
    plan, api, objects = payload_tests._fixture()

    class ForeignRegistry(CoordinatorRegistry):
        pass

    registry = ForeignRegistry(
        plan.after_registry.repository, plan.after_registry.entries
    )
    with pytest.raises(ValueError, match="fields must be native"):
        payload_tests._cas(replace(plan, after_registry=registry), api, objects)
    assert not objects.calls
    assert not api.calls


@pytest.mark.parametrize(
    ("suffix", "field", "factory"),
    [
        ("", None, _ObjectTrap),
        ("", "full_name", _TextTrap),
        ("", "default_branch", _TextTrap),
        (f"issues/{audit_tests.ISSUE}", None, _ObjectTrap),
        (f"issues/{audit_tests.ISSUE}", "state", _TextTrap),
        (f"issues/{audit_tests.ISSUE}", "repository_url", _TextTrap),
        ("git/ref/heads/main", None, _ObjectTrap),
        ("git/ref/heads/main", "ref", _TextTrap),
        ("git/ref/heads/main", "object", _ObjectTrap),
    ],
)
def test_foreign_enrollment_source_is_rejected_before_registry_plan(
    suffix, field, factory
) -> None:
    class Api(audit_tests.FakeGitHub):
        def _request_json(self, method, path, *, query=None):
            row = super()._request_json(method, path, query=query)
            if path == suffix:
                if field is None:
                    return factory(row)
                row[field] = factory(row[field])
            return row

    api = Api()
    with pytest.raises(ValueError):
        audit_tests.audit(api)
    assert not any(path.startswith("git/commits/") for _, path in api.calls)


@pytest.mark.parametrize("field", ["type", "sha"])
def test_foreign_ref_target_identity_refused_before_registry_plan(field) -> None:
    class Api(audit_tests.FakeGitHub):
        def _request_json(self, method, path, *, query=None):
            row = super()._request_json(method, path, query=query)
            if path == "git/ref/heads/main":
                row["object"][field] = _TextTrap(row["object"][field])
            return row

    api = Api()
    with pytest.raises(ValueError, match="exact nonzero commit"):
        audit_tests.audit(api)
    assert not any(path.startswith("git/commits/") for _, path in api.calls)


def test_foreign_source_key_refused_before_lookup() -> None:
    class Api(audit_tests.FakeGitHub):
        def _request_json(self, method, path, *, query=None):
            row = super()._request_json(method, path, query=query)
            if path == "":
                return {_KeyTrap(key): value for key, value in row.items()}
            return row

    api = Api()
    with pytest.raises(ValueError, match="native keys"):
        audit_tests.audit(api)
    assert len(api.calls) == 1


def test_foreign_expected_issue_id_rejected_before_transaction_io() -> None:
    graphql, rest, objects = transaction_tests.fixture()
    with pytest.raises(ValueError, match="Issue node"):
        transaction_tests.run(
            graphql, rest, objects, expected_issue_node_id=_TextTrap("I_issue")
        )
    assert not graphql.calls
    assert not rest.calls


@pytest.mark.parametrize(
    "reader_type", [GitHubGitObjectHttpApi, GitHubEnrollmentReadHttpApi]
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("method", _Text("GET")),
        ("suffix", _Text("git/trees/" + cas_tests.A)),
        ("query", _ObjectTrap({"recursive": "1"})),
        ("query", {"recursive": _Text("1")}),
        ("query", {_KeyTrap("recursive"): "1"}),
    ],
)
def test_rest_foreign_request_rejected_before_http(
    monkeypatch, reader_type, field, value
) -> None:
    api = reader_type(repository=cas_tests.REPO, token="test-token")

    def forbid(*_args, **_kwargs):
        raise AssertionError("unexpected HTTP request")

    monkeypatch.setattr(api, "_get_json", forbid)
    fields = {
        "method": "GET",
        "suffix": "git/trees/" + cas_tests.A,
        "query": {"recursive": "1"},
    }
    fields[field] = value
    with pytest.raises(ValueError):
        api._request_json(**fields)


def test_foreign_bound_transport_repository_rejected_during_construction() -> None:
    with pytest.raises(ValueError, match="repository"):
        GitHubGraphQLHttpApi(
            "test-token",
            repository=_Text(cas_tests.REPO),
            ref_name=cas_tests.REF,
            expected_oid=cas_tests.A,
            after_oid=cas_tests.B,
            expected_coordinator_login="coordinator-bot",
        )
    with pytest.raises(ValueError, match="repository"):
        GitHubGitObjectHttpApi(repository=_Text(cas_tests.REPO), token="test-token")
