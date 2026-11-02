"""Token-safe, bounded GraphQL HTTP client oracles (no network)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from tools.github_coordination_disposable_e2e import (
    GitHubGitObjectHttpApi,
    GitHubGraphQLHttpApi,
    _NoRedirect,
    main,
)


def _stub_post(
    api: GitHubGraphQLHttpApi, monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> list[urllib.request.Request]:
    observed: list[urllib.request.Request] = []

    def fake_request(request: urllib.request.Request) -> bytes:
        observed.append(request)
        return raw

    monkeypatch.setattr(api, "_request", fake_request)
    return observed


def test_real_http_request_is_fixed_host_post_bearer_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubGraphQLHttpApi("secret-never-log")
    requests = _stub_post(api, monkeypatch, b'{"data":{"viewer":{"login":"bot"}}}')
    assert api.execute("query X { viewer { login } }", {"x": 1}) == {
        "data": {"viewer": {"login": "bot"}},
    }
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == "https://api.github.com/graphql"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer secret-never-log"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data or b"") == {
        "query": "query X { viewer { login } }",
        "variables": {"x": 1},
    }
    assert "secret-never-log" not in repr(api)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (b"", "empty or oversized"),
        (b"\xff", "not strict JSON"),
        (b"not-json", "not strict JSON"),
        (b"[]", "must be an object"),
        (b'{"data":{},"data":{}}', "duplicate JSON fields"),
        (b'{"data":{"viewer":{},"viewer":{}}}', "duplicate JSON fields"),
        (b'{"data":{"value":NaN}}', "non-finite JSON constants"),
        (b'{"data":{"value":Infinity}}', "non-finite JSON constants"),
        (b"x" * (1024 * 1024 + 1), "empty or oversized"),
    ],
)
def test_graphql_transport_rejects_untrusted_response(
    monkeypatch: pytest.MonkeyPatch, response: bytes, message: str
) -> None:
    api = GitHubGraphQLHttpApi("secret-do-not-print")
    _stub_post(api, monkeypatch, response)
    with pytest.raises(ValueError, match=message) as error:
        api.execute("query X { viewer { login } }", {})
    assert "secret-do-not-print" not in str(error.value)


def test_graphql_transport_rejects_arbitrary_mutation_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubGraphQLHttpApi("safe-token")
    observed = _stub_post(api, monkeypatch, b'{"data":{}}')
    with pytest.raises(ValueError, match="coordination GraphQL operation"):
        api.execute(
            "mutation Arbitrary($input: UpdateRefsInput!) { updateRefs(input: $input) { clientMutationId } }",
            {"input": {"repositoryId": "R_production", "refUpdates": []}},
        )
    assert observed == []


def test_graphql_transport_never_leaks_token_in_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubGraphQLHttpApi("secret-must-not-leak")

    class FailingOpener:
        def open(self, request: object, *, timeout: int) -> Any:
            raise urllib.error.URLError("secret-must-not-leak")

    def fake_builder(*handlers: object) -> FailingOpener:
        assert len(handlers) == 1  # no redirect handler
        return FailingOpener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_builder)
    with pytest.raises(RuntimeError, match="mutation outcome may be unknown") as error:
        api.execute("mutation X { fake }", {})
    assert "secret-must-not-leak" not in str(error.value)


def test_missing_explicit_cli_mode_fails_before_any_network() -> None:
    arguments = [
        "--repository",
        "shuntatsu/trade-rl-coordination-disposable",
        "--expected-oid",
        "a" * 40,
        "--after-oid",
        "b" * 40,
        "--expected-coordinator-login",
        "coordinator-bot",
    ]
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2


def test_http_client_rejects_empty_and_multiline_tokens() -> None:
    for token in ("", "abc\ndef", "abc def"):
        with pytest.raises(ValueError, match="non-empty single-line"):
            GitHubGraphQLHttpApi(token)


def test_http_client_rejects_oversized_request_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubGraphQLHttpApi("safe-token")
    observed = _stub_post(api, monkeypatch, b'{"data":{}}')
    with pytest.raises(ValueError, match="bounded payload"):
        api.execute("query " + "x" * 65536, {})
    assert observed == []


def test_graphql_rejects_nonfinite_request_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubGraphQLHttpApi("safe-token")
    observed = _stub_post(api, monkeypatch, b'{"data":{}}')
    with pytest.raises(ValueError, match="Out of range float values"):
        api.execute("query X { viewer { login } }", {"invalid": float("nan")})
    assert observed == []


REPOSITORY = "shuntatsu/trade-rl-coordination-disposable"
COMMIT_OID = "a" * 40
TREE_OID = "b" * 40


def test_default_python_redirect_copies_auth_but_guard_blocks_it() -> None:
    request = urllib.request.Request(
        "https://api.github.com/repos/a/b",
        headers={"Authorization": "Bearer simulated-secret"},
    )
    url = "https://untrusted.example/steal"
    # Regression: standard urllib handler *does* copy the bearer credential.
    redirected = urllib.request.HTTPRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, url
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") == "Bearer simulated-secret"
    # The test-only transports must explicitly suppress that redirect path.
    assert _NoRedirect().redirect_request(request, None, 302, "Found", {}, url) is None


def test_rest_git_reader_uses_fixed_host_get_and_no_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubGitObjectHttpApi(repository=REPOSITORY, token="secret-never-log")
    seen: list[urllib.request.Request] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            assert size == 1024 * 1024 + 1
            return b'{"sha":"' + COMMIT_OID.encode() + b'"}'

    class Opener:
        def open(self, request: urllib.request.Request, *, timeout: int) -> Response:
            assert timeout == 30
            seen.append(request)
            return Response()

    def build(*handlers: object) -> Opener:
        assert len(handlers) == 1 and isinstance(handlers[0], _NoRedirect)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build)
    assert api._request_json("GET", f"git/commits/{COMMIT_OID}") == {"sha": COMMIT_OID}
    assert api._request_json(
        "GET", f"git/trees/{TREE_OID}", query={"recursive": "1"}
    ) == {"sha": COMMIT_OID}
    assert [req.full_url for req in seen] == [
        f"https://api.github.com/repos/{REPOSITORY}/git/commits/{COMMIT_OID}",
        f"https://api.github.com/repos/{REPOSITORY}/git/trees/{TREE_OID}?recursive=1",
    ]
    assert all(req.get_method() == "GET" for req in seen)
    assert all(
        req.get_header("Authorization") == "Bearer secret-never-log" for req in seen
    )
    assert "secret-never-log" not in repr(api)


@pytest.mark.parametrize(
    ("method", "suffix", "query"),
    [
        ("POST", f"git/commits/{COMMIT_OID}", None),
        ("GET", "branches", None),
        ("GET", "git/commits/../main", None),
        ("GET", f"git/blobs/{COMMIT_OID}?evil=true", None),
        ("GET", f"git/commits/{COMMIT_OID}", {"recursive": "1"}),
        ("GET", f"git/trees/{TREE_OID}", None),
        ("GET", f"git/trees/{TREE_OID}", {"recursive": "0"}),
    ],
)
def test_rest_read_rejects_unapproved_method_path_or_query_before_io(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    suffix: str,
    query: dict[str, str] | None,
) -> None:
    api = GitHubGitObjectHttpApi(repository=REPOSITORY, token="test-token")

    def forbid(*_args: object) -> None:
        raise AssertionError("unauthorized HTTP call")

    monkeypatch.setattr(urllib.request, "build_opener", forbid)
    with pytest.raises(ValueError, match="REST reader"):
        api._request_json(method, suffix, query=query)


def test_rest_redirect_and_errors_never_reveal_bearer_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubGitObjectHttpApi(repository=REPOSITORY, token="secret-never-log")

    class FailingOpener:
        def open(self, request: urllib.request.Request, *, timeout: int) -> None:
            raise urllib.error.HTTPError(
                request.full_url, 302, "redirect with secret-never-log", {}, None
            )

    def builder(*handlers: object) -> FailingOpener:
        assert len(handlers) == 1 and isinstance(handlers[0], _NoRedirect)
        return FailingOpener()

    monkeypatch.setattr(urllib.request, "build_opener", builder)
    with pytest.raises(RuntimeError, match="REST transport failed") as exc:
        api._request_json("GET", f"git/commits/{COMMIT_OID}")
    assert "secret-never-log" not in str(exc.value)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (b"", "empty or oversized"),
        (b"x" * (1024 * 1024 + 1), "empty or oversized"),
        (b"not-json", "not strict JSON"),
        (b'{"sha":"a","sha":"b"}', "duplicate JSON fields"),
        (b'{"sha":"a","noise":-Infinity}', "non-finite JSON constants"),
        (b"[]", "must be an object"),
    ],
)
def test_rest_reader_rejects_malformed_untrusted_json(
    monkeypatch: pytest.MonkeyPatch, response: bytes, message: str
) -> None:
    api = GitHubGitObjectHttpApi(repository=REPOSITORY, token="test-token")

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return response

    class FakeOpener:
        def open(self, _req: urllib.request.Request, *, timeout: int) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: FakeOpener())
    with pytest.raises(ValueError, match=message):
        api._request_json("GET", f"git/commits/{COMMIT_OID}")


@pytest.mark.parametrize(
    ("repository", "message"),
    [
        ("shuntatsu/trade_rl", "disposable"),
        ("shuntatsu/not-disposable", "disposable"),
        ("bad", "repository"),
    ],
)
def test_rest_disposable_repository_guard_is_independent_of_cli(
    repository: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GitHubGitObjectHttpApi(repository=repository, token="test-token")
