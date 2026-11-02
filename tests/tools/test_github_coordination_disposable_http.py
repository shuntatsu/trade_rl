"""Token-safe, bounded GraphQL HTTP client oracles (no network)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from tools.github_coordination_disposable_e2e import GitHubGraphQLHttpApi, main


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
