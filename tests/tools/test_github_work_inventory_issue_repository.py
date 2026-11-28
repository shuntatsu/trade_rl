"""Issue repository URLs must not admit non-ASCII casefold aliases."""

from __future__ import annotations

import pytest

from tools.github_work_inventory import _issue_identity


def _issue(repository_url: str) -> dict[str, object]:
    return {
        "number": 1057,
        "id": 123456,
        "node_id": "I_TEST_1057",
        "repository_url": repository_url,
        "state": "open",
        "comments": 0,
        "updated_at": "2026-11-28T00:00:00Z",
    }


@pytest.mark.parametrize(
    ("repository", "repository_url"),
    [
        ("owner/repo", "http\u017f://api.github.com/repos/owner/repo"),
        ("owner/repo", "https://api.github.com/repo\u017f/owner/repo"),
        ("shuntatsu/trade_rl", "https://api.github.com/repos/\u017fhuntatsu/trade_rl"),
        ("owner/k", "https://api.github.com/repos/owner/\u212a"),
        ("owner/ffi", "https://api.github.com/repos/owner/\ufb03"),
        ("owner/repo", "https://api.github.com/repos/other/repo"),
        ("owner/repo", "https://api.github.com/repos/owner/repo/"),
    ],
)
def test_issue_repository_url_rejects_invalid_identity(
    repository: str, repository_url: str
) -> None:
    with pytest.raises(ValueError, match="open Issue repository identity differs"):
        _issue_identity(_issue(repository_url), repository)


@pytest.mark.parametrize(
    ("repository", "repository_url"),
    [
        ("shuntatsu/trade_rl", "https://api.github.com/repos/shuntatsu/trade_rl"),
        ("owner/repo", "HTTPS://API.GITHUB.COM/REPOS/OWNER/REPO"),
        ("owner-name/repo.name", "https://api.github.com/repos/Owner-Name/Repo.Name"),
    ],
)
def test_issue_repository_url_accepts_native_ascii_identity(
    repository: str, repository_url: str
) -> None:
    assert _issue_identity(_issue(repository_url), repository) == (
        1057,
        123456,
        "I_TEST_1057",
        "open",
        0,
        "2026-11-28T00:00:00Z",
    )
