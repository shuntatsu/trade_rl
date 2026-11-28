"""PR repository identity must distinguish unavailable from truncated metadata."""

from __future__ import annotations

import pytest

from tools.github_work_inventory import _pull_identity

_REPOSITORY = "shuntatsu/trade_rl"


def _pr(head_repository_fields: dict[str, object]) -> dict[str, object]:
    return {
        "number": 123,
        "head": {
            "ref": "feature/current",
            "sha": "d" * 40,
            **head_repository_fields,
        },
        "base": {
            "ref": "main",
            "sha": "a" * 40,
            "repo": {"full_name": _REPOSITORY},
        },
    }


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({}, id="missing-repo-key"),
        pytest.param({"repo": {}}, id="missing-full-name"),
        pytest.param({"repo": {"full_name": None}}, id="null-full-name"),
        pytest.param({"repo": {"full_name": ""}}, id="empty-full-name"),
        pytest.param({"repo": {"full_name": " \t"}}, id="blank-full-name"),
    ],
)
def test_incomplete_head_repository_is_rejected(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="PR head repository"):
        _pull_identity(_pr(fields), _REPOSITORY)


@pytest.mark.parametrize(
    ("repo", "expected_ref"),
    [
        pytest.param(None, None, id="explicitly-unavailable"),
        pytest.param({"full_name": "other/fork"}, None, id="fork"),
        pytest.param(
            {"full_name": "Shuntatsu/Trade_RL"}, "feature/current", id="local"
        ),
    ],
)
def test_complete_head_repository_identity_is_preserved(
    repo: dict[str, str] | None, expected_ref: str | None
) -> None:
    assert _pull_identity(_pr({"repo": repo}), _REPOSITORY) == (
        123,
        "d" * 40,
        "main",
        "a" * 40,
        expected_ref,
    )


@pytest.mark.parametrize(
    "full_name",
    (
        " shuntatsu/trade_rl",
        "shuntatsu/trade_rl ",
        "shuntatsu//trade_rl",
        "shuntatsu/trade_rl/extra",
        "shuntatsu/",
        "/trade_rl",
        "shuntatsu",
        "\u017fhuntatsu/trade_rl",
        "shuntatsu/trade_RL\t",
        "shuntatsu/../trade_rl",
        "shuntatsu/\ud800",
    ),
)
def test_malformed_head_repository_name_fails_closed(full_name: str) -> None:
    with pytest.raises(ValueError, match="PR head repository"):
        _pull_identity(_pr({"repo": {"full_name": full_name}}), _REPOSITORY)


@pytest.mark.parametrize(
    "full_name",
    (
        "\u017fhuntatsu/trade_rl",
        "\u017fHUNTATSU/TRADE_RL",
        " shuntatsu/trade_rl",
        "shuntatsu/trade_rl ",
    ),
)
def test_malformed_base_repository_name_fails_closed(full_name: str) -> None:
    pull = _pr({"repo": {"full_name": _REPOSITORY}})
    base = pull["base"]
    assert isinstance(base, dict)
    base["repo"] = {"full_name": full_name}
    with pytest.raises(ValueError, match="PR base"):
        _pull_identity(pull, _REPOSITORY)


@pytest.mark.parametrize(
    ("full_name", "expected_head"),
    (
        ("SHUNTATSU/TRADE_RL", "feature/current"),
        ("organization-with-hyphen/repo.with.dot", None),
        ("organization/repo_a-b", None),
    ),
)
def test_native_ascii_repository_names_keep_local_and_fork_contract(
    full_name: str, expected_head: str | None
) -> None:
    assert _pull_identity(
        _pr({"repo": {"full_name": full_name}}), _REPOSITORY
    )[-1] == expected_head
