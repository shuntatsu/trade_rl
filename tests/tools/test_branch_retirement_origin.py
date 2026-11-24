"""Retirement must have exactly one trusted effective Git destination."""

import subprocess
from pathlib import Path

import pytest

from tools.branch_hygiene import BranchInfo, GitHubApi, require_retirement_origin

EXPECTED = "https://github.com/shuntatsu/trade_rl.git"
OTHER = "https://github.com/fixture/other.git"


@pytest.fixture
def checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.chdir(repo)
    subprocess.run(["git", "remote", "add", "origin", EXPECTED], check=True)
    return repo


def _config(*args: str) -> None:
    subprocess.run(["git", "config", *args], check=True)


def test_single_canonical_origin_is_accepted(checkout: Path) -> None:
    require_retirement_origin()
    _config("--add", "remote.origin.pushurl", EXPECTED.removesuffix(".git"))
    require_retirement_origin()


@pytest.mark.parametrize("key", ["remote.origin.url", "remote.origin.pushurl"])
@pytest.mark.parametrize("extra", [OTHER, EXPECTED])
def test_multiple_effective_destinations_rejected(
    checkout: Path, key: str, extra: str
) -> None:
    if key == "remote.origin.pushurl":
        _config("--add", key, EXPECTED)
    _config("--add", key, extra)
    with pytest.raises(RuntimeError, match="exact trade_rl"):
        require_retirement_origin()


def test_rewritten_push_destination_rejected(checkout: Path) -> None:
    _config(f"url.{OTHER}.pushInsteadOf", EXPECTED)
    with pytest.raises(RuntimeError, match="exact trade_rl"):
        require_retirement_origin()


@pytest.mark.parametrize("drift", ["multiple", "foreign"])
def test_transport_rechecks_origin_before_any_push(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    require_retirement_origin()
    if drift == "multiple":
        _config("--add", "remote.origin.pushurl", EXPECTED)
    _config("--add", "remote.origin.pushurl", OTHER)
    original_run = subprocess.run
    attempted: list[list[str]] = []

    def observe(command: list[str] | tuple[str, ...], **kwargs: object):
        if tuple(command[:2]) == ("git", "push"):
            attempted.append(list(command))
            return subprocess.CompletedProcess(command, 0, "", "")
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", observe)
    api = GitHubApi(
        repository="shuntatsu/trade_rl",
        token="synthetic",
        api_url="https://api.github.com",
    )
    with pytest.raises(RuntimeError, match="exact trade_rl"):
        api.delete_branches_with_leases((BranchInfo("fix/synthetic", "1" * 40, False),))
    assert attempted == []


def test_transport_checks_same_environment_as_push(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(f"url.{OTHER}.pushInsteadOf", EXPECTED)
    # The inherited override hides the pushInsteadOf rule; the transport
    # replaces GIT_CONFIG_COUNT with its auth header before executing Git.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "remote.origin.pushurl")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", EXPECTED)
    require_retirement_origin()
    original_run = subprocess.run
    attempted: list[list[str]] = []

    def observe(command: list[str] | tuple[str, ...], **kwargs: object):
        if tuple(command[:2]) == ("git", "push"):
            attempted.append(list(command))
            return subprocess.CompletedProcess(command, 0, "", "")
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", observe)
    api = GitHubApi(
        repository="shuntatsu/trade_rl",
        token="synthetic",
        api_url="https://api.github.com",
    )
    with pytest.raises(RuntimeError, match="exact trade_rl"):
        api.delete_branches_with_leases((BranchInfo("fix/synthetic", "1" * 40, False),))
    assert attempted == []


def test_foreign_repository_rejected_without_git_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_io(*args: object, **kwargs: object):
        pytest.fail("unexpected Git I/O for a foreign repository")

    monkeypatch.setattr(subprocess, "run", reject_io)
    api = GitHubApi(
        repository="fixture/other", token="fake", api_url="https://api.github.com"
    )
    with pytest.raises(ValueError, match="restricted to trade_rl"):
        api.delete_branches_with_leases((BranchInfo("fix/synthetic", "1" * 40, False),))
