from pathlib import Path

import pytest
from fastapi import FastAPI

from scripts.run_server import build_app_from_env


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    values = {
        "TRADE_RL_SERVING_TOKEN": "secret",
        "TRADE_RL_RELEASE_GIT_SHA": "a" * 40,
        "TRADE_RL_REGISTRY_DIR": str(tmp_path / "registry"),
        "TRADE_RL_AUDIT_DB": str(tmp_path / "audit.sqlite3"),
        "TRADE_RL_DATA_DIR": str(tmp_path / "data"),
    }
    values.update(overrides)
    return values


def test_server_requires_release_git_sha(tmp_path: Path) -> None:
    env = _env(tmp_path)
    del env["TRADE_RL_RELEASE_GIT_SHA"]

    with pytest.raises(RuntimeError, match="TRADE_RL_RELEASE_GIT_SHA"):
        build_app_from_env(env)


def test_server_rejects_invalid_release_git_sha(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        build_app_from_env(
            _env(tmp_path, TRADE_RL_RELEASE_GIT_SHA="not-a-commit")
        )


def test_server_builds_with_strict_release_identity(tmp_path: Path) -> None:
    app = build_app_from_env(_env(tmp_path))

    assert isinstance(app, FastAPI)
