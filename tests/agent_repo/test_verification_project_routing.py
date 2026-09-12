from __future__ import annotations

import subprocess
from pathlib import Path

from tools.agent_repo.verification import VerificationStep, plan_verification


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _baseline(root: Path) -> None:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    _write(root, "trade_rl/__init__.py", "")
    _write(root, "pyproject.toml", "[project]\nname='example'\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "baseline")


def _extended(steps: tuple[VerificationStep, ...]) -> tuple[VerificationStep, ...]:
    return tuple(step for step in steps if step.tier == "extended")


def test_optional_extra_change_routes_optional_capability_smoke(tmp_path: Path) -> None:
    _baseline(tmp_path)
    _write(
        tmp_path,
        "pyproject.toml",
        "[project]\n"
        "name='example'\n\n"
        "[project.optional-dependencies]\n"
        "forecast-gbm=['lightgbm==4.7.0']\n",
    )

    extended = _extended(plan_verification(tmp_path, base_ref="main"))

    assert any("optional capability extras" in step.command for step in extended)
    assert all("project script" not in step.command for step in extended)


def test_project_script_change_routes_script_smoke_without_optional_extra(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    _write(
        tmp_path,
        "pyproject.toml",
        "[project]\n"
        "name='example'\n\n"
        "[project.scripts]\n"
        "trade-rl-check='trade_rl.cli:main'\n",
    )

    extended = _extended(plan_verification(tmp_path, base_ref="main"))

    assert any("project script" in step.command for step in extended)
    assert all("optional capability extras" not in step.command for step in extended)


def test_unrelated_project_metadata_change_does_not_route_optional_environment_smoke(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    _write(
        tmp_path,
        "pyproject.toml",
        "[project]\n"
        "name='example'\n"
        "description='metadata only'\n",
    )

    extended = _extended(plan_verification(tmp_path, base_ref="main"))

    assert all("optional capability extras" not in step.command for step in extended)
    assert all("project script" not in step.command for step in extended)
