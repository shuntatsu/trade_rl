from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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


def _repository(root: Path) -> None:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    _write(root, "trade_rl/__init__.py", "")
    _write(root, "trade_rl/evaluation/__init__.py", "")
    _write(
        root,
        "trade_rl/evaluation/runs/__init__.py",
        "from .config import CandidateRunConfig\n"
        "__all__ = ['CandidateRunConfig']\n",
    )
    _write(
        root,
        "trade_rl/evaluation/runs/config.py",
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class CandidateRunConfig:\n"
        "    value: int\n",
    )
    _write(root, ".github/workflows/ci.yml", "name: CI\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "baseline")
    _git(root, "switch", "-c", "feature")


def _run(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not current else f"{ROOT}{os.pathsep}{current}"
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", *args],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_commands_emit_json_without_mutation(tmp_path: Path) -> None:
    _repository(tmp_path)
    before = _git(tmp_path, "status", "--porcelain=v1")

    preflight = _run(tmp_path, "preflight", "--base", "main")
    context = _run(tmp_path, "context", "trade_rl/evaluation/runs/config.py")
    impact = _run(
        tmp_path,
        "impact",
        "trade_rl/evaluation/runs/config.py",
        "trade_rl/evaluation/runs/__init__.py",
    )
    diff = _run(tmp_path, "diff", "--base", "main")
    verify = _run(tmp_path, "verify", "--base", "main")

    assert preflight.returncode == 0, preflight.stderr
    assert context.returncode == 0, context.stderr
    assert impact.returncode == 0, impact.stderr
    assert diff.returncode == 0, diff.stderr
    assert verify.returncode == 0, verify.stderr
    assert json.loads(preflight.stdout)["branch"] == "feature"
    assert json.loads(context.stdout)["capability"] == "runs"
    contexts = json.loads(impact.stdout)["contexts"]
    assert [item["path"] for item in contexts] == [
        "trade_rl/evaluation/runs/__init__.py",
        "trade_rl/evaluation/runs/config.py",
    ]
    assert json.loads(diff.stdout) == {"signals": []}
    verification = json.loads(verify.stdout)["steps"]
    assert any(item["tier"] == "final" for item in verification)
    assert _git(tmp_path, "status", "--porcelain=v1") == before


def test_cli_returns_nonzero_for_invalid_context_path(tmp_path: Path) -> None:
    _repository(tmp_path)

    result = _run(tmp_path, "context", "trade_rl/missing.py")

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr
