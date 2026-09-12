from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIMENSIONS = (
    "authority_discovery",
    "authority_reuse",
    "boundary_compliance",
    "scope_discipline",
    "test_discovery",
    "verification_selection",
    "compatibility_awareness",
    "context_efficiency",
)


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
        "from .config import CandidateRunConfig\n__all__ = ['CandidateRunConfig']\n",
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


def test_eval_cli_lists_shows_and_scores_without_persisting_output(
    tmp_path: Path,
) -> None:
    _repository(tmp_path)
    score_path = tmp_path / "scores.json"
    score_path.write_text(
        json.dumps(
            {
                dimension: [2, f"evidence for {dimension}"]
                for dimension in EVAL_DIMENSIONS
            }
        ),
        encoding="utf-8",
    )
    before = _git(tmp_path, "status", "--porcelain=v1")

    listed = _run(tmp_path, "eval-list")
    shown = _run(tmp_path, "eval-show", "run-config-extension")
    scored = _run(tmp_path, "eval-score", "run-config-extension", str(score_path))
    unknown = _run(tmp_path, "eval-show", "missing-task")

    assert listed.returncode == 0, listed.stderr
    listing = json.loads(listed.stdout)
    assert listing["rubric_version"] == "agent_repo_rubric_v1"
    assert listing["task_ids"] == [
        "run-config-extension",
        "sha256-validation-reuse",
        "dataset-scope-change",
        "run-artifact-compatible-extension",
        "binance-fallback-fix",
    ]
    assert shown.returncode == 0, shown.stderr
    task = json.loads(shown.stdout)
    assert set(task) == {"task_id", "prompt"}
    assert task["task_id"] == "run-config-extension"
    assert task["prompt"]
    assert scored.returncode == 0, scored.stderr
    score = json.loads(scored.stdout)
    assert score["total"] == 16
    assert score["maximum"] == 16
    assert score["critical_failure"] is False
    assert unknown.returncode != 0
    assert unknown.stdout == ""
    assert _git(tmp_path, "status", "--porcelain=v1") == before


def test_cli_returns_nonzero_for_invalid_context_path(tmp_path: Path) -> None:
    _repository(tmp_path)

    result = _run(tmp_path, "context", "trade_rl/missing.py")

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr
