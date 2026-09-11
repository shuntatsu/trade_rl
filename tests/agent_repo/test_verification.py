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
    _write(root, "trade_rl/integrations/__init__.py", "")
    _write(root, "trade_rl/integrations/binance/__init__.py", "")
    _write(
        root,
        "trade_rl/integrations/binance/transport.py",
        "def load() -> bytes:\n    return b'x'\n",
    )
    _write(
        root,
        "tests/evaluation/runs/test_config.py",
        "from trade_rl.evaluation.runs import CandidateRunConfig\n",
    )
    _write(
        root,
        "tests/integrations/test_binance.py",
        "def test_transport(): assert True\n",
    )
    for name in (
        "test_runs_capability_facade.py",
        "test_lean_dependency_boundaries.py",
        "test_import_collector_contract.py",
        "test_current_docs_layout.py",
    ):
        _write(
            root, f"tests/architecture/{name}", "def test_placeholder(): assert True\n"
        )
    _write(root, "docs/README.md", "# Docs\n")
    _write(root, "pyproject.toml", "[project]\nname='example'\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "baseline")


def _by_tier(steps: tuple[VerificationStep, ...], tier: str) -> list[VerificationStep]:
    return [step for step in steps if step.tier == tier]


def test_runs_change_gets_targeted_fast_tests_and_full_final_gate(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    _write(
        tmp_path,
        "trade_rl/evaluation/runs/config.py",
        "from dataclasses import dataclass\n\n"
        "DEFAULT_VALUE = 2\n\n"
        "@dataclass(frozen=True)\n"
        "class CandidateRunConfig:\n"
        "    value: int\n",
    )

    steps = plan_verification(tmp_path, base_ref="main")

    fast_commands = {step.command for step in _by_tier(steps, "fast")}
    assert any(
        "tests/evaluation/runs/test_config.py" in value for value in fast_commands
    )
    assert any("test_runs_capability_facade.py" in value for value in fast_commands)
    final_commands = {step.command for step in _by_tier(steps, "final")}
    assert {
        "uv run ruff check trade_rl tests tools",
        "uv run ruff format --check trade_rl tests tools",
        "uv run mypy trade_rl",
        "uv run mypy tools/agent_repo tests/architecture/distribution.py",
        "uv run pytest -q tests",
        "uv build",
    } <= final_commands
    coverage = _by_tier(steps, "signal")
    assert len(coverage) == 1
    assert "--cov-fail-under=0" in coverage[0].command
    assert "target=80%" in coverage[0].reason


def test_network_effect_adds_only_relevant_extended_check(tmp_path: Path) -> None:
    _baseline(tmp_path)
    _write(
        tmp_path,
        "trade_rl/integrations/binance/transport.py",
        "import urllib.request\n\n"
        "def load() -> bytes:\n"
        "    return urllib.request.urlopen('https://example.invalid').read()\n",
    )

    steps = plan_verification(tmp_path, base_ref="main")

    extended = _by_tier(steps, "extended")
    assert any("tests/integrations" in step.command for step in extended)
    assert any("network" in step.reason.lower() for step in extended)
    assert all("windows" not in step.reason.lower() for step in extended)


def test_schema_change_requests_compatibility_review(tmp_path: Path) -> None:
    _baseline(tmp_path)
    _write(
        tmp_path,
        "trade_rl/evaluation/runs/config.py",
        "from dataclasses import dataclass\n\n"
        "RUN_SCHEMA = 'run_v2'\n\n"
        "@dataclass(frozen=True)\n"
        "class CandidateRunConfig:\n"
        "    value: int\n",
    )

    steps = plan_verification(tmp_path, base_ref="main")

    assert any(
        step.tier == "extended"
        and "compatibility" in step.reason.lower()
        and "tamper" in step.reason.lower()
        for step in steps
    )


def test_docs_only_change_gets_docs_fast_check_without_coverage_signal(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    _write(tmp_path, "docs/README.md", "# Docs\n\nChanged.\n")

    steps = plan_verification(tmp_path, base_ref="main")

    assert any(
        step.tier == "fast" and "test_current_docs_layout.py" in step.command
        for step in steps
    )
    assert _by_tier(steps, "signal") == []
    assert any(
        step.tier == "final" and step.command == "uv run pytest -q tests"
        for step in steps
    )
