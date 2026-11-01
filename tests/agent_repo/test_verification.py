from __future__ import annotations

import subprocess
from pathlib import Path

from tools.agent_repo.verification import VerificationStep, plan_verification

FINAL_COMMANDS = {
    "uv run ruff check trade_rl tests tools",
    "uv run ruff format --check trade_rl tests tools",
    "uv run mypy trade_rl",
    "uv run mypy tools/agent_repo tests/architecture/distribution.py",
    "uv run pytest -q tests",
    "uv build",
    "uv run python -m tests.architecture.distribution dist/*.tar.gz dist/*.whl",
    "CI composite: rebuild wheel from sdist and verify source closure",
    "CI composite: clean installed core smoke",
    "CI composite: package identity",
}


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
    assert final_commands == FINAL_COMMANDS
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
    assert {step.command for step in _by_tier(steps, "final")} == FINAL_COMMANDS


def _test_stub(root: Path, path: str) -> None:
    _write(root, path, "def test_contract() -> None:\n    assert True\n")


def test_run_config_change_routes_persisted_contract_and_cli_tests(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    for name in (
        "test_candidate_run_config.py",
        "test_candidate_run_cli.py",
        "test_candidate_run_artifact.py",
        "test_candidate_run_provenance.py",
    ):
        _test_stub(tmp_path, f"tests/evaluation/{name}")
    _write(
        tmp_path,
        "trade_rl/evaluation/runs/config.py",
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\nclass CandidateRunConfig:\n"
        "    value: int\n    extra_field: int\n",
    )
    fast = " ".join(step.command for step in _by_tier(
        plan_verification(tmp_path, base_ref="main"), "fast"
    ))
    for name in (
        "test_candidate_run_config.py",
        "test_candidate_run_cli.py",
        "test_candidate_run_artifact.py",
        "test_candidate_run_provenance.py",
    ):
        assert name in fast


def test_sha256_contract_change_routes_existing_common_validator_tests(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    _write(tmp_path, "trade_rl/_validation.py",
           "def require_sha256(value: str, *, field: str) -> str:\n"
           "    return value\n")
    _test_stub(tmp_path, "tests/test_validation.py")
    _write(
        tmp_path,
        "trade_rl/evaluation/runs/config.py",
        "from trade_rl._validation import require_sha256\n\n"
        "def validate_digest(digest: str) -> str:\n"
        "    return require_sha256(digest, field='artifact_digest')\n",
    )
    fast = " ".join(step.command for step in _by_tier(
        plan_verification(tmp_path, base_ref="main"), "fast"
    ))
    assert "tests/test_validation.py" in fast


def test_dataset_scope_change_routes_shared_strategy_contract(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    _test_stub(tmp_path, "tests/strategies/test_dataset_scope_contract.py")
    _write(
        tmp_path,
        "trade_rl/strategies/scope.py",
        "def select_feature_names(names: tuple[str, ...]) -> tuple[str, ...]:\n"
        "    return tuple(sorted(names))\n",
    )
    fast = " ".join(step.command for step in _by_tier(
        plan_verification(tmp_path, base_ref="main"), "fast"
    ))
    assert "tests/strategies/test_dataset_scope_contract.py" in fast


def test_run_artifact_change_routes_legacy_compatibility_and_tamper_oracles(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    for name in (
        "test_candidate_run_artifact.py",
        "test_candidate_run_artifact_identity.py",
        "test_candidate_run_malformed_archive_errors.py",
        "test_candidate_run_provenance.py",
    ):
        _test_stub(tmp_path, f"tests/evaluation/{name}")
    _write(
        tmp_path,
        "trade_rl/evaluation/runs/artifact.py",
        "RESULT_SCHEMA = 'candidate_result_v2'\n"
        "def publish() -> str:\n    return RESULT_SCHEMA\n",
    )
    steps = plan_verification(tmp_path, base_ref="main")
    fast = " ".join(step.command for step in _by_tier(steps, "fast"))
    for name in (
        "test_candidate_run_artifact.py",
        "test_candidate_run_artifact_identity.py",
        "test_candidate_run_malformed_archive_errors.py",
        "test_candidate_run_provenance.py",
    ):
        assert name in fast
    assert any("compatibility" in step.reason.lower()
               for step in _by_tier(steps, "extended"))


def test_binance_fallback_condition_change_routes_negative_integrity_tests(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    for name in (
        "test_binance_exchange_info_fallback.py",
        "test_binance_response_bounds.py",
        "test_binance_cache_integrity_contract.py",
    ):
        _test_stub(tmp_path, f"tests/integrations/{name}")
    _write(
        tmp_path,
        "trade_rl/integrations/binance/transport.py",
        "def load():\n"
        "    try:\n        return b'primary'\n"
        "    except OSError:\n        return b'fallback'\n",
    )
    steps = plan_verification(tmp_path, base_ref="main")
    fast = " ".join(step.command for step in _by_tier(steps, "fast"))
    for name in (
        "test_binance_exchange_info_fallback.py",
        "test_binance_response_bounds.py",
        "test_binance_cache_integrity_contract.py",
    ):
        assert name in fast
    assert any(
        "fallback" in step.reason.lower() and "integrity" in step.reason.lower()
        for step in _by_tier(steps, "extended")
    )
