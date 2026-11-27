"""The completed P3 diagnostic is historical; shared PPO runtime stays in core."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
HISTORY = "7342f9c224371a687e98a7e0d3f6a9f121c8d802"
RETIRED = {
    "tools/issue810_p3_rollout_boundary_fixture.py": "9c5eef3f4b5ccd2187f17b01f52a5d4e456ec4e0fef1e8964a68f76a95db213d",
    "tools/issue810_p3_rollout_boundary_diagnostic.py": "987776c3bec3b26ac243d29ab1d08ab009247fdd1293a6a9e3a06a6a182c3ed9",
    "tests/strategies/rl/test_issue810_p3_rollout_boundary_fixture.py": "ebe7bea2477faee684ff9191ec576fec4a390239c35a04c2f7c1acb2791a007c",
    "tests/strategies/rl/test_issue810_p3_rollout_boundary_runner.py": "42058ebb3dbaa60d6d2612895ef71fb727f1cd8e6eba53f7d94aece01e463abe",
}
DOCUMENTS = {
    "docs/research/ppo-rollout-boundary-protocol.md": "9a469fcb9b8e8784d9a0d4786a5d62630de0354b2962e96a079b8227bf8ed823",
    "docs/research/ppo-rollout-boundary-results.md": "da9ceedb22e0320664551e670437181ef7be13f57713813a93676499ebe39572",
}
MODULES = tuple(
    path.removesuffix(".py").replace("/", ".")
    for path in RETIRED
    if path.startswith("tools/")
)


def test_closed_study_sources_and_duplicate_tests_are_absent() -> None:
    for path in RETIRED:
        assert not (ROOT / path).exists(), path
    for module in MODULES:
        assert importlib.util.find_spec(module) is None


def test_original_source_and_tests_are_byte_preserved() -> None:
    for path, digest in RETIRED.items():
        raw = subprocess.check_output(["git", "show", f"{HISTORY}:{path}"], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == digest, path


def test_frozen_protocol_and_result_documents_are_unchanged() -> None:
    # Markdown byte digests, distinct from the canonical protocol JSON digest.
    for path, digest in DOCUMENTS.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path


def test_no_runtime_imports_of_retired_diagnostic_and_fixture() -> None:
    for package_name in ("trade_rl", "tools"):
        package = ROOT / package_name
        collector = ImportCollector(package)
        for path in package.rglob("*.py"):
            assert not any(
                within_module(name, module)
                for name in collector.collect(path)
                for module in MODULES
            ), path


def test_shared_ppo_runtime_is_retained() -> None:
    for module in (
        "trade_rl.strategies.rl.portfolio_env",
        "trade_rl.strategies.rl.portfolio_transform",
        "trade_rl.simulation.stateful.execution",
    ):
        assert importlib.util.find_spec(module) is not None
