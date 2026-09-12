from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_agent_coordination_stays_outside_production_runtime_and_wheel() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.packages.find]" in pyproject
    assert 'include = ["trade_rl*"]' in pyproject

    forbidden = "tools.agent_repo"
    for path in sorted((ROOT / "trade_rl").rglob("*.py")):
        assert forbidden not in path.read_text(encoding="utf-8"), path.relative_to(ROOT)


def test_agent_docs_define_coordination_operator_and_evidence_contract() -> None:
    agents = (ROOT / "docs" / "AGENTS.md").read_text(encoding="utf-8")
    for required in (
        "Agent Coordination Plane",
        "task digest",
        "task ready",
        "read_only",
        "lease epoch",
        "Task Contract digest",
        "exact HEAD",
    ):
        assert required in agents


def test_package_boundaries_define_coordination_as_repository_tooling() -> None:
    boundaries = (ROOT / "docs" / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "tools/agent_repo/coordination/",
        "Task Packet",
        "Resource Key",
        "lease epoch",
        "phase + condition",
        "production wheel",
    ):
        assert required in boundaries


def test_completed_coordination_ephemeral_docs_are_removed() -> None:
    completed = (
        ROOT / "docs" / "specs" / "2026-09-12-agent-coordination-plane-v1-design.md",
        ROOT
        / "docs"
        / "plans"
        / "2026-09-12-agent-coordination-plane-v1-implementation.md",
    )
    for path in completed:
        assert not path.exists(), path.relative_to(ROOT)

    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    for path in completed:
        assert path.name not in index
