from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "pull_request_template.md"
ROOT_AGENTS = ROOT / "AGENTS.md"
DOC_AGENTS = ROOT / "docs" / "AGENTS.md"
PACKAGE_BOUNDARIES = ROOT / "docs" / "architecture" / "package-boundaries.md"
DESIGN_SPEC = (
    ROOT / "docs" / "specs" / "2026-09-11-agent-repository-control-plane-v1-design.md"
)
MERGE_PLAN = (
    ROOT
    / "docs"
    / "plans"
    / "2026-09-11-agent-repository-merge-safety-implementation.md"
)
REQUIRED = (
    "## Objective",
    "## Non-goals",
    "## Acceptance Criteria",
    "## Invariants",
    "## Failure Modes",
    "## Test Oracle",
    "## Changed authorities / public surfaces",
    "## Tests / verification",
    "## Falsification",
    "## Unverified items",
    "## Residual risk",
)


def test_pr_template_contains_quality_contract() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = [heading for heading in REQUIRED if heading not in text]
    assert missing == []


def test_merge_policy_requires_tested_head_to_include_current_main() -> None:
    root_agents = ROOT_AGENTS.read_text(encoding="utf-8")
    doc_agents = DOC_AGENTS.read_text(encoding="utf-8")
    package_boundaries = PACKAGE_BOUNDARIES.read_text(encoding="utf-8")
    design_spec = DESIGN_SPEC.read_text(encoding="utf-8")
    merge_plan = MERGE_PLAN.read_text(encoding="utf-8")

    required = "tested PR head contains current `main`"
    for text in (root_agents, doc_agents, package_boundaries, design_spec):
        assert required in text
    assert "## Repository integration boundary" in package_boundaries
    assert 'Do not require "branch must be up to date"' not in merge_plan
    assert (
        "Require the PR branch to be up to date with current `main` before merge"
        in merge_plan
    )
