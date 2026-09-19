from __future__ import annotations

import json
from pathlib import Path

from tools.agent_repo.research_assurance import evaluate_assurance

ROOT = Path(__file__).resolve().parents[2]


def test_research_assurance_authority_is_routed_from_repository_instructions() -> None:
    root_agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    docs_agents = (ROOT / "docs/AGENTS.md").read_text(encoding="utf-8")
    docs_readme = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    authority = (ROOT / "docs/architecture/research-assurance.md").read_text(
        encoding="utf-8"
    )

    assert "docs/architecture/research-assurance.md" in root_agents
    assert "Research Assurance Gate" in root_agents
    assert "architecture/research-assurance.md" in docs_agents
    assert "assurance check assurance.json" in docs_agents
    assert "architecture/research-assurance.md" in docs_readme
    assert "Issue #667" in authority
    assert "項目を全部埋めただけでは" in authority
    assert "PASS" in authority


def test_generic_assurance_example_is_valid_but_not_self_authorizing() -> None:
    payload = json.loads(
        (ROOT / "tools/agent_repo/examples/research_assurance.json").read_text(
            encoding="utf-8"
        )
    )

    result = evaluate_assurance(payload)

    assert result.status == "UNREVIEWED"
    assert result.economic_execution_authorized is False
    assert result.errors == ()
