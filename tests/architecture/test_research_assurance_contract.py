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


def test_versioned_author_and_review_schemas_are_separate_and_strict() -> None:
    base = ROOT / "tools/agent_repo/schemas"
    author = json.loads((base / "research_assurance_v1.schema.json").read_text())
    review = json.loads((base / "research_assurance_review_v1.schema.json").read_text())
    example = json.loads(
        (ROOT / "tools/agent_repo/examples/research_assurance.json").read_text()
    )
    assert author["$schema"].endswith("/2020-12/schema")
    assert author["additionalProperties"] is False
    assert set(author["required"]) == set(example)
    assert "review" not in author["properties"]
    assert review["additionalProperties"] is False
    assert set(review["required"]) >= {
        "reviewed_record_digest",
        "protocol_head",
        "implementation_head",
        "reviewer_identity",
        "reviewer_independence",
        "result_blind",
        "blocking_findings",
        "missing_evidence",
        "disposition",
    }
    mechanism = author["properties"]["mechanism"]
    assert "changed_boundaries" in mechanism["required"]
    assert "independent_oracles" in mechanism["required"]
    for boundary in example["mechanism"]["changed_boundaries"]:
        oracle = example["mechanism"]["independent_oracles"][boundary]
        assert set(oracle) == {
            "statement",
            "counterexample",
            "oracle",
            "known_limitations",
        }


def test_review_adapter_is_bounded_to_real_authenticated_github_readback() -> None:
    adapter = (ROOT / "tools/research_assurance_github.py").read_text()
    assert '"authenticated_pr_review"' in adapter
    assert "pulls/{pr_number}" in adapter
    assert "commit_id" in adapter
    assert "reviewer_id == author_id" in adapter
    assert "pr_final = api_read(pr_url)" in adapter
    assert 'result.status == "PASS"' in adapter
