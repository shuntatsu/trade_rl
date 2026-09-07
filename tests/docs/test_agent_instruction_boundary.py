from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_repository_and_docs_scoped_agent_instructions_exist() -> None:
    assert (ROOT / "AGENTS.md").is_file()
    assert (ROOT / "docs" / "AGENTS.md").is_file()


def test_docs_scoped_agent_instructions_are_not_site_documents() -> None:
    from scripts.docs.metadata import discover_documents

    documents = discover_documents(ROOT / "docs")
    assert "AGENTS.md" not in {document.path.as_posix() for document in documents}


def test_agent_context_commands_use_declared_docs_dependencies() -> None:
    required = "uv run --extra dev python scripts/docs/context.py"
    root_agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    docs_agents = (ROOT / "docs" / "AGENTS.md").read_text(encoding="utf-8")
    docs_portal = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    assert required in root_agents
    assert required in docs_agents
    assert required in docs_portal
