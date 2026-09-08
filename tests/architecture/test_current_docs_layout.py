from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

EXPECTED_DOC_FILES = {
    "AGENTS.md",
    "README.md",
    "architecture/lean-core.md",
    "architecture/package-boundaries.md",
    "research/current-status.md",
}


def _doc_files() -> set[str]:
    result: set[str] = set()
    for path in DOCS.rglob("*"):
        if path.is_file():
            result.add(path.relative_to(DOCS).as_posix())
    return result


def test_docs_tree_contains_only_current_authorities() -> None:
    assert _doc_files() == EXPECTED_DOC_FILES
    assert not (DOCS / "history").exists()
    assert not (DOCS / "archive").exists()
    assert not (DOCS / "plans").exists()
    assert not (DOCS / "specs").exists()


def test_root_agent_entry_routes_to_docs_contract() -> None:
    root_agents = ROOT / "AGENTS.md"
    assert root_agents.is_file()
    text = root_agents.read_text(encoding="utf-8")
    assert "docs/README.md" in text
    assert "docs/AGENTS.md" in text


def test_docs_index_routes_to_every_current_doc() -> None:
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for target in (
        "AGENTS.md",
        "architecture/lean-core.md",
        "architecture/package-boundaries.md",
        "research/current-status.md",
    ):
        assert target in index


def test_agent_contract_defines_update_and_retention_policy() -> None:
    contract = (DOCS / "AGENTS.md").read_text(encoding="utf-8")
    for required in (
        "Git history",
        "LICENSES/",
        "architecture/",
        "research/",
        "specs/",
        "plans/",
        "docs/history",
        "docs/archive",
    ):
        assert required in contract


def test_root_readme_uses_current_docs_entry_point() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/README.md" in readme
    assert "docs/trade_rl_lean_redesign_20260908.md" not in readme
