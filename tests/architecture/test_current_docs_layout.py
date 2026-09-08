from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

REQUIRED_DOC_FILES = {
    "AGENTS.md",
    "README.md",
    "architecture/lean-core.md",
    "architecture/package-boundaries.md",
    "research/current-status.md",
}
EPHEMERAL_DOC_ROOTS = {"plans", "specs"}


def _doc_files() -> set[str]:
    result: set[str] = set()
    for path in DOCS.rglob("*"):
        if path.is_file():
            result.add(path.relative_to(DOCS).as_posix())
    return result


def test_docs_tree_contains_current_authorities_and_only_active_ephemeral_docs() -> (
    None
):
    files = _doc_files()
    assert REQUIRED_DOC_FILES <= files

    for relative in sorted(files - REQUIRED_DOC_FILES):
        path = Path(relative)
        assert path.parts[0] in EPHEMERAL_DOC_ROOTS, relative
        assert path.suffix == ".md", relative
        text = (DOCS / path).read_text(encoding="utf-8")
        assert "Status: Active" in text, relative

    assert not (DOCS / "history").exists()
    assert not (DOCS / "archive").exists()


def test_phase4b_preview_workflow_is_absent() -> None:
    path = ROOT / ".github" / "workflows" / "phase4b-format-preview.yml"
    assert not path.exists()


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
        "Status: Active",
        "docs/history",
        "docs/archive",
    ):
        assert required in contract


def test_current_docs_preserve_core_and_research_contracts() -> None:
    lean_core = (DOCS / "architecture" / "lean-core.md").read_text(encoding="utf-8")
    package_boundaries = (DOCS / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    research = (DOCS / "research" / "current-status.md").read_text(encoding="utf-8")

    for required in (
        "quantity-preserving hold",
        "MarketExecutor + BookState",
        "feature_available_time <= decision_time",
        "SHORT",
        "FLAT",
        "LONG",
    ):
        assert required in lean_core

    for required in (
        "artifacts / data / integrations / risk / simulation / strategies / evaluation",
        "_validation -> standard library only",
        "evaluation/experiments/",
        "private module path",
    ):
        assert required in package_boundaries

    for required in (
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
        "ppo",
        "cash",
        "constant_long",
        "constant_short",
        "fit_symbol_names",
        "no winner",
        "real-data development comparison not run",
        "python -m trade_rl.evaluation.runs.candidate",
        "fee adverse",
        "spread adverse",
        "+1 decision latency",
        "Production/live order routing",
    ):
        assert required in research


def test_removed_monolithic_doc_is_not_referenced() -> None:
    old_path = "docs/trade_rl_lean_redesign_20260908.md"
    offenders: list[str] = []
    for path in ROOT.rglob("*.md"):
        if old_path in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_root_readme_uses_current_docs_entry_point() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/README.md" in readme
    assert "docs/trade_rl_lean_redesign_20260908.md" not in readme
