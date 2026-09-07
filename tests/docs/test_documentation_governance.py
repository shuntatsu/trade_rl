from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_documentation_governance_infrastructure_exists() -> None:
    required = (
        ROOT / "AGENTS.md",
        ROOT / "mkdocs.yml",
        ROOT / "scripts" / "docs" / "metadata.py",
        ROOT / "scripts" / "docs" / "manifest.py",
        ROOT / "scripts" / "docs" / "generate.py",
        ROOT / "scripts" / "docs" / "validate.py",
        ROOT / "scripts" / "docs" / "context.py",
    )
    missing = [
        path.relative_to(ROOT).as_posix() for path in required if not path.is_file()
    ]
    assert missing == []


def test_current_documentation_uses_intent_and_authority_layout() -> None:
    required = (
        ROOT / "docs" / "index.md",
        ROOT / "docs" / "getting-started" / "quickstart.md",
        ROOT / "docs" / "guides" / "binance-data.md",
        ROOT / "docs" / "reference" / "architecture.md",
        ROOT / "docs" / "reference" / "configuration.md",
        ROOT / "docs" / "reference" / "single-symbol.md",
        ROOT / "docs" / "reference" / "universal-trade-rl.md",
        ROOT / "docs" / "reference" / "universal-training.md",
        ROOT / "docs" / "reference" / "reward-objective.md",
        ROOT / "docs" / "reference" / "execution-robustness.md",
        ROOT / "docs" / "reference" / "run-reporting.md",
        ROOT / "docs" / "reference" / "nautilus-migration.md",
        ROOT / "docs" / "research" / "status.md",
        ROOT / "docs" / "research" / "multi-timeframe.md",
        ROOT / "docs" / "legal" / "licensing.md",
        ROOT / "docs" / "legal" / "licensing-provenance.md",
        ROOT / "docs" / "history" / "index.md",
    )
    missing = [
        path.relative_to(ROOT).as_posix() for path in required if not path.is_file()
    ]
    assert missing == []


def test_obsolete_mixed_authority_document_roots_are_removed() -> None:
    obsolete = (
        ROOT / "docs" / "implementation",
        ROOT / "docs" / "implementation-plans",
        ROOT / "docs" / "architecture",
        ROOT / "docs" / "ARCHITECTURE.md",
        ROOT / "docs" / "CONFIGURATION.md",
        ROOT / "docs" / "RESEARCH_STATUS.md",
        ROOT / "docs" / "UNIVERSAL_TRAINING.md",
    )
    remaining = [
        path.relative_to(ROOT).as_posix() for path in obsolete if path.exists()
    ]
    assert remaining == []


def test_current_docs_do_not_reference_obsolete_history_roots() -> None:
    from scripts.docs.metadata import discover_documents

    obsolete = (
        "docs/implementation/",
        "docs/implementation-plans/",
        "docs/architecture/",
    )
    offenders: list[str] = []
    for document in discover_documents(ROOT / "docs"):
        if document.lifecycle != "current":
            continue
        text = (ROOT / "docs" / document.path).read_text(encoding="utf-8")
        for marker in obsolete:
            if marker in text:
                offenders.append(f"docs/{document.path.as_posix()}: {marker}")
    assert offenders == []


def test_generated_documentation_outputs_are_gitignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    ignored = {line.strip() for line in gitignore.splitlines()}
    assert "/site" in ignored
    assert "/.docs-build/" in ignored


def test_agent_bootstrap_routes_to_governed_documentation() -> None:
    agents = ROOT / "AGENTS.md"
    assert agents.is_file()
    text = agents.read_text(encoding="utf-8")
    for required in (
        "docs/index.md",
        "scripts/docs/context.py",
        "docs/history/",
        "current runtime authority",
        "source_of_truth_for",
    ):
        assert required in text
