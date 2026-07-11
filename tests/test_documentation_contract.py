from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NORMATIVE_DOCS = {
    "ARCHITECTURE.md",
    "OPERATIONS.md",
    "SECURITY.md",
    "MODEL_LIFECYCLE.md",
    "TESTING.md",
    "PRODUCTION_READINESS.md",
    "DECISIONS.md",
    "RESEARCH_HISTORY.md",
}


def test_normative_documentation_set_is_complete() -> None:
    docs_dir = ROOT / "docs"
    actual = {path.name for path in docs_dir.glob("*.md")}
    assert actual == NORMATIVE_DOCS
    assert (ROOT / "README.md").is_file()


def test_root_readme_declares_no_go_and_links_architecture() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/ARCHITECTURE.md" in readme
    assert "Production: NO-GO" in readme


def test_normative_docs_do_not_describe_removed_paths_as_current() -> None:
    text = "\n".join(
        (ROOT / "docs" / name).read_text(encoding="utf-8")
        for name in sorted(NORMATIVE_DOCS)
    )
    forbidden = (
        "mars_lite.server.model_registry",
        "mars_lite.serving.model_store",
        "from mars_lite.server.metrics_server import run_server",
        'allow_origins=["*"]',
    )
    for value in forbidden:
        assert value not in text


def test_production_readiness_remains_no_go() -> None:
    readiness = (ROOT / "docs" / "PRODUCTION_READINESS.md").read_text(
        encoding="utf-8"
    )
    assert "Current decision: **NO-GO**" in readiness
    assert "- [ ]" in readiness
