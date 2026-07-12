from __future__ import annotations

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
JAPANESE_DOCS = NORMATIVE_DOCS | {"README.md"}


def test_normative_documentation_set_is_complete() -> None:
    docs_dir = ROOT / "docs"
    actual = {path.name for path in docs_dir.glob("*.md")}
    assert actual == NORMATIVE_DOCS
    assert (ROOT / "README.md").is_file()


def test_japanese_documentation_set_is_complete() -> None:
    japanese_dir = ROOT / "docs" / "ja"
    actual = {path.name for path in japanese_dir.glob("*.md")}
    assert actual == JAPANESE_DOCS
    assert (ROOT / "README.ja.md").is_file()


def test_root_readme_declares_no_go_and_links_architecture() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/ARCHITECTURE.md" in readme
    assert "Production: NO-GO" in readme
    assert "README.ja.md" in readme
    assert "docs/ja/README.md" in readme


def test_japanese_docs_link_to_english_sources() -> None:
    japanese_dir = ROOT / "docs" / "ja"
    for name in sorted(NORMATIVE_DOCS):
        text = (japanese_dir / name).read_text(encoding="utf-8")
        assert f"../{name}" in text


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
    readiness = (ROOT / "docs" / "PRODUCTION_READINESS.md").read_text(encoding="utf-8")
    assert "Current decision: **NO-GO**" in readiness
    assert "- [ ]" in readiness

    japanese_readiness = (ROOT / "docs" / "ja" / "PRODUCTION_READINESS.md").read_text(
        encoding="utf-8"
    )
    assert "現在の判断: **NO-GO**" in japanese_readiness
    assert "- [ ]" in japanese_readiness


def test_local_validation_contract_is_documented_in_both_languages() -> None:
    english = "\n".join(
        [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "OPERATIONS.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "PRODUCTION_READINESS.md").read_text(encoding="utf-8"),
        ]
    )
    japanese = "\n".join(
        [
            (ROOT / "README.ja.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "ja" / "ARCHITECTURE.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "ja" / "OPERATIONS.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "ja" / "PRODUCTION_READINESS.md").read_text(
                encoding="utf-8"
            ),
        ]
    )
    required = (
        "--p0-days",
        "content-addressed",
        "completed bar",
        "TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1",
        "uv run python scripts/run_local_gameday.py",
        "single-node",
    )
    for phrase in required:
        assert phrase in english
        assert phrase in japanese
