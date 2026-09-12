from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "guide"


def test_human_guide_is_non_authoritative_and_isolated() -> None:
    readme = GUIDE / "README.md"
    package = GUIDE / "package.json"

    assert readme.is_file()
    assert package.is_file()

    text = readme.read_text(encoding="utf-8")
    assert "non-authoritative" in text
    assert "docs/architecture" in text
    assert "docs/research/current-status.md" in text

    assert not (ROOT / "docs" / "history").exists()
    assert not (ROOT / "docs" / "archive").exists()


def test_human_guide_node_dependencies_do_not_leak_into_python_package() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for dependency in ("react", "vite", "tailwindcss", "playwright", "vitest"):
        assert dependency not in pyproject.lower()

    package = (GUIDE / "package.json").read_text(encoding="utf-8")
    assert '"private": true' in package
    assert '"node": ">=24 <25"' in package


def test_human_guide_generated_outputs_are_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for path in (
        "guide/node_modules/",
        "guide/dist/",
        "guide/coverage/",
        "guide/playwright-report/",
        "guide/test-results/",
    ):
        assert path in ignored


def test_human_guide_has_permanent_repository_routing_contract() -> None:
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_readme = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    docs_agents = (ROOT / "docs" / "AGENTS.md").read_text(encoding="utf-8")

    assert "guide/README.md" in root_readme
    assert "non-authoritative" in root_readme

    assert "guide/README.md" in docs_readme
    assert "非正本" in docs_readme

    assert "guide/content/topics" in docs_agents
    assert "content_contract.py --check" in docs_agents
    assert "非正本" in docs_agents


def test_completed_human_guide_design_artifacts_are_removed() -> None:
    assert not (ROOT / "docs/specs/2026-09-12-interactive-human-guide-v1.md").exists()
    assert not (ROOT / "docs/plans/2026-09-12-interactive-human-guide-v1.md").exists()
