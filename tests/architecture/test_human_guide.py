from pathlib import Path
import json


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

    package = json.loads((GUIDE / "package.json").read_text(encoding="utf-8"))
    assert package["private"] is True
    assert package["engines"]["node"].startswith("24")


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
