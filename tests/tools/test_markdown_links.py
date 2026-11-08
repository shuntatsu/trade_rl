"""Network-free Markdown link-checker contract tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.markdown_links import check_links, destinations, main, tracked_markdown


def _write(root: Path, relative: str, content: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_inline_and_reference_links_resolve_without_network(tmp_path: Path) -> None:
    _write(tmp_path, "docs/README.md", "# Source\n")
    _write(tmp_path, "guide/content/page.md", "# Guide\n")
    _write(
        tmp_path,
        "docs/start.md",
        "[Source](README.md#section) ![image](../guide/content/page.md)\n"
        "[reference]: <README.md> \"description\"\n"
        "[Web](https://example.invalid/path) [Mail](mailto:x@example.invalid)\n",
    )
    checked, problems = check_links(tmp_path, (Path("docs/start.md"),))
    assert checked == 3
    assert problems == []


def test_broken_link_in_guide_is_checked(tmp_path: Path) -> None:
    _write(tmp_path, "guide/content/page.md", "[Missing](../../docs/missing.md)\n")
    count, errors = check_links(tmp_path, (Path("guide/content/page.md"),))
    assert count == 1
    assert len(errors) == 1
    assert errors[0].reason == "missing local target"
    assert errors[0].line == 1


def test_code_fences_inline_code_and_html_comments_are_not_links() -> None:
    markdown = (
        "[Actual](real.md) ` [code](missing.md) `\n"
        "```md\n[fenced](missing.md)\n```\n"
        "<!--\n[comment](missing.md)\n-->\n"
        "<!-- [inline comment](missing.md) --> [Second](next.md)\n"
        "~~~\n[fenced](missing.md)\n~~~\n"
    )
    assert destinations(markdown) == [(1, "real.md"), (8, "next.md")]


def test_nested_parentheses_and_escaped_parens_in_path(tmp_path: Path) -> None:
    _write(tmp_path, "docs/name(v2).md", "# Valid\n")
    _write(tmp_path, "docs/main.md", "[A](name(v2).md) [B](name\\(v2\\).md)\n")
    count, errors = check_links(tmp_path, (Path("docs/main.md"),))
    assert count == 2
    assert errors == []


def test_traversal_and_unsafe_schemes_fail_closed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "guide/content/page.md",
        "[Escape](../../../outside.md) [Absolute](/etc/passwd) "
        "[Script](javascript:alert(1))\n",
    )
    count, errors = check_links(tmp_path, (Path("guide/content/page.md"),))
    assert count == 2
    assert [error.reason for error in errors] == [
        "path escapes checkout",
        "absolute local path",
        "unsupported URI scheme",
    ]


def test_nonexistent_tracked_markdown_fails_closed(tmp_path: Path) -> None:
    checked, problems = check_links(tmp_path, (Path("missing.md"),))
    assert checked == 0
    assert problems[0].reason == "unreadable Markdown"


def test_markdown_source_outside_checkout_is_never_read(tmp_path: Path) -> None:
    checked, problems = check_links(tmp_path, (Path("../outside.md"),))
    assert checked == 0
    assert problems[0].reason == "source escapes checkout"


def test_tracked_files_include_guide_and_active_specs_without_untracked(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for path in ("README.md", "guide/content/page.md", "docs/specs/current.md"):
        _write(tmp_path, path, "# Test\n")
    _write(tmp_path, "docs/untracked.md", "# Untracked\n")
    subprocess.run(
        ["git", "add", "README.md", "guide", "docs/specs"],
        cwd=tmp_path,
        check=True,
    )
    assert tracked_markdown(tmp_path) == (
        Path("README.md"),
        Path("docs/specs/current.md"),
        Path("guide/content/page.md"),
    )
    assert main(["--root", str(tmp_path), "--check"]) == 0


def test_empty_repo_does_not_claim_success(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(ValueError, match="no tracked Markdown"):
        tracked_markdown(tmp_path)
