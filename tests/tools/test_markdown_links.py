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
        '[reference]: <README.md> "description"\n'
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
    # CommonMark treats the entire line beginning with <!-- as raw HTML.
    assert destinations(markdown) == [(1, "real.md")]


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


def test_multiline_inline_code_does_not_create_links() -> None:
    assert destinations("`literal\n[Missing](missing.md)\n`\n") == []


def test_indented_code_is_excluded_but_paragraph_continuation_is_not() -> None:
    assert destinations("    [literal](missing.md)\n") == []
    assert destinations("Paragraph\n    [real](present.md)\n") == [(2, "present.md")]


def test_nested_bracket_labels_detect_the_outer_link() -> None:
    assert destinations("[outer [inner]](missing.md)\n") == [(1, "missing.md")]


def test_data_url_is_not_whitelisted(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "[payload](data:text/plain,hello)\n")
    count, problems = check_links(tmp_path, (Path("README.md"),))
    assert count == 0
    assert [problem.reason for problem in problems] == ["unsupported URI scheme"]


def test_plain_unclosed_link_is_not_considered_valid_markdown() -> None:
    assert destinations("[link](missing.md\n") == []


def test_quoted_indented_code_and_quoted_fence_are_literal() -> None:
    assert destinations(">     [literal](missing.md)\n") == []
    assert destinations("> Example\n>     [real](present.md)\n") == [(2, "present.md")]
    assert destinations("> ```md\n> [literal](missing.md)\n> ```\n") == []


def test_raw_html_block_is_not_interpreted_as_markdown_links() -> None:
    assert destinations("<div>\n[missing](missing.md)\n</div>\n") == []
    html = "<div>\n[missing](missing.md)\n\n[real](present.md)\n"
    assert destinations(html) == [(4, "present.md")]
    assert destinations("<https://example.org> [real](present.md)\n") == [
        (1, "present.md")
    ]


def test_blockquote_link_and_html_tag_following_blank_line() -> None:
    assert destinations("> [real](present.md)\n") == [(1, "present.md")]
    assert destinations(
        "<pre>\n[missing](missing.md)\n</pre>\n[real](present.md)\n"
    ) == [(4, "present.md")]


def test_html_inline_tag_does_not_hide_following_link() -> None:
    assert destinations("Use <b>text</b> [real](present.md)\n") == [(1, "present.md")]


def test_html_comment_at_block_start_swallows_same_line_markdown() -> None:
    assert destinations("<!-- comment --> [not-link](missing.md)\n") == []
    assert destinations("before <!-- comment --> [real](present.md)\n") == [
        (1, "present.md")
    ]
    assert destinations("<!-- comment -->\n[real](present.md)\n") == [(2, "present.md")]


def test_type_seven_inline_html_not_block_if_interrupts_paragraph() -> None:
    assert destinations("Before\n<span>\n[real](present.md)\n</span>\n") == [
        (3, "present.md")
    ]
    assert destinations("<span>\n[not-link](missing.md)\n</span>\n") == []


@pytest.mark.parametrize("marker", ["-", "*", "+", "1.", "12.", "1)"])
def test_list_marker_indented_code_is_not_a_link(marker: str) -> None:
    assert destinations(f"{marker}     [literal](missing.md)\n") == []
    assert destinations(f"{marker}    [real](present.md)\n") == [(1, "present.md")]


def test_blank_line_ends_paragraph_and_code_span() -> None:
    # Unlike a single soft line break, the blank line ends the inline block.
    assert destinations("`\n[real](present.md)\n\n`\n") == [(2, "present.md")]


@pytest.mark.parametrize("marker", ["-", "1.", "12."])
def test_tab_indented_list_code_respects_visual_columns(marker: str) -> None:
    assert destinations(f"{marker}\t\t[literal](missing.md)\n") == []
    assert destinations(f"{marker}\t[real](present.md)\n") == [(1, "present.md")]


@pytest.mark.parametrize("marker", ["0.", "2.", "12.", "999."])
def test_non_one_ordered_marker_cannot_interrupt_paragraph(marker: str) -> None:
    assert destinations(f"Before\n{marker}     [real](present.md)\n") == [
        (2, "present.md")
    ]


def test_one_ordered_marker_may_interrupt_paragraph_as_code() -> None:
    assert destinations("Before\n001.     [literal](missing.md)\n") == []
