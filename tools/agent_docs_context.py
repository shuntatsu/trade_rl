"""Render task-relevant sections of the canonical docs/AGENTS.md without side effects."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Every level-two section belongs to at least one topic. Unknown additions fail
# closed until an author explicitly routes them; no contract is silently hidden.
SECTION_GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "目的",
        "読む順序",
        "必要時参照",
        "Research Assurance Gate (#687)",
        "Git / PR integration safety",
        "更新matrix",
        "フォルダと保持ルール",
        "法務の境界",
        "実装時の原則",
        "Verification contract",
        "作業線の終了と変更の証拠",
    ),
    "guide": ("Human Guide update contract",),
    "claude": ("Claude Code host preflight (#1158)",),
    "tooling": ("Local repository tooling",),
    "coordination": ("Agent Coordination Plane",),
    "eval": ("Agent Eval protocol",),
}

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_ATX_HEADING = re.compile(r"^ {0,3}##[ \t]+(.+?)\s*$")
_HTML_COMMENT_START = re.compile(r"^ {0,3}<!--")


def _heading_positions(source: str) -> list[tuple[str, int]]:
    """Find actual level-two headings, excluding fenced code examples."""
    headings: list[tuple[str, int]] = []
    fence: tuple[str, int] | None = None
    html_comment = False
    offset = 0
    for line in source.splitlines(keepends=True):
        value = line.rstrip("\r\n")
        marker = _FENCE.match(value) if not html_comment else None
        comment_start = _HTML_COMMENT_START.match(value) if fence is None else None
        if html_comment:
            if "-->" in value:
                html_comment = False
        elif comment_start is not None:
            html_comment = "-->" not in value[comment_start.end() :]
        elif marker:
            chars = marker.group(1)
            if fence is None:
                fence = (chars[0], len(chars))
            elif (
                fence[0] == chars[0]
                and len(chars) >= fence[1]
                and not value[marker.end() :].strip()
            ):
                fence = None
        elif fence is None:
            heading = _ATX_HEADING.match(value)
            if heading is not None:
                headings.append((heading.group(1).strip(), offset))
        offset += len(line)
    if html_comment:
        raise ValueError("unclosed HTML comment")
    if fence is not None:
        raise ValueError("unclosed Markdown code fence")
    return headings


def split_sections(source: str) -> tuple[str, dict[str, str]]:
    """Validate exact section coverage and preserve their original bytes/order."""
    headings = _heading_positions(source)
    if not headings:
        raise ValueError("no level-two headings")
    sections: dict[str, str] = {}
    for index, (name, start) in enumerate(headings):
        if name in sections:
            raise ValueError(f"duplicate section: {name}")
        end = headings[index + 1][1] if index + 1 < len(headings) else len(source)
        sections[name] = source[start:end]

    expected = {name for group in SECTION_GROUPS.values() for name in group}
    unmapped = sorted(set(sections) - expected)
    missing = sorted(expected - set(sections))
    if unmapped or missing:
        raise ValueError(f"unmapped sections: {unmapped}; missing sections: {missing}")
    return source[: headings[0][1]], sections


def render(source: str, topics: tuple[str, ...]) -> str:
    """Select exact documented sections in source order, without rewording."""
    preamble, sections = split_sections(source)
    if topics == ("all",):
        return source
    if not topics or "all" in topics:
        raise ValueError("select one or more topics, or 'all' alone")
    if set(topics) - SECTION_GROUPS.keys():
        raise ValueError("unknown topic")
    selected = {name for topic in ("core", *topics) for name in SECTION_GROUPS[topic]}
    return preamble + "".join(
        body for name, body in sections.items() if name in selected
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        nargs="+",
        choices=(*SECTION_GROUPS, "all"),
        default=["core"],
        help="One or more task topics; 'all' prints the unabridged contract",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify section coverage without printing instructions",
    )
    args = parser.parse_args(argv)
    try:
        source = (Path(__file__).resolve().parents[1] / "docs" / "AGENTS.md").read_text(
            encoding="utf-8"
        )
        output = render(source, tuple(args.topic))
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(f"agent documentation routing failed: {type(exc).__name__}")
    if args.check:
        print(f"Agent section routing OK: {len(split_sections(source)[1])} sections")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
