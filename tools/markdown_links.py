"""Read-only validation of local links across every Git-tracked Markdown file.

This checks relative paths, not remote URLs, generated GitHub heading slugs,
or the meaning of the linked document. It does not rewrite Markdown.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INLINE_LINK = re.compile(r"!?\[[^\]\n]+\]\(")
_REFERENCE = re.compile(r"^ {0,3}\[[^\]\n]+\]:[ \t]*(<[^>]+>|\S+)")
_ALLOWED_SCHEMES = {"http", "https", "mailto", "tel", "data"}


@dataclass(frozen=True)
class LinkProblem:
    source: str
    line: int
    target: str
    reason: str


def _mask_inline_code(line: str) -> str:
    """Blank matched backtick spans without changing line offsets."""
    chars = list(line)
    index = 0
    while index < len(line):
        if line[index] != "`":
            index += 1
            continue
        end = index
        while end < len(line) and line[end] == "`":
            end += 1
        marker = line[index:end]
        close = line.find(marker, end)
        if close < 0:
            index = end
            continue
        stop = close + len(marker)
        chars[index:stop] = " " * (stop - index)
        index = stop
    return "".join(chars)


def _destination(line: str, start: int) -> str | None:
    """Read a Markdown inline-link destination with escaped/balanced parens."""
    index = start
    while index < len(line) and line[index] in " \t":
        index += 1
    if index == len(line):
        return None
    if line[index] == "<":
        close = line.find(">", index + 1)
        return line[index + 1 : close] if close >= 0 else None
    begin = index
    depth = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line):
            index += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char in " \t" and depth == 0:
            break
        index += 1
    return line[begin:index]


def destinations(markdown: str) -> list[tuple[int, str]]:
    """Locate inline, image and reference-definition links outside literal text."""
    targets: list[tuple[int, str]] = []
    fence: str | None = None
    comment = False
    for number, raw_line in enumerate(markdown.splitlines(), start=1):
        line = _mask_inline_code(raw_line)
        if comment:
            end = line.find("-->")
            if end < 0:
                continue
            line = " " * (end + 3) + line[end + 3 :]
            comment = False
        fence_marker = _FENCE.match(line)
        if fence_marker:
            marker = fence_marker.group(1)
            if fence is None:
                fence = marker
            elif (
                marker[0] == fence[0]
                and len(marker) >= len(fence)
                and not line[fence_marker.end() :].strip()
            ):
                fence = None
            continue
        if fence is not None:
            continue
        while "<!--" in line:
            begin = line.index("<!--")
            end = line.find("-->", begin + 4)
            if end < 0:
                line = line[:begin]
                comment = True
                break
            line = line[:begin] + " " * (end + 3 - begin) + line[end + 3 :]
        for match in _INLINE_LINK.finditer(line):
            target = _destination(line, match.end())
            if target is not None:
                targets.append((number, target))
        reference = _REFERENCE.match(line)
        if reference is not None:
            value = reference.group(1).strip("<>")
            targets.append((number, value))
    return targets


def tracked_markdown(root: Path) -> tuple[Path, ...]:
    """Use the Git index as scope; never walk generated/untracked artifacts."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z", "--", "*.md"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = [
        Path(entry.decode("utf-8")) for entry in result.stdout.split(b"\0") if entry
    ]
    paths.sort()
    if not paths:
        raise ValueError("no tracked Markdown files")
    return tuple(paths)


def check_links(
    root: Path, markdown_paths: tuple[Path, ...]
) -> tuple[int, list[LinkProblem]]:
    """Reject broken paths, unsupported schemes and links escaping the checkout."""
    root = root.resolve()
    checked = 0
    problems: list[LinkProblem] = []
    for relative in markdown_paths:
        source = root / relative
        label = relative.as_posix()
        try:
            source.resolve().relative_to(root)
        except ValueError:
            problems.append(LinkProblem(label, 0, "", "source escapes checkout"))
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            problems.append(LinkProblem(label, 0, "", "unreadable Markdown"))
            continue
        for line, original in destinations(text):
            target = original.replace(r"\(", "(").replace(r"\)", ")")
            if not target or target.startswith("#"):
                continue
            if target.startswith("//"):
                continue
            parsed = urlsplit(target)
            if parsed.scheme:
                if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
                    problems.append(
                        LinkProblem(label, line, original, "unsupported URI scheme")
                    )
                continue
            checked += 1
            if target.startswith("/"):
                problems.append(
                    LinkProblem(label, line, original, "absolute local path")
                )
                continue
            # Decode percent-escaped paths but do not decode query/fragment text.
            path = unquote(parsed.path)
            if not path:
                continue
            resolved = (source.parent / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                problems.append(
                    LinkProblem(label, line, original, "path escapes checkout")
                )
                continue
            if not resolved.exists():
                problems.append(
                    LinkProblem(label, line, original, "missing local target")
                )
    return checked, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without edits")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    try:
        tracked = tracked_markdown(args.root)
        count, problems = check_links(args.root, tracked)
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(f"Markdown link check unavailable: {type(exc).__name__}")
    for item in problems:
        print(f"{item.source}:{item.line}: {item.reason}: {item.target}")
    print(
        f"Markdown links: {len(tracked)} tracked files, "
        f"{count} local targets, {len(problems)} problems"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
