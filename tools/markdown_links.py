"""Read-only validation of local links across every Git-tracked Markdown file.

This checks relative paths, not remote URLs, generated GitHub heading slugs,
or the meaning of the linked document. It does not rewrite Markdown.
"""

from __future__ import annotations

import argparse
import re
import string
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_QUOTE = re.compile(r"^ {0,3}(?:>[ \t]?)+")
_LIST_INDENTED_CODE = re.compile(r"^ {0,3}(?:[-+*]|(?P<number>[0-9]{1,9})[.)]) {5,}\S")
_HTML_START = re.compile(r"^ {0,3}</?([A-Za-z][A-Za-z0-9-]*)(?:[ \t]|/?>)")
_HTML_STANDALONE = re.compile(
    r"^ {0,3}</?[A-Za-z][A-Za-z0-9-]*(?:[ \t]+[^<>]*)?/?>[ \t]*$"
)
_HTML_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "base",
    "basefont",
    "blockquote",
    "body",
    "caption",
    "center",
    "col",
    "colgroup",
    "dd",
    "details",
    "dialog",
    "dir",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "frame",
    "frameset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hgroup",
    "hr",
    "html",
    "iframe",
    "legend",
    "li",
    "link",
    "main",
    "menu",
    "menuitem",
    "nav",
    "noframes",
    "ol",
    "optgroup",
    "option",
    "p",
    "param",
    "search",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "title",
    "tr",
    "track",
    "ul",
}
_HTML_RAW_TAGS = {"pre", "script", "style", "textarea"}
_REFERENCE = re.compile(r"^ {0,3}\[([^\]\n]+)\]:[ \t]*(<[^>]+>|\S+)")
_IMAGE_REFERENCE = re.compile(r"\[([^\[\]\r\n]+)\]")
_INLINE_HEADING = re.compile(r" {0,3}#{1,6}(?:[ \t]|$)")
_INLINE_RULE = re.compile(
    r" {0,3}(?:=+|-+|(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})[ \t]*$"
)
_INLINE_LIST_START = re.compile(r" {0,3}(?:[-+*]|0{0,8}1[.)])[ \t]+\S")
_ALLOWED_SCHEMES = {"http", "https", "mailto", "tel"}


@dataclass(frozen=True)
class LinkProblem:
    source: str
    line: int
    target: str
    reason: str


def _blank(text: str) -> str:
    """Mask literal content but keep line separators for source locations."""
    return "".join(char if char in "\r\n" else " " for char in text)


def _mask_block_code(markdown: str) -> str:
    """Mask Markdown fenced/indented code and raw HTML block bodies.

    Preserve original bytes on unmasked lines, and line separators everywhere.
    Blockquote prefixes are inspected only to determine literal-block context.
    """
    output: list[str] = []
    fence: str | None = None
    indented = False
    paragraph = False
    html_block = False
    html_comment = False
    html_raw_tag: str | None = None
    quote_level = 0
    for raw in markdown.splitlines(keepends=True):
        value = raw.rstrip("\r\n")
        quote = _QUOTE.match(value)
        current_quote_level = quote.group().count(">") if quote else 0
        if current_quote_level != quote_level:
            fence = None
            indented = False
            paragraph = False
            html_block = False
            html_comment = False
            html_raw_tag = None
            quote_level = current_quote_level
        logical = value[quote.end() :] if quote else value
        marker = _FENCE.match(logical)
        if html_comment:
            if "-->" in logical:
                html_comment = False
            output.append(_blank(raw))
            paragraph = False
            continue
        if fence is not None:
            if marker is not None and (
                marker.group(1)[0] == fence[0]
                and len(marker.group(1)) >= len(fence)
                and not logical[marker.end() :].strip(" \t")
            ):
                fence = None
            output.append(_blank(raw))
            paragraph = False
            continue
        if html_block:
            if html_raw_tag is not None:
                if re.search(rf"</{html_raw_tag}[ \t]*>", logical, re.IGNORECASE):
                    html_block = False
                    html_raw_tag = None
            elif not logical.strip():
                html_block = False
                output.append(raw)
                paragraph = False
                continue
            output.append(_blank(raw))
            paragraph = False
            continue
        if marker is not None and (
            marker.group(1)[0] != "`" or "`" not in logical[marker.end() :]
        ):
            fence = marker.group(1)
            output.append(_blank(raw))
            indented = False
            paragraph = False
            continue
        if not logical.strip():
            output.append(raw)
            paragraph = False
            continue
        if re.match(r"^ {0,3}<!--", logical):
            html_comment = "-->" not in logical
            output.append(_blank(raw))
            paragraph = False
            continue
        html_start = _HTML_START.match(logical)
        if html_start is not None and (
            html_start.group(1).lower() in _HTML_BLOCK_TAGS
            or html_start.group(1).lower() in _HTML_RAW_TAGS
            or (not paragraph and _HTML_STANDALONE.match(logical) is not None)
        ):
            tag = html_start.group(1).lower()
            html_raw_tag = tag if tag in _HTML_RAW_TAGS else None
            html_block = not (
                html_raw_tag is not None
                and re.search(rf"</{html_raw_tag}[ \t]*>", logical, re.IGNORECASE)
            )
            output.append(_blank(raw))
            paragraph = False
            continue
        list_code = _LIST_INDENTED_CODE.match(logical.expandtabs(4))
        if list_code is not None and (
            not paragraph
            or list_code.group("number") is None
            or int(list_code.group("number")) == 1
        ):
            output.append(_blank(raw))
            indented = True
            paragraph = False
            continue
        if (logical.startswith("    ") or logical.startswith("\t")) and (
            indented or not paragraph
        ):
            output.append(_blank(raw))
            indented = True
            paragraph = False
            continue
        indented = False
        output.append(raw)
        paragraph = re.match(r"^ {0,3}#{1,6}(?:[ \t]|$)", logical) is None
    return "".join(output)


def _escaped(source: str, index: int) -> bool:
    count = 0
    index -= 1
    while index >= 0 and source[index] == "\\":
        count += 1
        index -= 1
    return count % 2 == 1


def _mask_inline_code(source: str) -> str:
    """Mask paired CommonMark backtick spans, including multiline spans."""
    chars = list(source)
    index = 0
    while index < len(source):
        if source[index] != "`" or _escaped(source, index):
            index += 1
            continue
        end = index
        while end < len(source) and source[end] == "`":
            end += 1
        marker = source[index:end]
        search = end
        while True:
            close = source.find(marker, search)
            if close < 0 or re.search(r"\n[ \t]*\n", source[end:close]):
                index = end
                break
            stop = close + len(marker)
            if (close > 0 and source[close - 1] == "`") or (
                stop < len(source) and source[stop] == "`"
            ):
                search = close + 1
                continue
            for char_index in range(index, stop):
                if chars[char_index] not in "\r\n":
                    chars[char_index] = " "
            index = stop
            break
    return "".join(chars)


def _mask_comments(source: str) -> str:
    """Ignore links embedded in HTML comments outside code spans."""
    chars = list(source)
    start = 0
    while True:
        begin = source.find("<!--", start)
        if begin < 0:
            break
        if _escaped(source, begin):
            start = begin + 4
            continue
        end = source.find("-->", begin + 4)
        stop = len(source) if end < 0 else end + 3
        for index in range(begin, stop):
            if chars[index] not in "\r\n":
                chars[index] = " "
        start = stop
    return "".join(chars)


def _destination(source: str, start: int) -> tuple[str, int] | None:
    """Parse an inline destination and return its exclusive consumed end."""
    index = start
    while index < len(source) and source[index] in " \t\r\n":
        index += 1
    if index == len(source):
        return None
    if source[index] == "<":
        begin = index + 1
        index = begin
        while index < len(source):
            char = source[index]
            if char in "\r\n" or (char == "<" and not _escaped(source, index)):
                return None
            if char == ">" and not _escaped(source, index):
                break
            index += 1
        if index == len(source):
            return None
        target = source[begin:index]
        index += 1
    else:
        begin = index
        depth = 0
        while index < len(source):
            char = source[index]
            if (
                char == "\\"
                and index + 1 < len(source)
                and source[index + 1] in string.punctuation
            ):
                index += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            elif char in " \t\r\n":
                if depth:
                    return None
                break
            elif ord(char) < 32 or ord(char) == 127:
                return None
            index += 1
        target = source[begin:index]
        if depth != 0:
            return None
    destination_end = index
    while index < len(source) and source[index] in " \t\r\n":
        index += 1
    if index < len(source) and source[index] in ('"', "'", "("):
        if index == destination_end:
            return None
        opener = source[index]
        closer = ")" if opener == "(" else opener
        index += 1
        title_start = index
        while index < len(source):
            char = source[index]
            if char == closer and not _escaped(source, index):
                break
            if opener == "(" and char == "(" and not _escaped(source, index):
                return None
            index += 1
        if index == len(source) or re.search(
            r"\r?\n[ \t]*\r?\n", source[title_start:index]
        ):
            return None
        index += 1
        while index < len(source) and source[index] in " \t\r\n":
            index += 1
    if (
        index >= len(source)
        or source[index] != ")"
        or re.search(r"\r?\n[ \t]*\r?\n", source[start:index])
    ):
        return None
    return target, index + 1


def _inline_targets(
    markdown: str,
    text: str,
    known_references: frozenset[str] | None,
) -> tuple[list[tuple[int, str]], str]:
    """Scan inline candidates, retaining alt contents until an image is recognized.

    The reference-discovery pass provisionally recognizes explicit image
    references. Its targets are discarded; only its masked definition text
    is used. The final pass requires a discovered reference definition.
    """
    targets: list[tuple[int, str]] = []
    # is_image, contains_rendered_link, number of targets before the opener
    stack: list[tuple[bool, bool, int]] = []
    reference_text = list(text)
    consumed_end = 0
    previous_single_line_block = False
    previous_quote_level = 0
    for index, char in enumerate(text):
        if index < consumed_end:
            continue
        if index == 0 or markdown[index - 1] == "\n":
            line_end = markdown.find("\n", index)
            line = markdown[index : len(markdown) if line_end < 0 else line_end]
            line = line.rstrip("\r")
            quote = _QUOTE.match(line)
            quote_level = quote.group().count(">") if quote else 0
            logical = line[quote.end() :] if quote else line
            single_line_block = (
                _INLINE_HEADING.match(logical) is not None
                or _INLINE_RULE.fullmatch(logical) is not None
            )
            if (
                previous_single_line_block
                or single_line_block
                or not logical.strip(" \t")
                or quote_level > previous_quote_level
                or _INLINE_LIST_START.match(logical)
            ):
                stack.clear()
            previous_single_line_block = single_line_block
            previous_quote_level = quote_level
        if char == "[" and not _escaped(text, index):
            is_image = (
                index > 0 and text[index - 1] == "!" and not _escaped(text, index - 1)
            )
            stack.append((is_image, False, len(targets)))
        elif char == "]" and stack and not _escaped(text, index):
            is_image, nested_link, first_target = stack.pop()
            parsed = (
                _destination(text, index + 2)
                if index + 1 < len(text) and text[index + 1] == "("
                else None
            )
            rendered_link = False
            if parsed is not None and (is_image or not nested_link):
                target, consumed_end = parsed
                reference_text[index + 1 : consumed_end] = _blank(
                    text[index + 1 : consumed_end]
                )
                if is_image:
                    # An actual image renders the label as alt text, not links.
                    del targets[first_target:]
                targets.append((text.count("\n", 0, index) + 1, target))
                rendered_link = not is_image
            elif is_image:
                reference = _IMAGE_REFERENCE.match(text, index + 1)
                if reference is not None and (
                    known_references is None
                    or " ".join(reference.group(1).split()).casefold()
                    in known_references
                ):
                    # The separate reference-definition scan checks its target.
                    del targets[first_target:]
                    consumed_end = reference.end()
                else:
                    # A literal/unfinished image label cannot hide real links.
                    rendered_link = nested_link
            if rendered_link:
                # Nested links invalidate enclosing link labels, but a
                # recognized image consumes its alt contents at its own close.
                for parent in range(len(stack) - 1, -1, -1):
                    image, _, target_start = stack[parent]
                    stack[parent] = (image, True, target_start)
                    if image:
                        break
    return targets, "".join(reference_text)


def destinations(markdown: str) -> list[tuple[int, str]]:
    """Find local-link candidates outside literal code and HTML comments."""
    text = _mask_comments(_mask_inline_code(_mask_block_code(markdown)))
    _, reference_text = _inline_targets(markdown, text, None)
    references = frozenset(
        " ".join(reference.group(1).split()).casefold()
        for line in reference_text.splitlines()
        if (reference := _REFERENCE.match(line)) is not None
    )
    targets, reference_text = _inline_targets(markdown, text, references)
    for number, line in enumerate(reference_text.splitlines(), start=1):
        reference = _REFERENCE.match(line)
        if reference is not None:
            targets.append((number, reference.group(2).strip("<>")))
    return sorted(targets, key=lambda item: item[0])


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


def _markdown_unescape_url(destination: str) -> str:
    """Unescape CommonMark ASCII punctuation without losing literal #, ? or %."""
    characters: list[str] = []
    index = 0
    while index < len(destination):
        char = destination[index]
        if (
            char == "\\"
            and index + 1 < len(destination)
            and destination[index + 1] in string.punctuation
        ):
            escaped = destination[index + 1]
            # Percent-encode reserved punctuation before urlsplit; unquote
            # later restores literal filesystem names without re-parsing them.
            characters.append(f"%{ord(escaped):02X}" if escaped in "#?%" else escaped)
            index += 2
        else:
            characters.append(char)
            index += 1
    return "".join(characters)


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
            target = _markdown_unescape_url(original)
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
