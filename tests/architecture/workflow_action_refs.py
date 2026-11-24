"""Shared bounded source grammar for setup-uv action pin contracts.

This is not a YAML parser. Literal setup-uv references must occupy one
supported block-mapping uses line; unsupported literal forms fail closed.
The version contract separately verifies the action's own with: block.
"""

from __future__ import annotations

import re
from pathlib import Path

_SETUP_UV_LINE = re.compile(
    r"(?m)^(?P<indent> +)(?P<dash>-[ ]+)?"
    r"""(?:uses|'uses'|"uses")[ ]*:[ ]+(?P<quote>['"]?)"""
    r"""(?i:astral-sh/setup-uv)@(?P<ref>[^\s#'"\\]+)(?P=quote)"""
    r"(?:[ \t]+#[^\n]*)?[ \t]*$"
)
_SETUP_UV_LITERAL = re.compile(r"astral-sh/setup-uv", re.IGNORECASE)


def setup_uv_action_matches(source: str, path: Path) -> list[re.Match[str]]:
    """Inventory plain/quoted scalar refs, rejecting other literal spellings."""
    matches = list(_SETUP_UV_LINE.finditer(source))
    matched_lines = {match.start() for match in matches}
    offset = 0
    for line in source.splitlines(keepends=True):
        if not line.lstrip().startswith("#") and _SETUP_UV_LITERAL.search(line):
            assert offset in matched_lines, (
                f"{path}: unsupported setup-uv action syntax"
            )
        offset += len(line)
    return matches
