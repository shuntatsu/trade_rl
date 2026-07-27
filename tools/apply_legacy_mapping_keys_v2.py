"""Migrate legacy encoder keys embedded in Python dictionary literals."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _offsets(text: str) -> list[int]:
    values = [0]
    for index, character in enumerate(text):
        if character == "\n":
            values.append(index + 1)
    return values


def _absolute(offsets: list[int], line: int, column: int) -> int:
    return offsets[line - 1] + column


def _extend_comma(text: str, end: int) -> int:
    cursor = end
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    if cursor < len(text) and text[cursor] == ",":
        cursor += 1
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    return cursor


def _source(text: str, offsets: list[int], node: ast.expr) -> str:
    return text[
        _absolute(offsets, node.lineno, node.col_offset) : _absolute(
            offsets, node.end_lineno, node.end_col_offset
        )
    ]


def _resolved_encoder_projection(source: str, *, legacy_key: str) -> str | None:
    direct_suffix = f".{legacy_key}"
    stripped = source.strip()
    if stripped.endswith(direct_suffix):
        return stripped[: -len(direct_suffix)] + ".observation_encoder"
    expected = "asset_set" if legacy_key == "asset_set_encoder" else "hierarchical_sequence_v2"
    match = re.fullmatch(
        rf"\(?\s*(?P<base>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.observation_encoder"
        rf"\s*==\s*['\"]{expected}['\"]\s*\)?",
        stripped,
    )
    if match is None:
        return None
    return f"{match.group('base')}.observation_encoder"


def _encoder_expression(
    text: str,
    offsets: list[int],
    encoder_entries: dict[str, tuple[ast.expr, ast.expr]],
) -> str:
    if len(encoder_entries) == 1:
        legacy_key = next(iter(encoder_entries))
        projection = _resolved_encoder_projection(
            _source(text, offsets, encoder_entries[legacy_key][1]),
            legacy_key=legacy_key,
        )
        if projection is not None:
            return projection

    expressions: dict[str, str] = {}
    for key, (_key_node, value) in encoder_entries.items():
        if isinstance(value, ast.Constant) and isinstance(value.value, bool):
            expressions[key] = repr(value.value)
        else:
            expressions[key] = _source(text, offsets, value)
    sequence = expressions.get("sequence_encoder", "False")
    asset = expressions.get("asset_set_encoder", "True")
    return (
        '"invalid_legacy_combination" '
        f"if ({sequence}) and ({asset}) else "
        f'"hierarchical_sequence_v2" if ({sequence}) else '
        f'"asset_set" if ({asset}) else "flat_mlp"'
    )


def _migrate(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not any(
        token in text
        for token in (
            '"sequence_encoder"',
            '"asset_set_encoder"',
            '"sequence_capacity"',
            '"sequence_attention_heads"',
            '"sequence_attention_layers"',
        )
    ):
        return
    tree = ast.parse(text)
    offsets = _offsets(text)
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        entries: dict[str, tuple[ast.expr, ast.expr]] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                entries[key.value] = (key, value)
        encoder_entries = {
            key: value
            for key, value in entries.items()
            if key in {"sequence_encoder", "asset_set_encoder"}
        }
        if encoder_entries:
            ordered = sorted(
                encoder_entries.values(),
                key=lambda pair: (pair[0].lineno, pair[0].col_offset),
            )
            first_key, first_value = ordered[0]
            start = _absolute(offsets, first_key.lineno, first_key.col_offset)
            end = _absolute(offsets, first_value.end_lineno, first_value.end_col_offset)
            expression = _encoder_expression(text, offsets, encoder_entries)
            edits.append((start, end, f'"observation_encoder": {expression}'))
            for key, value in ordered[1:]:
                start = _absolute(offsets, key.lineno, key.col_offset)
                end = _absolute(offsets, value.end_lineno, value.end_col_offset)
                edits.append((start, _extend_comma(text, end), ""))
        rename = {"sequence_capacity": "sequence_tcn_capacity"}
        duplicate = {
            "sequence_attention_heads": (
                "sequence_timeframe_attention_heads",
                "sequence_asset_attention_heads",
            ),
            "sequence_attention_layers": (
                "sequence_timeframe_attention_layers",
                "sequence_asset_attention_layers",
            ),
        }
        for old, new in rename.items():
            if old not in entries:
                continue
            key, _ = entries[old]
            start = _absolute(offsets, key.lineno, key.col_offset)
            end = _absolute(offsets, key.end_lineno, key.end_col_offset)
            edits.append((start, end, repr(new)))
        for old, (first, second) in duplicate.items():
            if old not in entries:
                continue
            key, value = entries[old]
            start = _absolute(offsets, key.lineno, key.col_offset)
            end = _absolute(offsets, value.end_lineno, value.end_col_offset)
            source = _source(text, offsets, value)
            indent = " " * key.col_offset
            edits.append(
                (
                    start,
                    end,
                    f"{first!r}: {source},\n{indent}{second!r}: {source}",
                )
            )
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for root in (ROOT / "trade_rl", ROOT / "tests", ROOT / "examples", ROOT / "tools"):
        for path in root.rglob("*.py"):
            if path.name.startswith("apply_") and path.parent.name == "tools":
                continue
            _migrate(path)


if __name__ == "__main__":
    main()
