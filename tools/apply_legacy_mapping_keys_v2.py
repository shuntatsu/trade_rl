"""Migrate legacy encoder keys embedded in Python dictionary literals."""

from __future__ import annotations

import ast
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


def _literal_bool(node: ast.expr, *, path: Path) -> bool:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, bool):
        raise RuntimeError(f"legacy encoder mapping must use booleans: {path}")
    return node.value


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
            sequence = (
                _literal_bool(encoder_entries["sequence_encoder"][1], path=path)
                if "sequence_encoder" in encoder_entries
                else False
            )
            asset = (
                _literal_bool(encoder_entries["asset_set_encoder"][1], path=path)
                if "asset_set_encoder" in encoder_entries
                else True
            )
            if sequence and asset:
                encoder = "invalid_legacy_combination"
            elif sequence:
                encoder = "hierarchical_sequence_v2"
            elif asset:
                encoder = "asset_set"
            else:
                encoder = "flat_mlp"
            ordered = sorted(
                encoder_entries.values(),
                key=lambda pair: (pair[0].lineno, pair[0].col_offset),
            )
            first_key, first_value = ordered[0]
            start = _absolute(offsets, first_key.lineno, first_key.col_offset)
            end = _absolute(offsets, first_value.end_lineno, first_value.end_col_offset)
            quote = '"'
            edits.append(
                (start, end, f'{quote}observation_encoder{quote}: {quote}{encoder}{quote}')
            )
            for key, value in ordered[1:]:
                start = _absolute(offsets, key.lineno, key.col_offset)
                end = _absolute(offsets, value.end_lineno, value.end_col_offset)
                edits.append((start, _extend_comma(text, end), ""))
        rename = {
            "sequence_capacity": "sequence_tcn_capacity",
        }
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
            value_start = _absolute(offsets, value.lineno, value.col_offset)
            value_end = _absolute(offsets, value.end_lineno, value.end_col_offset)
            source = text[value_start:value_end]
            indent = " " * key.col_offset
            edits.append(
                (
                    start,
                    end,
                    f'{first!r}: {source},\n{indent}{second!r}: {source}',
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
