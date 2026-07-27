"""Teach the branch-local migration tool to translate parameterized booleans."""

from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).with_name("apply_hierarchical_sequence_v2.py")
text = TARGET.read_text(encoding="utf-8")
old = '''        if old:
            values: dict[str, bool] = {}
            for item in old:
                if not isinstance(item.value, ast.Constant) or not isinstance(
                    item.value.value, bool
                ):
                    raise RuntimeError(
                        f"non-literal legacy encoder keyword in {path}:{item.lineno}"
                    )
                assert item.arg is not None
                values[item.arg] = item.value.value
            sequence = values.get("sequence_encoder", False)
            asset = values.get("asset_set_encoder", True)
            if sequence and asset:
                encoder = "invalid_legacy_combination"
            elif sequence:
                encoder = SEQUENCE_ENCODER
            elif asset:
                encoder = "asset_set"
            else:
                encoder = "flat_mlp"
            ordered = sorted(
                old,
                key=lambda item: (item.lineno, item.col_offset),
            )
            first = ordered[0]
            start = _absolute(offsets, first.lineno, first.col_offset)
            end = _absolute(offsets, first.end_lineno, first.end_col_offset)
            edits.append((start, end, f'observation_encoder="{encoder}"'))
            for item in ordered[1:]:
                start = _absolute(offsets, item.lineno, item.col_offset)
                end = _absolute(offsets, item.end_lineno, item.end_col_offset)
                edits.append((start, _extend_through_comma(text, end), ""))
'''
new = '''        if old:
            expressions: dict[str, str] = {}
            for item in old:
                assert item.arg is not None
                if isinstance(item.value, ast.Constant) and isinstance(
                    item.value.value, bool
                ):
                    expressions[item.arg] = repr(item.value.value)
                else:
                    value_start = _absolute(
                        offsets, item.value.lineno, item.value.col_offset
                    )
                    value_end = _absolute(
                        offsets, item.value.end_lineno, item.value.end_col_offset
                    )
                    expressions[item.arg] = text[value_start:value_end]
            sequence = expressions.get("sequence_encoder", "False")
            # Calls that toggle sequence_encoder without an explicit asset flag
            # historically use a flat/non-sequence fallback.  Explicit asset flags
            # retain their exact old state-machine semantics.
            asset = expressions.get("asset_set_encoder", "False")
            encoder_expression = (
                '"invalid_legacy_combination" '
                f'if ({sequence}) and ({asset}) else '
                f'"{SEQUENCE_ENCODER}" if ({sequence}) else '
                f'"asset_set" if ({asset}) else "flat_mlp"'
            )
            ordered = sorted(
                old,
                key=lambda item: (item.lineno, item.col_offset),
            )
            first = ordered[0]
            start = _absolute(offsets, first.lineno, first.col_offset)
            end = _absolute(offsets, first.end_lineno, first.end_col_offset)
            edits.append(
                (start, end, f"observation_encoder=({encoder_expression})")
            )
            for item in ordered[1:]:
                start = _absolute(offsets, item.lineno, item.col_offset)
                end = _absolute(offsets, item.end_lineno, item.end_col_offset)
                edits.append((start, _extend_through_comma(text, end), ""))
'''
count = text.count(old)
if count != 1:
    raise RuntimeError(f"expected one encoder keyword migration block, found {count}")
TARGET.write_text(text.replace(old, new, 1), encoding="utf-8")
