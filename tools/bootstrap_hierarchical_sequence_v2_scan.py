"""Make the migration's legacy scan reject API syntax, not descriptive test names."""

from __future__ import annotations

import re
from pathlib import Path

TARGET = Path(__file__).with_name("apply_hierarchical_sequence_v2.py")
text = TARGET.read_text(encoding="utf-8")
pattern = re.compile(
    r"def _assert_legacy_contract_removed\(\) -> None:\n.*?\n\ndef main\(\) -> None:",
    re.DOTALL,
)
replacement = r'''def _assert_legacy_contract_removed() -> None:
    forbidden = (
        re.compile(r"\bsequence_encoder\s*[:=]"),
        re.compile(r"\basset_set_encoder\s*[:=]"),
        re.compile(r"\.sequence_encoder\b"),
        re.compile(r"\.asset_set_encoder\b"),
        re.compile(r"\bsequence_capacity\s*[:=]"),
        re.compile(r"\bsequence_attention_heads\s*[:=]"),
        re.compile(r"\bsequence_attention_layers\s*[:=]"),
        re.compile(r'"sequence_encoder"\s*:'),
        re.compile(r'"asset_set_encoder"\s*:'),
        re.compile(r'"sequence_capacity"\s*:'),
        re.compile(r'"sequence_attention_heads"\s*:'),
        re.compile(r'"sequence_attention_layers"\s*:'),
    )
    roots = (
        ROOT / "trade_rl",
        ROOT / "tests",
        ROOT / "examples/binance-multitimeframe",
    )
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".json"}:
                continue
            source = path.read_text(encoding="utf-8")
            for expression in forbidden:
                match = expression.search(source)
                if match is not None:
                    violations.append(
                        f"{path.relative_to(ROOT)}: {match.group(0)}"
                    )
    if violations:
        raise RuntimeError("legacy configuration remains:\n" + "\n".join(violations))


def main() -> None:'''
updated, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise RuntimeError("could not patch syntax-aware legacy scan")
TARGET.write_text(updated, encoding="utf-8")
