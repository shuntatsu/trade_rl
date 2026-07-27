"""Patch the branch-local migration tool before executing it.

The generated source is restored before committing production changes, so this helper
only protects the CI migration process itself.
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET = Path(__file__).with_name("apply_hierarchical_sequence_v2.py")
text = TARGET.read_text(encoding="utf-8")
pattern = re.compile(
    r"    replacements = \{\n.*?"
    r"        label=\"asset transformer config\",\n"
    r"    \)\n",
    re.DOTALL,
)
replacement = '''    shared_transformer_config = (
        "            heads=architecture.attention_heads,\\n"
        "            layers=architecture.attention_layers,\\n"
        "            ffn_multiplier=architecture.attention_ffn_multiplier,\\n"
        "            dropout=architecture.dropout,\\n"
        "            gate_bias=architecture.attention_gate_bias,\\n"
    )
    if text.count(shared_transformer_config) != 2:
        raise RuntimeError(
            "expected exactly two shared transformer configuration blocks"
        )
    text = text.replace(
        shared_transformer_config,
        "            heads=architecture.timeframe_attention_heads,\\n"
        "            layers=architecture.timeframe_attention_layers,\\n"
        "            ffn_multiplier=architecture.timeframe_ffn_multiplier,\\n"
        "            dropout=architecture.dropout,\\n"
        "            gate_bias=architecture.timeframe_gate_bias,\\n",
        1,
    )
    text = text.replace(
        shared_transformer_config,
        "            heads=architecture.asset_attention_heads,\\n"
        "            layers=architecture.asset_attention_layers,\\n"
        "            ffn_multiplier=architecture.asset_ffn_multiplier,\\n"
        "            dropout=architecture.dropout,\\n"
        "            gate_bias=architecture.asset_gate_bias,\\n",
        1,
    )
'''
updated, count = pattern.subn(lambda _match: replacement, text, count=1)
if count != 1:
    raise RuntimeError("could not patch duplicate transformer migration block")
TARGET.write_text(updated, encoding="utf-8")
