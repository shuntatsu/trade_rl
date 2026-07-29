from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, *, field: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise RuntimeError(f"{field} fragment was not found")
    return text.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: task9_remediate_extractor.py ROOT")
    root = Path(sys.argv[1]).resolve()
    path = root / "trade_rl/rl/policies.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from torch import nn\n",
        "from torch import nn\n\nfrom trade_rl.rl.observations import CURRENT_WEIGHT_SOURCE\n",
        field="current weight source import",
    )
    text = replace_once(
        text,
        '''        global_width: int,
        n_symbols: int,
        sequence_tcn_capacity: str = "standard",
''',
        '''        global_width: int,
        n_symbols: int,
        current_weight_source: str = CURRENT_WEIGHT_SOURCE,
        current_weight_shape: tuple[int, ...] | None = None,
        sequence_tcn_capacity: str = "standard",
''',
        field="sequence extractor current weight arguments",
    )
    text = replace_once(
        text,
        '''        timeframes = ("15m", "1h", "4h", "1d")
        if tuple(feature_counts) != timeframes or tuple(window_lengths) != timeframes:
''',
        '''        timeframes = ("15m", "1h", "4h", "1d")
        if current_weight_source != CURRENT_WEIGHT_SOURCE:
            raise ValueError("sequence current weight source is unsupported")
        resolved_current_weight_shape = (
            (n_symbols,) if current_weight_shape is None else tuple(current_weight_shape)
        )
        if resolved_current_weight_shape != (n_symbols,):
            raise ValueError("sequence current weight shape does not match symbols")
        if tuple(feature_counts) != timeframes or tuple(window_lengths) != timeframes:
''',
        field="sequence extractor current weight validation",
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
