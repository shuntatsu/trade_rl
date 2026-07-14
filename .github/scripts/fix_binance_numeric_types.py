from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/integrations/binance.py")
text = path.read_text(encoding="utf-8")
old = '''def _finite_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
'''
new = '''def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"invalid {field}: {value!r}")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"invalid {field}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
'''
if new not in text:
    if old not in text:
        raise RuntimeError("expected _finite_float implementation was not found")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
