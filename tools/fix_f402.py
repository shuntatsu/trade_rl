from pathlib import Path

path = Path("trade_rl/rl/actions.py")
text = path.read_text(encoding="utf-8")
old = "for field, value in ("
if old not in text:
    raise RuntimeError("ResidualAction shadowing loop was not found")
text = text.replace(old, "for field_name, value in (", 1)
text = text.replace('f"{field} must be finite"', 'f"{field_name} must be finite"', 1)
text = text.replace(
    'f"{field} must be within [-1, 1]"',
    'f"{field_name} must be within [-1, 1]"',
    1,
)
path.write_text(text, encoding="utf-8")
