from __future__ import annotations

from pathlib import Path

path = Path("examples/binance-multitimeframe/run_full_research.py")
text = path.read_text(encoding="utf-8")
text = text.replace("13_104", "13_128")
text = text.replace("13,104", "13,128")
path.write_text(text, encoding="utf-8")
