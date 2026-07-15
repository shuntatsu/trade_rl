from __future__ import annotations

from pathlib import Path

path = Path("trade_rl/simulation/accounting.py")
text = path.read_text(encoding="utf-8")
old = '''        self._refresh_economic_state()
        if not self.insolvent and self.peak_value + _TOLERANCE < self.portfolio_value:
            raise ValueError("peak_value cannot be below portfolio_value")
'''
new = '''        self._refresh_economic_state()
        portfolio_value = self.portfolio_value
        comparison_tolerance = max(
            _TOLERANCE,
            abs(portfolio_value) * 1e-12,
            abs(self.peak_value) * 1e-12,
        )
        if (
            not self.insolvent
            and self.peak_value + comparison_tolerance < portfolio_value
        ):
            raise ValueError("peak_value cannot be below portfolio_value")
'''
if new not in text:
    if old not in text:
        raise RuntimeError("BookState peak invariant boundary was not found")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
