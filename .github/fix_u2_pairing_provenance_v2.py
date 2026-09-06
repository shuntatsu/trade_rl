from pathlib import Path

path = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = path.read_text(encoding="utf-8")

old = '''        expected_cell = _U2_PAIRED_REPLAY_WINDOW_CELLS.get(self.source_window)
        if expected_cell is None or self.cell != expected_cell:
            raise ValueError("U2 paired replay window/cell identity is invalid")
'''
new = '''        if not isinstance(self.source_window, str) or not self.source_window:
            raise ValueError("U2 paired replay source window must be non-empty")
        if not isinstance(self.cell, str) or not self.cell:
            raise ValueError("U2 paired replay cell must be non-empty")
'''
if text.count(old) != 1:
    raise SystemExit(f"paired scope responsibility anchor count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''            paired_scope_evidence = tuple(
                sorted(paired_scope_evidence, key=lambda pair: pair.identity)
            )
'''
new = '''            paired_scope_evidence = tuple(
                sorted(
                    paired_scope_evidence,
                    key=lambda pair: (
                        U2_DEVELOPMENT_WINDOWS.index(pair.source_window),
                        pair.identity,
                    ),
                )
            )
'''
if text.count(old) != 1:
    raise SystemExit(f"paired canonical-order anchor count={text.count(old)}")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
