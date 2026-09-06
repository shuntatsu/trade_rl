from pathlib import Path

path = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = path.read_text(encoding="utf-8")
old = '''        wealth = tuple(
            _positive_wealth(value, field="U2 Development robustness seed wealth")
            if value <= 0.0
            else _finite(value, field="U2 Development robustness seed wealth")
            for value in self.seed_symbol_balanced_net_wealth
        )
'''
new = '''        wealth = tuple(
            _finite(value, field="U2 Development robustness seed wealth")
            for value in self.seed_symbol_balanced_net_wealth
        )
'''
if old not in text:
    raise SystemExit("nonpositive wealth patch anchor missing")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
