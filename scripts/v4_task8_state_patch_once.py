from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/learning/causal_alpha_v4.py")
text = path.read_text(encoding="utf-8")
old = '''        states = np.full(volatility.shape, V4ForecastState.NORMAL, dtype=object)\n        high_volatility = volatility >= self.high_realized_volatility_threshold\n        low_liquidity = liquidity_values <= self.low_liquidity_threshold\n        positioning_stress = (\n            np.abs(stress) >= self.basis_positioning_stress_threshold\n        )\n        states[high_volatility] = V4ForecastState.HIGH_REALIZED_VOLATILITY\n        states[low_liquidity] = V4ForecastState.LOW_LIQUIDITY\n        states[positioning_stress] = V4ForecastState.BASIS_POSITIONING_STRESS\n        return states\n'''
new = '''        states = np.empty(volatility.shape, dtype=object)\n        for row in range(volatility.size):\n            states[row] = V4ForecastState.NORMAL\n        high_volatility = volatility >= self.high_realized_volatility_threshold\n        low_liquidity = liquidity_values <= self.low_liquidity_threshold\n        positioning_stress = (\n            np.abs(stress) >= self.basis_positioning_stress_threshold\n        )\n        for row in np.flatnonzero(high_volatility):\n            states[row] = V4ForecastState.HIGH_REALIZED_VOLATILITY\n        for row in np.flatnonzero(low_liquidity):\n            states[row] = V4ForecastState.LOW_LIQUIDITY\n        for row in np.flatnonzero(positioning_stress):\n            states[row] = V4ForecastState.BASIS_POSITIONING_STRESS\n        return states\n'''
if text.count(old) != 1:
    raise SystemExit(f"expected one V4 state block, got {text.count(old)}")
text = text.replace(old, new, 1)
old_mask = "            mask = positive & (states == state)\n"
new_mask = '''            state_mask = np.fromiter(\n                (value is state for value in states),\n                dtype=np.bool_,\n                count=rows,\n            )\n            mask = positive & state_mask\n'''
if text.count(old_mask) != 1:
    raise SystemExit(f"expected one V4 state mask, got {text.count(old_mask)}")
text = text.replace(old_mask, new_mask, 1)
path.write_text(text, encoding="utf-8")
