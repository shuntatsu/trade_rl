from __future__ import annotations

from pathlib import Path

path = Path("trade_rl/integrations/binance.py")
text = path.read_text(encoding="utf-8")
old = '''        else:
            one_day = max(2, int(round(24.0 / native_hours)))
            features.append(
                FeatureSpec(
                    name=f"{timeframe}__realized_volatility_{one_day}bar",
                    kind=FeatureKind.REALIZED_VOLATILITY,
                    timeframe=timeframe,
                    lookback=one_day,
                    max_staleness_hours=staleness,
                )
            )
'''
new = '''        else:
            volatility_lookback = 4 if timeframe == "15m" else max(
                2, int(round(24.0 / native_hours))
            )
            features.append(
                FeatureSpec(
                    name=(
                        f"{timeframe}__realized_volatility_"
                        f"{volatility_lookback}bar"
                    ),
                    kind=FeatureKind.REALIZED_VOLATILITY,
                    timeframe=timeframe,
                    lookback=volatility_lookback,
                    max_staleness_hours=staleness,
                )
            )
'''
if new not in text:
    if old not in text:
        raise RuntimeError("expected Binance multi-timeframe preset was not found")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
