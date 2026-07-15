# Native Multi-Timeframe Research

Trade RL treats multi-timeframe market context as a first-class causal dataset contract. The maintained complete Binance example calculates every feature on its own closed native clock, then performs availability-aware as-of alignment onto the one-hour decision clock. No backward filling or future Ichimoku shift is permitted.

## Complete research contract

The maintained example uses:

- decision clock: `1h`;
- native feature clocks: `15m`, `1h`, `4h`, and `1d`;
- instruments: `BTCUSDT`, `ETHUSDT`, and `BNBUSDT` USDⓈ-M futures;
- fixed closed range: `2024-12-01T00:00:00Z` through `2026-06-01T00:00:00Z`;
- expected hourly decisions: `13,128`;
- market features: `24` per native clock, `96` total;
- action layout: fast tilt, slow tilt, and three net-zero relative-asset factors;
- risk tilt: disabled, so the policy cannot learn the all-cash escape action;
- full PPO seeds: `0`, `1`, and `2`;
- full PPO timesteps: `262,144` per seed;
- nested walk-forward: two outer folds, three seeds, `65,536` candidate timesteps per seed and fold.

## Features on each native clock

Each of the four clocks contributes the same 24-feature contract:

1. one-bar, four-bar, and 24-bar log returns;
2. four-bar and 24-bar realized volatility;
3. 24-bar volume z-score and causal funding basis points;
4. RSI(14);
5. MACD line, signal, and histogram using 12/26/9;
6. Bollinger position and bandwidth using 20 bars and two standard deviations;
7. ATR percentage and ADX using 14 bars;
8. Stochastic %K(14) and %D(14,3);
9. CCI(20), Williams %R(14), and 24-bar OBV slope;
10. Ichimoku Tenkan distance, Kijun distance, cloud position, and cloud thickness using 9/26/52.

The Ichimoku features deliberately do not use a forward-shifted Senkou value or a backward-shifted Chikou span. Those charting conventions would either require future information or duplicate historical close information. The maintained representation uses only information available at the current native bar close.

A four-hour or daily feature remains unavailable before every source bar in its calculation window is available. The feature value, availability mask, age, staleness, and missing-reason fields are all bound into the dataset and observation identities.

## Running the complete example

```bash
uv sync --extra dev --extra train-sb3
uv run python examples/binance-multitimeframe/run_full_research.py \
  --work-root var/binance-multitimeframe-full
```

The runner downloads or reuses immutable Binance Vision archives, records the exchange metadata source, builds the dataset twice, rejects any identity mismatch, requires the exact 96-feature contract, writes the net-zero relative factor artifact, trains all three PPO seeds, and runs the two-fold nested walk-forward evaluation.

The complete dataset must be rebuilt after a feature-contract change. An older 10-feature artifact is rejected and must not be silently reused.

Binance Vision historical archives are cached by immutable URL. Current `exchangeInfo` is requested separately and is never loaded from the historical archive cache. When the REST endpoint is blocked, the runner records the error and uses the explicit checked-in metadata fallback.

Production status remains `NO-GO`. This workflow does not authenticate an account, place orders, support inverse COIN-M accounting, or claim profitability.
