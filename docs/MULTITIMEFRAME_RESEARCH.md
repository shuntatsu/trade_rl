# Native Multi-Timeframe Research

Trade RL treats multi-timeframe context as a first-class feature contract rather than as long lookbacks on a single bar series.

The maintained Binance research example uses:

- decision clock: `1h`;
- native feature clocks: `15m`, `1h`, `4h`, and `1d`;
- instruments: `BTCUSDT`, `ETHUSDT`, and `BNBUSDT` USDⓈ-M futures;
- fixed closed range: `2024-12-01T00:00:00Z` through `2026-06-01T00:00:00Z`;
- expected hourly decisions: `13,128`;
- full PPO seeds: `0`, `1`, and `2`;
- full PPO timesteps: `131,072` per seed;
- nested walk-forward candidate timesteps: `32,768` per seed and fold.

Each feature is calculated on its native closed-bar clock. The generic dataset builder then performs an availability-aware causal as-of alignment to the one-hour decision clock. A four-hour or daily feature is unavailable before the native bar and all of its source observations are available. No backward filling is used.

The full runner is:

```bash
uv sync --extra dev --extra train-sb3
uv run python examples/binance-multitimeframe/run_full_research.py \
  --work-root var/binance-multitimeframe-full
```

It builds the fixed dataset twice, verifies identical dataset and artifact digests, performs three-seed PPO training, runs nested walk-forward evaluation, and writes `summary.json` together with the exact metadata source used.

Binance Vision historical archives are cached by immutable URL. Current `exchangeInfo` is requested separately and never loaded from the historical archive cache. When the REST endpoint is blocked by regional HTTP 451, the runner records the error and uses the checked-in metadata fallback explicitly.

Production status remains `NO-GO`. This workflow does not authenticate an account, place orders, support inverse COIN-M accounting, or claim profitability.
