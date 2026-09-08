# Trade RL

Trade RL is a lean research system for testing one symbol-agnostic long/short strategy across multiple markets under causal data, one execution/accounting ledger, hard risk constraints, and unused-data evaluation.

The current design and execution contract is:

- `docs/trade_rl_lean_redesign_20260908.md`

## Current status

- M1 lean core: **complete**
- M2 universal comparison infrastructure: **complete**
- M2 real-data development comparison: **not run yet**
- M3 frozen final evaluation / stress / deletion: **not started**
- Profitability claim: **none**
- Production/live order routing: **not authorized**

The old U-series / Causal Alpha generation stack and mandatory `teacher -> admission -> BC -> RL` route are not the current architecture.

## What is compared

One shared setup compares five candidates and three controls:

- `trend`
- `mean_reversion`
- `ridge24`
- `lightgbm24`
- `ppo`
- `cash`
- `constant_long`
- `constant_short`

Ridge, LightGBM, and PPO are trained as universal models/policies without symbol identity features. The same frozen strategy is replayed independently for every symbol; per-symbol results are retained instead of being hidden by aggregate P&L.

## Run a development comparison

A canonical filesystem market dataset artifact and one JSON run config are required. Market-data artifacts are not committed to this repository.

```bash
uv run --extra forecast-gbm --extra train-sb3 \
  python -m trade_rl.evaluation.runs.candidate \
  --dataset <dataset-artifact-dir> \
  --config <run-config.json> \
  --output <new-result-dir>
```

The output directory is immutable and contains:

```text
summary.json
returns.npz
```

`summary.json` records the dataset artifact identity, complete resolved candidate configuration, evaluation scope, and every symbol × strategy metric/diagnostic. `returns.npz` preserves the raw interval-return series for later paired or block-bootstrap analysis.

See `docs/trade_rl_lean_redesign_20260908.md` for the exact config schema, evaluation rules, and next-step decision process.

## Research rule

Do not add model complexity to make a result pass. First verify causality, data quality, execution accounting, costs, symbol-level robustness, and unused-data evidence. A correct `no winner` result is preferable to an overfit winner.
