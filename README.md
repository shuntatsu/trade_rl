# Trade RL

Trade RL is a lean research system for testing one symbol-agnostic long/short strategy across multiple markets under causal data, one execution/accounting ledger, hard risk constraints, and unused-data evaluation.

## Current status

- M1 lean core: **complete**
- M2 comparison infrastructure: **complete**
- M2 real-data development comparison: **not run yet**
- M3 frozen final evaluation / stress / deletion: **not started**
- Profitability claim: **none**
- Production/live order routing: **not authorized**

The maintained architecture is intentionally small. Historical research generations and transitional pipelines are not runtime authorities.

## Documentation

- [Documentation portal](docs/README.md)
- [Current research status](docs/research/current-status.md)
- [Licensing](LICENSES/LICENSING.md)
- [Provenance](LICENSES/PROVENANCE.md)

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

Ridge, LightGBM, and PPO are universal models/policies without symbol identity features. Fit scope is explicit, and the same frozen strategy is replayed independently for every evaluation symbol. Per-symbol evidence remains visible rather than being hidden by aggregate P&L.

## Run a development comparison

A canonical filesystem market dataset artifact and one JSON run config are required. Market-data artifacts are not committed to this repository.

```bash
uv run --extra forecast-gbm --extra train-sb3 \
  python -m trade_rl.evaluation.runs.candidate \
  --dataset <dataset-artifact-dir> \
  --config <run-config.json> \
  --output <new-result-dir>
```

The output directory is immutable and contains `summary.json` plus `returns.npz`. See the [current research status](docs/research/current-status.md) for the evidence protocol and decision rules.

## Research rule

Do not add model complexity merely to make a result pass. Verify causality, data quality, execution accounting, costs, symbol-level robustness, and unused-data evidence first. A correct `no winner` result is preferable to an overfit winner.
