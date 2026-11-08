# Trade RL

Trade RL is a lean research system for testing one symbol-agnostic long/short strategy across multiple markets under causal data, one execution/accounting ledger, hard risk constraints, and unused-data evaluation.

## Documentation

Use the [documentation index](docs/README.md) to find the authoritative contract for each task. Start with:

- [Lean core architecture](docs/architecture/lean-core.md) — causal data, risk, execution and common accounting.
- [Package boundaries](docs/architecture/package-boundaries.md) — source ownership and public API.
- [Current research decisions](docs/research/current-status.md) — recorded evidence, limitations and next gates.
- [Final-data authorization](docs/architecture/final-evaluation-authorization.md) — sealed/unused evaluation boundary.
- [Human Guide](https://shuntatsu.github.io/trade_rl/) ([maintenance](guide/README.md)) — readable but **non-authoritative**.
- [Agent entry](AGENTS.md) and [canonical agent contract](docs/AGENTS.md) — change/review instructions.

For Markdown changes, run `python -m tools.markdown_links --check` (local paths in **all tracked** `.md` files) and the applicable source/Guide/architecture checks. It does not validate remote URLs or heading fragments.

Agents must confirm current source, tests, `main`, open PRs, and the applicable canonical instructions before changing code. Integration needs a final PR HEAD containing current `main`, exact-HEAD Full CI, and the required review evidence. A `software_source_review_v2` receipt is publisher-attested **software-only** evidence; neither that receipt nor a passing CI authorizes economic experiments, unused/final-data access, live orders, or automatic merging.

## Current status

- M1 lean core: **implemented; end-to-end defect audit remains open**
- M2 universal comparison + Controlled Experiment Loop infrastructure: **complete**
- M2 canonical real-data baseline and Portable Controlled Experiment 0001: **verified; KEEP_BASELINE**
- M3 unused-future authorization boundary: **implemented; no final data is opened by it**
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

The bot diagnostic CLI also runs chronological parameter tuning and walk-forward
checks on an existing Dataset artifact:

```bash
uv run python -m trade_rl.evaluation.bot --mode walk-forward \
  --strategy adaptive --dataset <dataset-artifact-dir> \
  --signal-feature 1h__log_return_24bar \
  --objective balanced --windows 3 --max-combinations 60 --json
```

Each fold resets capital and strategy state. Reports expose terminal settlement
and residual positions; their compounded return is a hypothetical summary.
These are development diagnostics. Use `--demo` explicitly for a synthetic
software smoke. Channel strategies require the four named prior-candle channel
features from `with_price_channels`; arbitrary first columns are rejected.
Choose `--signal-feature` from the artifact's exact feature names for signal-based
strategies. The same feature is fixed for tuning and evaluation; omission keeps
the first-feature default. This example requires a Dataset containing the named
24-hour return feature.

The candidate comparison below requires a canonical filesystem market dataset
artifact and one JSON run config. Market-data artifacts are not committed to this
repository.

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
provenance.json
```

See `docs/research/current-status.md` for the accepted config keys, fit/evaluation rules, evidence outputs, and next-step decision process.

## Prospective carry paper operation

The separate BTC/ETH spot-long/perpetual-short candidate uses a public-data **paper-only** collector; it never places live orders. The frozen economic screen is 90 UTC days in three 30-day blocks, with declared costs and a 10% observed drawdown stop. Short software probes and earlier development returns do not qualify as a passed screen.

For the **current** attempt number, reviewed lineage, source/runtime binding, time tolerances, stop/failure rules, operator commands and authorization status, use the [active carry specification](docs/specs/funding-carry-paper.md), [implementation plan](docs/plans/funding-carry-paper.md), and [current research status](docs/research/current-status.md). These are versioned and may change between attempts; a copied command, historical start time, README example or passing CI is **not** permission to seal, collect, replay or evaluate. Keep previous failure receipts and study roots immutable.

Even a passed paper screen requires an independent live-execution review and never enables production routing automatically.

## Optional perfect-information bound

The public perfect-information robustness bound uses SciPy. Install it through the capability extra:

```bash
uv sync --extra oracle
```

The `dev` extra also includes the same SciPy range for the repository test suite.

## Research rule

Do not add model complexity to make a result pass. First verify causality, data quality, execution accounting, costs, symbol-level robustness, and unused-data evidence. A correct `no winner` result is preferable to an overfit winner.
