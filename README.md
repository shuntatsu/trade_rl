# Trade RL

Trade RL is a lean research system for testing one symbol-agnostic long/short strategy across multiple markets under causal data, one execution/accounting ledger, hard risk constraints, and unused-data evaluation.

## Documentation

Start from:

- [Interactive Guide](https://shuntatsu.github.io/trade_rl/) — published human-facing Guide (non-authoritative; technical/research truth remains under `docs/`)
- `guide/README.md` — Interactive Guide development and deployment maintenance
- `docs/README.md` — current documentation index
- `docs/architecture/lean-core.md` — causal data, strategy/risk, execution/accounting, artifact invariants
- `docs/architecture/package-boundaries.md` — current package ownership and dependency direction
- `docs/architecture/controlled-experiment-loop.md` — append-only Study/Experiment/EvidenceSet lifecycle and freeze contract
- `docs/architecture/final-evaluation-authorization.md` — one-shot authorization boundary between frozen WINNER development evidence and sealed unused-future evaluation
- `docs/research/current-status.md` — research status, candidate comparison, development/final protocol

Agents should read root `AGENTS.md` and `docs/AGENTS.md` before making changes. Pull requests are gated by CI checks and an exact-HEAD Independent Research Review (audited by an external AI reviewer; findings trigger changes requested/BLOCKED, clean reviews allow automated squash-merge and close).

## Current status

- M1 lean core: **complete**
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

The separate BTC/ETH spot-long/perpetual-short candidate has a public-data paper
runner. It does not place live orders. Its economic screen is ninety UTC days,
with three thirty-day blocks, fixed costs and a 10% observed drawdown stop.
Past development returns and short software probes do not qualify this screen.
See `docs/research/current-status.md` for the evidence and limitations.

Use a dedicated checkout and unchanged Python environment for the entire study.
Choose an aware ISO start at least five minutes in the future. Sealing reserves
a new directory and prints a protocol digest; preserve that digest outside the
study directory before starting collection. Prepare a UTF-8 JSON lineage file
from a result-blind review of earlier attempts, following
`docs/specs/funding-carry-paper.md`. For the currently documented series, the
next attempt is number 2 and must cite attempt 1 using the identifiers recorded
in `docs/research/current-status.md`. Do not seal a successor until an
independent result-blind review has resolved the remaining attempt-uniqueness
and source-to-ledger checks; no successor is currently authorized.

```bash
python -m trade_rl.evaluation.paper.cli seal \
  --root <new-study-dir> \
  --start-at <future-UTC-ISO-time> \
  --attempt-lineage-json <reviewed-lineage.json>
python -m trade_rl.evaluation.paper.cli run --root <study-dir> --protocol-sha256 <sealed-digest>
python -m trade_rl.evaluation.paper.cli status --root <study-dir> --protocol-sha256 <sealed-digest>
```

`run` collects once per minute through the fixed close and 180-second grace.
Keep its terminal or service running. A failure is permanent for that study;
restarting does not erase a gap or an unfinished capture. `status` checks the
journal chain and exposes positions, costs and last observation, but is not a
financial audit. After the deadline and collector exit, preserve the final tip
externally and run the complete offline replay:

```bash
python -m trade_rl.evaluation.paper.cli evaluate --root <study-dir> --protocol-sha256 <sealed-digest> --expected-tip <pinned-final-tip> --output <new-assessment.json>
```

The assessment output is write-once. A passed paper screen still requires live
execution review and never enables production routing automatically.

## Optional perfect-information bound

The public perfect-information robustness bound uses SciPy. Install it through the capability extra:

```bash
uv sync --extra oracle
```

The `dev` extra also includes the same SciPy range for the repository test suite.

## Research rule

Do not add model complexity to make a result pass. First verify causality, data quality, execution accounting, costs, symbol-level robustness, and unused-data evidence. A correct `no winner` result is preferable to an overfit winner.

## Opt-in causal path signatures (software-only)

A separately versioned rolling piecewise-linear Path Signature is available as an
**opt-in Dataset augmentation**, without changing canonical candidate sets:

```python
from trade_rl.data.features import with_path_signatures

augmented = with_path_signatures(
    verified_dataset, window_bars=24, depth=2, include_volume=False
)
# Explicitly select augmented.feature_names in a new development run.
```

The channels are normalized bar-step time and log close (optionally positive
log volume), with degrees 1–3. All window bars must be observable by their
own close. Missing or late inputs produce masked-out features, never fills or
signals. No OHLC intrabar ordering is inferred. Input window/depth, training
scope and execution economics must be fixed in a separate Controlled Experiment
before claiming any after-cost improvement. This capability establishes no
profitable strategy, sealed-final qualification, or production authorization.

## Native Multi-Timeframe rolling Signature (opt-in)

Use `with_native_multitimeframe_signatures` to calculate each clock on
**its own completed native bars** before as-of synchronization with an already
verified base Dataset. A 15m path is never reconstructed from carried 1h values.

```python
from trade_rl.data.features import (
    SignatureClock,
    with_native_multitimeframe_signatures,
)

augmented = with_native_multitimeframe_signatures(
    verified_dataset,
    native_market_source,   # load_timeframe(symbol, timeframe)
    instruments,            # exact verified_dataset.symbols order
    clocks=(
        SignatureClock("15m", window_bars=24, depth=2, max_staleness_hours=0.25),
        SignatureClock("4h", window_bars=12, depth=2, max_staleness_hours=4.0),
    ),
)
```

The source must supply native `RawMarketSeries` with reliable event and
`available_at` clocks, and be bounded to the Dataset evaluation time scope.
A native gap or late row resets the rolling path; invalid or stale features are
masked. A time-normalized Chen product is updated using a bounded two-stack
rolling queue (amortized O(1) segment changes per bar), and the source-input
content digests are bound to a new Dataset identity. This does **not** establish
historical provider publication timing, calibrated market costs, profitable
alpha, or eligibility to inspect sealed unused-future data.

As with the base-clock API, fitting and evaluation must use a **new**
preregistered Controlled Factor and unchanged common execution/accounting.
