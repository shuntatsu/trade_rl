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
- `docs/research/current-status.md` — research status, candidate comparison, development/final protocol

Agents should read root `AGENTS.md` and `docs/AGENTS.md` before making changes.

## Current status

- M1 lean core: **complete**
- M2 universal comparison + Controlled Experiment Loop infrastructure: **complete**
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
study directory before starting collection.

```bash
python -m trade_rl.evaluation.paper.cli seal --root <new-study-dir> --start-at <future-UTC-ISO-time>
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
