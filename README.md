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
- `docs/architecture/final-evaluation-authorization.md` — WINNER-only one-shot gate before any unused/final-data access
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

## Optional perfect-information bound

The public perfect-information robustness bound uses SciPy. Install it through the capability extra:

```bash
uv sync --extra oracle
```

The `dev` extra also includes the same SciPy range for the repository test suite.

## Research rule

Do not add model complexity to make a result pass. First verify causality, data quality, execution accounting, costs, symbol-level robustness, and unused-data evidence. A correct `no winner` result is preferable to an overfit winner.
