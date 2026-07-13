# Residual Walk-Forward Architecture Hardening Design

## Status

Approved for implementation on `feature/residual-walk-forward` and PR #12.

## Goal

Make the residual nested Walk-Forward path structurally safe, deterministic, dependency-correct, and auditable before merge. The change must preserve the legacy direct-weight Walk-Forward behavior while fixing stale artifacts, zero-fold success, duplicate candidate logic, validation reuse, weak provenance, annualization drift, and incomplete aggregate reporting.

## Scope

This design changes only the Baseline-Anchored Residual research path and shared evaluation helpers that are already used by that path. It does not enable Registry registration, sealed release evidence, deployment, or Production readiness. `release_eligible` remains `false`.

## Architectural boundaries

### Evaluation layer

`mars_lite/eval/residual_walk_forward.py` becomes a pure evaluation module. It may depend on feature and numerical utility modules, but must not import `mars_lite.pipeline`.

It owns:

- immutable fold specification types;
- deterministic fold boundary construction;
- validation of non-overlapping train/checkpoint/configuration/OOS windows;
- stitched OOS return aggregation;
- fold summary statistics;
- strict JSON-safe report shaping helpers.

It does not own dataset construction, model training, artifact writes, CLI handling, or candidate selection.

### Pipeline layer

`mars_lite/pipeline/residual_candidates.py` becomes the single authoritative A/B/C/D candidate implementation for both the single-split residual runner and residual Walk-Forward.

It owns:

- Identity and fixed-alpha diagnostic agents;
- B and D ensemble training;
- checkpoint-validation use;
- configuration-selection use;
- A/B/C/D evaluation matrices;
- final selection and model digest generation.

`mars_lite/pipeline/residual_walk_forward.py` owns orchestration:

- dataset construction and quality checks;
- immutable run configuration resolution;
- fold-local alpha fitting;
- candidate training and selection;
- outer OOS evaluation at 1x and 2x costs;
- staging-directory writes;
- atomic publication of a completed run;
- failed-run isolation.

`mars_lite/pipeline/residual_pipeline.py` uses the same public candidate API for its single-split workflow and must no longer duplicate candidate construction.

### Dependency direction

The allowed direction is:

```text
pipeline -> eval
pipeline -> learning/trading/features
eval -> features/numerical utilities
```

`eval -> pipeline` is forbidden and enforced by a test.

## Fold design

Every outer fold uses four chronological segments:

```text
policy train
-> purge
-> checkpoint validation
-> purge
-> configuration selection
-> purge
-> outer OOS
```

Requirements:

- all boundaries are absolute base-bar indices;
- each segment is non-empty and meets minimum size requirements;
- checkpoint validation and configuration selection never overlap;
- outer OOS is never passed to candidate training or selection;
- at least two folds must complete successfully;
- fewer than two completed folds fails the run without publishing a success report.

Default proportions within each outer training prefix are:

- policy train: first 70%;
- checkpoint validation: next 15% after purge;
- configuration selection: final 15% after purge.

The effective purge is `max(requested_purge, horizon, 24)`.

## Configuration contract

The pipeline converts CLI arguments into an immutable `ResidualWalkForwardConfig` before any fold runs. It stores both requested and effective values.

The configuration records:

- requested and effective `decision_every`;
- requested fold count;
- effective purge;
- requested and effective ensemble size;
- base timeframe and `bars_per_year`;
- horizon;
- run tier;
- signal model;
- dataset identity;
- Git SHA when supplied;
- fee profile and cost parameters.

The `argparse.Namespace` is not mutated after configuration resolution.

## Candidate training contract

The public candidate API accepts separate datasets:

```python
train_select_residual_candidates(
    *,
    args,
    train_fs,
    checkpoint_val_fs,
    selection_fs,
    trend_family,
    alpha,
    env_kwargs,
    output,
) -> ResidualCandidateSelection
```

Each PPO seed uses `checkpoint_val_fs` only for checkpoint restoration. A/B/C/D matrices and final configuration selection use `selection_fs` only.

The selected configuration remains restricted to A, B, or D. C remains diagnostic only.

The returned selection includes a deterministic model identity:

- A: `identity:base_trend_v2`;
- B/D single model: SHA-256 of the saved model artifact;
- B/D ensemble: canonical SHA-256 over member file names and member digests.

The same identity must be recorded for both 1x and 2x OOS evaluations.

## Artifact transaction model

A run writes only under:

```text
<output>/.staging/<run_id>/
```

During execution:

- no file under the final successful run directory is changed;
- every fold report and model artifact is written under staging;
- the authoritative report is validated with strict JSON serialization;
- a run manifest with content digests is created;
- at least two folds must be complete;
- all expected fold reports must exist.

On success, staging is atomically renamed to:

```text
<output>/residual_wf_runs/<run_id>/
```

Then `<output>/residual_walk_forward.json` is atomically replaced with a small pointer/report containing the completed run identity and authoritative aggregate.

On failure, staging is moved to:

```text
<output>/failed/<run_id>/
```

and no success report is published. An existing prior success report may remain as the prior success, but it must include its own `run_id`; the failed run must never overwrite or partially mix with it.

## OOS aggregation

Each fold retains hybrid and shadow base-bar return series internally. The report may omit raw full arrays from the top level, but the aggregate must be computed from the chronological concatenation of non-overlapping fold series.

The stitched section reports:

- hybrid total return;
- shadow total return;
- excess log return;
- hybrid and shadow Sharpe;
- hybrid and shadow Sortino;
- hybrid and shadow max drawdown;
- combined moving-block bootstrap over hybrid minus shadow base-bar returns;
- total OOS base bars;
- total trades, turnover, and costs.

Fold mean and median metrics remain supplemental and must not be labeled as total Walk-Forward performance.

## Annualization and baseline parity

All residual evaluation and diagnostic baselines receive the same effective `bars_per_year` derived from the base timeframe.

`simulate_strategy`, oracle conversion, and any DSR calculation touched by this work must not use `BARS_PER_YEAR_1H` when another base timeframe is active.

Baseline rebalance rules used in residual reports must use absolute timestamps or absolute indices, not slice-relative `t % N`, so evaluation context does not change rebalance phase.

## Reporting and provenance

Every fold report includes:

- absolute split boundaries for all four segments;
- requested and effective configuration values;
- dataset identity;
- alpha artifact identity;
- selected configuration;
- selected model digest;
- selected seed fallback flags;
- 1x and 2x relative results;
- 1x and 2x baseline diagnostics;
- explicit evidence that both cost scenarios reference the same model digest;
- hybrid and shadow trade counts;
- action and weight-stage diagnostics.

The top-level report includes:

- run ID and completion status;
- completed and skipped folds;
- stitched OOS metrics;
- fold-distribution metrics;
- zero-trade warnings;
- release blocker;
- `release_eligible: false`.

## Documentation

`docs/ARCHITECTURE.md` is updated because it is the normative architecture description. It must describe:

- residual research Walk-Forward as a Control Plane responsibility;
- separation between evaluation primitives and pipeline orchestration;
- staging and atomic publication;
- research-only and fail-closed Registry boundary.

English and Japanese residual operating docs are updated to match the actual output layout and split semantics.

## Tests

Required contracts:

1. failed rerun cannot publish or mix a stale success result;
2. zero or one completed fold fails closed;
3. `mars_lite.eval.residual_walk_forward` does not import `mars_lite.pipeline`;
4. single-split and Walk-Forward use the same candidate function;
5. checkpoint validation and configuration selection do not overlap;
6. stitched total return matches concatenated fold log returns;
7. 4h and 1d annualization is correct for residual and baseline diagnostics;
8. baseline rebalance phase is stable under contextual slicing;
9. effective `decision_every` equals environment and report values;
10. 1x and 2x OOS results share the same model digest;
11. partial artifacts remain isolated under failed runs;
12. legacy direct Walk-Forward output names and dispatch remain unchanged;
13. strict JSON output rejects NaN or Infinity;
14. all existing tests and coverage gate continue to pass.

## Non-goals

- no Registry registration;
- no sealed holdout promotion;
- no automatic merge;
- no claim of profitability;
- no general rewrite of the direct-weight training stack.
