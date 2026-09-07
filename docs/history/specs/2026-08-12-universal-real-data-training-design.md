# Universal Real-Data Training Completion Design

## Objective

Run the maintained Universal U3-U6 training path to completion on the available
PostgreSQL-backed Binance data. Fix implementation or runtime defects encountered
along the way without weakening the authored training configuration, and inspect
both intermediate datasets and reward-function trends throughout the run.

Completion means all three U6 algorithm families—PPO, Lagrangian PPO, and
Discounted Lagrangian PPO—finish for the U5-selected `u_medium_direct`
architecture and produce verifiable manifests, checkpoints, telemetry, and
reward evidence. A completed training run is software success only; it must not
be presented as research success unless the repository's sealed U5/U6 evidence
gates also pass.

## Authoritative Inputs

The raw source is the existing `trade_rl_db` PostgreSQL database. Its `public`
schema remains read-only. Training uses these 15 symbols:

`ADAUSDT`, `APTUSDT`, `ARBUSDT`, `AVAXUSDT`, `BCHUSDT`, `BNBUSDT`,
`BTCUSDT`, `DOGEUSDT`, `ETHUSDT`, `LINKUSDT`, `LTCUSDT`, `OPUSDT`,
`SOLUSDT`, `SUIUSDT`, and `XRPUSDT`.

`PAXGUSDT` is excluded because it does not share the full common interval. The
source interval is the half-open UTC range `[2024-11-13T00:00:00Z,
2026-07-05T00:00:00Z)`. The resulting cache identity is:

`binance-usds-m-native-indicators-15x-20241113-20260705-v1`

The implementation must never reuse the existing 2021-2026 cache identity for
this shorter dataset. Frozen Binance instrument metadata already present in the
training volume is the metadata source. Missing symbol metadata or a digest
mismatch is fatal.

## Architecture and Boundaries

The work is split into two independently testable subsystems and a launch layer.

### 1. Native indicator cache materializer

A dedicated command reads `public.rl_klines`, `public.rl_derivatives`,
`public.rl_funding_rate`, and `public.rl_orderflow_1m` without changing them. It
publishes a new immutable generation in the `market_raw` schema of the same
database. Keeping source and destination in one database permits a transactional
publication and avoids an unverifiable cross-database copy.

The new table set is explicit and period-correct:

- `market_raw.binance_usds_m_klines_202411_202607`
- `market_raw.binance_usds_m_funding_202411_202607`
- `market_raw.binance_usds_m_indicator_manifests_202411_202607`
- `market_raw.binance_usds_m_indicator_artifacts_202411_202607`

The materializer:

- selects exactly the fixed symbols and common interval above;
- validates unique, monotonic one-minute event times and OHLCV invariants;
- resamples completed bars causally into 15m, 1h, 4h, and 1d native timeframes;
- aligns derivatives, funding, and order-flow data only when they were available;
- computes the canonical 206 target-local Universal feature values;
- records explicit availability masks rather than imputing future information;
- emits deterministic NPZ payloads, payload SHA-256 values, a feature-config
  digest, coverage statistics, and a canonical manifest; and
- publishes the manifest and all artifacts atomically and idempotently.

Re-running an identical identity must verify and reuse byte-identical artifacts.
If any existing row under that identity differs, publication fails rather than
overwriting it. Source tables are never dropped, truncated, or updated.

The loader accepts an immutable table-set value instead of embedding the old
2021-2026 generation as the only supported location. Its default remains the
legacy table set for backward compatibility, while the preflight manifest and
runtime factory explicitly select the new table set above. Table identifiers are
validated against a strict allowlist before interpolation into SQL.

### 2. Concrete Universal runtime factory

Add the maintained project-local factory:

`trade_rl.workflows.binance_universal_runtime:build_runtime`

It receives `algorithm`, the architecture-projected `run_config`, and a strict
runtime context. It loads the instrument bundle and native indicator cache,
then composes the existing U3-U6 helpers:

- `materialize_universal_train_datasets`
- `fit_universal_shared_normalizer`
- `bind_universal_normalizers`
- `publish_universal_train_dataset_artifacts`
- `CausalInstrumentContextProvider`
- `UniversalDatasetArtifactEnvironmentFactory`
- `UniversalRoutedEnvironmentFactory`
- `build_universal_training_runtime`

Dataset construction, normalizer fitting, behavioral-cloning teacher data, and
critic warm-start data may use training symbols only. Validation and test symbols
must not be loaded through any train preprocessing route. The three algorithm
families reuse the same verified dataset, feature schema, instrument partition,
and normalizer identities; only their authored algorithm contracts differ.

### 3. Preflight manifest and launch layer

A preflight command materializes and validates all static inputs before expensive
training. It writes a canonical runtime manifest containing non-secret identity
information: cache ID, table generation, symbol partition, source interval,
dataset roots, train index range, feature schema digest, normalizer digest,
instrument-catalog digest, metadata evidence digest, and source payload digests.
The PostgreSQL password or complete connection URL must not be stored in it.

The train index range is deterministic: start at the first row remaining after
the canonical feature warm-up and stop at the minimum complete row count shared
by every training-symbol dataset. Expressed in materialized dataset coordinates,
the range is `[0, shared_complete_row_count)`. No arbitrary example value such
as 100000 is accepted.

`scripts/run_universal_full_research.py` consumes the manifest, verifies every
declared digest against the live artifacts, and then invokes the runtime factory.
Explicit legacy arguments may remain for compatibility, but they must agree with
the manifest when both are supplied.

## Data Flow

1. Read the fixed raw PostgreSQL interval and frozen metadata.
2. Validate coverage, ordering, uniqueness, numeric finiteness, and OHLCV rules.
3. Build native causal bars, aligned auxiliary inputs, and 206-feature payloads.
4. Transactionally publish immutable indicator artifacts and their manifest.
5. Create the strict 15-symbol instrument catalog and train/validation/test
   partition using seed 17 and the maintained 9/3/3 split contract. The manifest
   records the resulting symbol membership; it is never reassigned after seeing
   validation or test results.
6. Materialize train-only datasets, fit the shared train-only normalizer, and
   publish dataset artifacts.
7. Write and independently verify the preflight runtime manifest.
8. Run small CPU integration and CUDA three-update smoke tests.
9. Build an immutable Docker image bound to the Git commit, lockfile digest,
   source digest, and runtime-manifest digest.
10. Launch full authored PPO, Lagrangian PPO, and Discounted Lagrangian PPO runs
    for `u_medium_direct`, baseline `supervised_allocator`, folds 0 and 1.
11. Preserve all checkpoints and telemetry and produce the final
    `universal-full-research-training.json` manifest.

The three canonical JSON configurations remain authoritative. Their 524288
timesteps, three seeds, eight environments, rollout size, batch size, device,
warm starts, reward contract, execution assumptions, and algorithm-specific
gamma values must not be silently reduced for the full run. Reduced settings are
allowed only in explicitly named smoke configurations and cannot be mistaken for
completion evidence.

## Intermediate Data and Reward Monitoring

Preflight produces per-symbol and per-timeframe reports with:

- raw and materialized row counts;
- first and last event time;
- missing, duplicate, or non-monotonic timestamps;
- incomplete native bars and source-row counts per bar;
- null, NaN, Inf, and availability-mask counts per feature;
- OHLCV violation counts;
- feature distribution summaries and extreme-value counts; and
- payload, feature-schema, metadata, and dataset digests.

During training, a durable heartbeat and append-only telemetry stream records at
least:

- episodic total reward and every reward component;
- baseline and policy portfolio value;
- rolling growth, drawdown, and downside statistics;
- fees, slippage, turnover, and trade count;
- constraint costs, violations, and Lagrangian multiplier;
- approximate KL, explained variance, policy loss, value loss, entropy, and
  action standard deviation;
- per-symbol sampling counts and reward summaries;
- step throughput, GPU/CPU memory, checkpoint age, and last successful update;
  and
- all non-finite observations, actions, rewards, gradients, and losses.

Monitoring reports reward levels and trends by algorithm, seed, symbol, and
checkpoint window. It flags reward collapse, flat or exploding value estimates,
vanishing action variance, persistent constraint violations, cost domination,
data starvation, NaN/Inf, OOM, or a stale heartbeat. A warning does not authorize
changing the research contract mid-run; it triggers evidence preservation and
root-cause diagnosis.

## Failure Handling and Repair Policy

The pipeline fails closed on source gaps, duplicate events, invalid OHLCV,
incomplete native bars, metadata or schema mismatch, digest drift, partition
leakage, non-finite values, missing telemetry, or incompatible algorithm
contracts.

On failure:

1. Preserve logs, checkpoints, manifests, telemetry, container inspection, and
   the exact command before stopping or replacing anything.
2. Classify the cause as data, feature/reward logic, runtime composition,
   infrastructure, or authored-model behavior.
3. Reproduce the smallest truthful failure and add a failing regression test.
4. Implement the narrow root-cause fix and run the relevant validation gates.
5. Create a new commit, Docker image identity, and run-generation directory.
6. Resume from a verified compatible checkpoint only; otherwise restart that
   algorithm/seed without overwriting the failed generation.

Authored hyperparameters or reward semantics may change only when evidence shows
they are the defect and the change is recorded as a new experiment contract.
Infrastructure failures must never be hidden by shrinking the workload.

## Verification Strategy

Unit tests cover half-open interval boundaries, every resampling boundary,
completed-bar causality, funding and auxiliary alignment, deterministic payloads,
SHA verification, cache identity selection, and idempotent publication.

Integration tests use PostgreSQL to prove transactional publication, loader
round-trips, corruption rejection, exact coverage, and failure without the
required metadata. Runtime tests prove that validation/test symbols never enter
dataset materialization, normalization, teacher training, or warm starts; they
also prove that all three algorithms share the same static identities.

An end-to-end smoke run proves the preflight-to-checkpoint path on CPU. A CUDA
three-update smoke proves device placement, vectorized environments, telemetry,
and checkpoint writing. Only after both pass is the immutable full training
generation launched.

Completion evidence consists of green relevant tests and validation gates, the
verified preflight manifest, immutable image/run identities, a final checkpoint
for every required algorithm/seed, complete reward telemetry, and the final U6
training manifest. `research_success=false` remains correct until the separate
sealed-evidence requirements documented by the repository are satisfied.

## Non-Goals

- Replacing PostgreSQL real data with synthetic, paper-only, or simplified data.
- Falling back to the legacy single-symbol or residual-baseline training path.
- Claiming profitability or research success from training completion alone.
- Rewriting unrelated repository components or cleaning unrelated user changes.
- Deleting old containers, volumes, runs, or artifacts to make the new run pass.
