# Current Architecture Documentation Sync and Audit Design

## Purpose

Synchronize the maintained user-facing documentation with the current `main` implementation, then perform an evidence-based architecture audit across the complete research, evaluation, serving, Studio, catalog, CI, and container paths.

This work separates documentation truth from historical design records and separates audit findings from later behavioral fixes. Passing tests, research validity, empirical profitability, release eligibility, and direct-exchange readiness remain distinct judgments.

## Scope

The documentation and audit branch is:

```text
docs/current-architecture-sync-20260722
```

The maintained current-state documents to update are:

```text
README.md
README.ja.md
START.md
docs/ARCHITECTURE.md
docs/RESEARCH_STATUS.md
docs/BINANCE.md
docs/operations/docker-gpu-full-training.md
studio/README.md
```

The audit report to create is:

```text
docs/verification/2026-07-22-post-merge-architecture-audit.md
```

Historical files under `docs/superpowers/specs/` and `docs/superpowers/plans/` remain immutable records of decisions made at their original dates. They are not rewritten to describe later implementations.


## Post-rebase baseline correction

Before implementation began, `main` advanced to `6bec98e43599c98fb4b86a1522ab455f5acd396b` through PR #78. That change already unified compatibility execution with the stateful engine, added `trade_rl.telemetry` as an enforced standard-library-only layer, added indexed strict telemetry reading, rejected duplicate seed streams, centralized canonical JSON, split PostgreSQL sealed-test reservations, and decomposed `ResidualMarketEnv`.

This documentation branch therefore audits the post-remediation state. It must remove stale descriptions of those findings as current, verify the remediations, and identify only remaining reproducible issues. The primary remaining audit target discovered during rebase is whether Live Training combines records from different vector environments or reset episodes into one apparent continuous market/equity series.

## Current confirmed documentation drift

The documentation update must correct at least these confirmed mismatches:

1. The maintained policy observation contract is `baseline_residual_observation_v5`, not observation schema v3.
2. Observation v5 includes seven causal pending-order coordinates per symbol: remaining notional ratio, order type, order status, age, eligible delay, trigger state, and expiry distance.
3. The maintained serving candidate is bundle v5, not bundle v4.
4. Execution is now stateful: explicit order intent, latency, eligibility, trigger, shared processing-bar capacity, partial-fill carry, time in force, cancellation/replacement, rejection, expiry, and deterministic audit events.
5. Conservative execution promotion requires explicit execution evidence and execution-policy identity. Optimistic and neutral path modes remain sensitivity-only.
6. The old statement that next-open execution capacity is always based on the previous completed bar's volume is no longer the maintained general contract. Capacity semantics depend on order type and the stateful execution policy; current-bar realized volume may be used only for completed-bar intrabar replay, while open-eligible market liquidity remains causal.
7. The actual dependency-layer configuration includes `studio` and `catalog`; the Architecture responsibility map must match `.importlinter` exactly.
8. `trade_rl.telemetry` is now an enforced layer below `artifacts` and above `domain`; maintained documents must remove the superseded ungoverned-layer claim.
9. PostgreSQL is a searchable artifact metadata, provenance, cache-identity, dependency, and lifecycle catalog. Immutable datasets, arrays, checkpoints, models, evidence, and run payloads remain filesystem artifacts.
10. Live Training telemetry is exploratory visualization evidence. It is not exchange-order evidence, selection evidence, checkpoint evaluation, sealed-test evidence, profitability evidence, or production authorization.

## Canonical-document hierarchy

The maintained documents have the following responsibilities:

- `README.md` and `README.ja.md`: capability boundary, project entry points, primary commands, and links to detailed contracts.
- `START.md`: minimal reproducible local dataset-to-training workflow and artifact inspection.
- `docs/ARCHITECTURE.md`: current responsibility map, dependency direction, data flow, identity contracts, execution semantics, evaluation separation, serving boundary, and explicit non-capabilities.
- `docs/RESEARCH_STATUS.md`: empirical evidence, failed or passed gates, historical results, current NO-GO reasons, and interpretation limits.
- `docs/BINANCE.md`: supported Binance products, causal public-data ingestion, execution-metadata modes, point-in-time limitations, and fixed smoke workflows.
- `docs/operations/docker-gpu-full-training.md`: actual CUDA-container operation, phase transitions, evidence collection, artifact extraction, retry, and cleanup.
- `studio/README.md`: Studio capabilities, telemetry semantics, job ownership, artifact read boundaries, checkpoint-evidence separation, and read-only serving display.
- `docs/verification/2026-07-22-post-merge-architecture-audit.md`: audit commit, commands, observations, confirmed findings, risks, non-findings, priorities, and remediation roadmap.

Stable schema names are documented only where they define an interoperability or identity boundary. Volatile test counts and build IDs belong in dated verification records, not permanent README prose.

## Architecture responsibility map

The maintained enforced layer order must be described exactly as configured:

```text
cli
studio
workflows
integrations
serving
learning
rl
risk
simulation
strategies
data
catalog
evaluation
release
artifacts
telemetry
domain
```

The audit must verify both direct and indirect import constraints, including:

- domain remains standard-library only;
- release remains verification-only and below serving;
- serving cannot import workflows, integrations, or training orchestration;
- learning and framework-neutral training contracts cannot import Stable-Baselines3, sb3-contrib, or PyTorch;
- runtime and training paths cannot import offline signer/private-key modules;
- catalog contracts remain independent of PostgreSQL adapters, NumPy, model frameworks, and higher application layers.

`trade_rl.telemetry` must be verified as an enforced standard-library-only layer. The audit also reviews the current package-initializer replacement pattern used to install indexed/strict implementations and records any maintainability risk without changing behavior in this documentation PR.

## Audit paths

The audit follows seven complete paths rather than reviewing files in isolation.

### 1. Market data and causality

Trace source data through `MarketDataset`, native multi-timeframe features, as-of alignment, availability, staleness, dataset identity, flat and sequence normalization, and policy observations.

Verify that no future return, future high/low, future tradability, post-period metadata revision, evaluation-period statistic, or future availability state enters a policy input or training transform.

### 2. Orders, execution, and accounting

Trace policy targets through risk projection, reconciliation, order intent, pending order state, latency, eligibility, OHLC path, trigger, capacity, partial fill, carry, fee/spread/impact/slippage, funding, borrow, corporate action, margin, liquidation, terminal accounting, and final wealth.

Verify fixed decision-time quantity semantics, no double issuance after partial fills, no self-cross, shared per-symbol capacity, causal market-order liquidity, correct quote/base/contract volume conversion, explicit time-in-force outcomes, deterministic evidence, and accounting conservation.

### 3. Training and evaluation separation

Trace behavior cloning, PPO, intermediate checkpoints, checkpoint validation, configuration selection, fixed-seed aggregation, baseline fallback, sealed fold testing, sealed outer access, execution sensitivity, selected-final training, and promotion evidence.

Verify that outer data, telemetry, post-selection sensitivity, and later confirmation never feed training or candidate selection. Verify that failed candidate gates remain baseline fallback and are never described as profitable RL selection.

### 4. Training-serving parity

Trace action specification, symbol ordering, observation v5 layout, pending-order state, flat/sequence normalizers, sequence windows, structured adapters, execution-policy digest, account state, serving snapshot, loader contract, and deterministic action probes.

Verify exact fail-closed behavior for identity, schema, shape, finite-value, bounds, ordering, staleness, pending state, and execution-policy mismatches.

### 5. Artifacts, PostgreSQL, and release

Trace dataset publication, run staging, exact file closure, canonical digests, catalog registration, cache identities, dependency edges, selection proposal, authorization, fresh confirmation, candidate bundle, detached release attestation, registry validation, and runtime activation.

Verify that filesystem artifacts remain authoritative, PostgreSQL registration is retryable and does not mutate artifact identity, private keys stay offline, public verification keys are purpose-bound, and release approval cannot be constructed from circular or incomplete evidence.

### 6. Studio and Live Training

Trace job creation, persisted ownership, process lifecycle, telemetry JSONL production, seed and environment identity, API resolution, browser buffering, replay selection, checkpoint evidence, run comparison, evidence display, and serving monitor boundaries.

Specifically test whether records from multiple vector environments or episodes can be combined into one false market/equity series and whether repeated JSONL polling has bounded cost for long runs.

### 7. CI, Docker, and privileged execution

Trace workflow permissions, exact-head checkout, dependency lock identity, import architecture, critical branch coverage, PostgreSQL service validation, Studio checks, Ubuntu/Windows compatibility, image construction, non-root execution, CUDA preflight, privileged runner restrictions, retained logs, and evidence artifacts.

Verify that CI and container success are described as software/package integrity only, never as profitability or production authorization.

## Finding model

Every audit item uses this structure:

```text
ID
Status: CONFIRMED | RISK | NOT_FOUND
Priority: P0 | P1 | P2 | P3
Affected responsibilities and files
Observed fact
Violated or protected invariant
Concrete impact
Reproduction or missing test
Recommended boundary
Independent remediation PR
```

Priorities are defined as:

- `P0`: leakage, incorrect accounting, sealed-test contamination, incorrect production promotion, private-key exposure, or silent live-capital risk.
- `P1`: training-serving mismatch, mixed order/telemetry identity, missing evidence identity, unbounded long-run behavior that makes a maintained workflow impractical, or fail-open validation.
- `P2`: responsibility concentration, unenforced dependency direction, duplicate configuration, unstable ownership, or material maintainability degradation.
- `P3`: naming, documentation clarity, small duplication, stale compatibility exports, or minor non-behavioral inconsistency.

A finding is `CONFIRMED` only when code, a failing test, an invariant proof, or a reproducible execution demonstrates it. Structural suspicion without reproduction is `RISK`. A checked path with no observed defect is recorded as `NOT_FOUND` when that negative evidence is important.

## Expected initial remediation split

The documentation/audit PR does not bundle behavioral fixes. Confirmed behavioral findings are split by responsibility, beginning with:

1. Live Training vector-environment and reset-episode stream isolation;
2. direct telemetry exports that remove hidden package-initializer replacement, if the audit confirms the indirection risk;
3. additional independently testable P0/P1 findings discovered by the post-remediation audit.

A confirmed P0 starts immediately on a separate branch and does not wait for the documentation PR to merge. P1 and lower findings normally reference the merged audit ID before implementation.

## Validation

The documentation/audit branch must run, at minimum:

```bash
uv sync --extra dev --extra train-sb3 --extra studio --extra postgres
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
npm ci --prefix studio --no-audit --no-fund
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
docker compose config
docker compose -f compose.training.yaml config
```

Focused validation covers:

- dataset identity, feature availability, staleness, and sequence alignment;
- stateful execution, latency, gap behavior, shared capacity, partial fills, cancellation, expiry, and deterministic replay;
- accounting oracle, reward, margin, and drawdown liquidation;
- walk-forward capabilities, sealed access, baseline fallback, promotion evidence, and execution sensitivity;
- observation snapshots, pending-order parity, structured serving, and execution-policy identity;
- release attestation and offline signer boundaries;
- PostgreSQL catalog unit and real-service integration;
- training telemetry, Studio telemetry API, checkpoint evidence, and Live Training frontend behavior.

Documentation commands and links must be checked against the exact branch head. The final audit report records exact commit SHA, commands, results, artifacts, and unresolved limitations.

## Completion criteria

The documentation/audit PR is complete only when:

- all maintained current-state documents agree on capability boundaries;
- observation v5, bundle v5, and stateful execution are accurately described;
- no maintained document retains the obsolete universal previous-bar capacity statement;
- PostgreSQL cannot be mistaken for the canonical storage location of model or numerical payloads;
- Live Training cannot be mistaken for exchange activity, sealed evaluation, or formal model selection;
- the Architecture responsibility map matches `.importlinter` exactly and explicitly reports the telemetry gap;
- every confirmed audit finding identifies evidence, impact, priority, and an independent remediation unit;
- document links, examples, and commands are verified;
- required static analysis, tests, Studio checks, Compose validation, and exact-head CI pass;
- production, direct exchange routing, and profitability remain explicitly `NO-GO` unless separate required empirical and operational gates pass.

## Non-goals

This PR does not:

- add direct exchange connectivity or authenticated account access;
- activate a serving bundle;
- create release approval or fresh confirmation evidence;
- alter model selection or profitability thresholds;
- rewrite historical design and plan records;
- combine all audit fixes into one large implementation PR;
- claim production readiness from successful documentation, CI, Docker, or smoke tests.
