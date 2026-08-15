# Causal Alpha V3 architecture hardening design

## Objective

Harden the research-only V3 runner so its persisted artifact graph is self-contained and independently auditable, resume is bound to the execution/runtime semantics that produced prior records, admission fails closed on net economics and hard-risk evidence, and a later process can load an admitted training teacher without access to in-memory state from the research run.

Canonical U6/V2 behavior is unchanged. This change does not run DAgger, BC, critic warm start, PPO, Lagrangian, or discounted PPO and makes no profitability or Production GO claim.

## Execution identity

A V3 run uses `CausalAlphaV3ExecutionIdentity` to bind the maintained Universal `training_contract_digest`, `instrument_context_schema_digest`, a digest of the installed `trade_rl` Python source tree, and per-symbol runtime semantics including dataset identity, execution-cost configuration, risk configuration, decision cadence, signal delay, episode horizon, and the hard market-notional cap. `CausalAlphaV3RunManifestV2` binds that execution identity. Reusing an output root after any bound identity changes therefore fails at the immutable run-manifest boundary before completed replay records may be skipped.

## Signal evidence

The runner persists each fit-config/symbol/episode `CausalAlphaV3SignalScopeMetric` under `signal/records/<fit>/<symbol>/<episode>.json`. Loaders require an exact schema and exact field set and revalidate the content digest. Aggregate signal uncertainty is computed over chronological episode clusters: same-episode symbol metrics are averaged into one cluster value before the moving-block bootstrap. Candidate target variants still share fit evidence, but correlated cross-symbol copies do not increase the independent-scope count.

## Single-writer persistence

`CausalAlphaV3RunLock` acquires one exclusive output-root lock before any stage side effect. A second writer fails before replay or admission. The lock is deliberately conservative: a stale lock is not silently stolen and requires explicit operator recovery. This prevents concurrent writers from evaluating the same holdout in parallel. A process crash between an external evaluation and its atomic record write can still cause that missing scope to be evaluated again after explicit lock recovery; the workflow therefore guarantees single-writer resume idempotency, not impossible physical exactly-once execution across crashes.

## Replay initial-state contract

Before each economic or admission replay, the live production environment resolves the contract's `initial_state_mode` at the contract start. The resolved weights must exactly match the frozen `contract.initial_weights`; otherwise replay fails closed. This prevents target compilation from using one initial portfolio while production replay starts from another after runtime/config drift.

## V3 admission v2

The maintained V2 admission function remains unchanged. V3 uses a separate `CausalAlphaV3AdmissionRecordV2` and aggregate gate. Each holdout record persists gross/net return, turnover, cost, drawdown, trade count, execution rejection reasons, risk projection reasons, and hard-risk violation. Admission fails when aggregate gross or aggregate net is negative, a majority of symbol gross returns are negative, any hard-risk violation occurs, or any unexplained execution rejection occurs.

## Durable training-only teacher package

Admission is evaluated from the full selected batch, whose last contract is the untouched holdout. Only after admission succeeds does the pipeline build the durable teacher package. The persisted `UniversalCausalAlphaV3TeacherPackageV2` contains **training contracts and targets only**; the admission holdout contract digest is retained separately as evidence and is prohibited from appearing in any package batch. Each symbol batch is persisted with full contract and target arrays under `teacher/batches/<symbol>.json`, and `teacher/package.json` binds their artifact digests plus partition, sample, selection, admission, candidate, run, and generator identities. A fresh process can reconstruct and revalidate the package from disk.

## Responsibility boundaries

`universal_causal_alpha_v3_runner.py` is a thin compatibility facade. Runtime preparation, identity contracts, signal statistics, persistence, replay/admission, durable teacher artifacts, and pipeline sequencing live in focused modules. Existing public runner entrypoints remain stable for the CLI and tests.

## Acceptance criteria

1. Signal leaf metrics are durable, strict-schema, digest-validated artifacts.
2. Same-episode symbol duplication does not increase the signal gate's independent scope count.
3. A changed execution/runtime/source identity cannot reuse an existing output root.
4. Replay fails before stepping when live initial weights differ from the frozen contract.
5. V3 admission rejects aggregate net-negative, hard-risk, and unexplained-rejection holdouts without changing V2/U6 admission behavior.
6. Only one active writer may operate on an output root.
7. Successful admission writes a reloadable training-only package whose batches cannot contain the admission holdout.
8. Hardened artifact mappings are immutable after digest construction.
9. The public runner is a thin orchestration facade and canonical U6 invariants remain unchanged.
10. Exact final-head targeted tests, full pytest, Ruff/format, Mypy, import architecture, coverage gates, compatibility, training image, and required CI checks pass before completion is claimed.
