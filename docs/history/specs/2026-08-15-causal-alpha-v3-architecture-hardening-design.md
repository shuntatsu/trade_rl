# Causal Alpha V3 architecture hardening design

## Objective

Harden the research-only V3 runner so its persisted artifact graph is self-contained and independently auditable, resume is bound to the execution/runtime semantics that produced prior records, admission fails closed on net economics and hard-risk evidence, and a later process can load an admitted training teacher without access to in-memory state from the research run.

Canonical U6/V2 behavior is unchanged. This change does not run DAgger, BC, critic warm start, PPO, Lagrangian, or discounted PPO and makes no profitability or Production GO claim.

## Execution identity

A V3 run uses `CausalAlphaV3ExecutionIdentity` to bind the maintained Universal `training_contract_digest`, `instrument_context_schema_digest`, a digest of the installed `trade_rl` Python source tree, the shared train-symbol market clock, the source-checkout dependency lock (`pyproject.toml` plus `uv.lock`), the exact Python implementation/version, and per-symbol runtime semantics including dataset identity, execution-cost configuration, risk configuration, decision cadence, signal delay, episode horizon, and the hard market-notional cap. `CausalAlphaV3RunManifestV2` binds that execution identity. Reusing an output root after any bound identity changes therefore fails at the immutable run-manifest boundary before completed replay or signal records may be skipped.

The container image identifier remains launcher-level provenance because an immutable image ID is not reliably discoverable from inside every supported runtime. The in-process artifact graph nevertheless binds code, dependencies, Python runtime, chronology, and per-symbol execution semantics directly.

## Signal evidence

V3 Signal Contract V2 distinguishes two evidence units explicitly:

- a **raw scope** is one `(fit config, symbol, signal contract)` metric;
- an **independent episode** is one chronological `(contract_start, contract_stop)` interval aggregated across symbols.

The runner persists each raw `CausalAlphaV3SignalScopeMetric` under `signal/records/<fit>/<symbol>/<episode>.json`. Each leaf is strict-schema, content-digest validated, and bound to the immutable `run_manifest_digest`, contract digest, contract interval, fit identity, forecast identity, symbol, and local episode index. A leaf copied from another run is rejected even when its fit/symbol/episode tuple happens to match.

Before pooled V3 fitting or signal evaluation, runtime preparation requires every train symbol to share the exact timestamp array, chronological episode `(episode_index, start, stop)` schedule, decision cadence, and signal delay. The shared clock is itself bound into execution identity. This closes the wall-clock and execution-timing assumptions behind pooled integer knowledge cutoffs and cross-symbol episode aggregation.

Aggregate signal uncertainty is computed only by the chronological clustered evaluator. Same-interval symbol metrics are averaged into one cluster value before moving-block bootstrap, while different intervals remain different independent observations even if they share a local `episode_index`. One aggregate evidence object may contain only one `fit_config_digest`, and every chronological cluster must contain one consistent pooled `fit_digest`; mixed fit configs or cluster-level fit drift fail closed both in the evaluator and in the public Evidence data contract. Signal Gate Evidence V2 persists `raw_scope_count`, `expected_raw_scope_count`, `raw_scope_coverage`, `independent_episode_count`, `expected_independent_episode_count`, `independence_unit=chronological_episode`, `aggregation_mode=cross_symbol_episode_mean`, the common run identity, and all bootstrap summaries.

The authored V2 configuration uses explicit units: `minimum_independent_episode_count` and `minimum_raw_scope_coverage`. The legacy ambiguous `minimum_scope_count` / `minimum_scope_coverage` surface is not a current execution contract and V1 authored configs are not silently reinterpreted.

Signal leaves are the resume source of truth. After the immutable run identity is closed, the pipeline constructs the complete expected leaf identity map and strictly loads all existing leaves once. Valid leaves are reused without rebuilding their fit/forecast metric, missing leaves alone are recomputed and atomically persisted, and corrupt, unknown, wrong-contract, wrong-path, duplicate, or cross-run leaves fail closed. Aggregate signal evidence is derived state and is recomputed from the validated leaf set on every run.

## Single-writer persistence

`CausalAlphaV3RunLock` acquires one exclusive output-root lock before any stage side effect. A second writer fails before replay or admission. The lock is deliberately conservative: a stale lock is not silently stolen and requires explicit operator recovery. This prevents concurrent writers from evaluating the same holdout in parallel. A process crash between an external evaluation and its atomic record write can still cause that missing scope to be evaluated again after explicit lock recovery; the workflow therefore guarantees single-writer resume idempotency, not impossible physical exactly-once execution across crashes.

## Replay initial-state contract

Before each economic or admission replay, the live production environment resolves the contract's `initial_state_mode` at the contract start. The resolved weights must exactly match the frozen `contract.initial_weights`; otherwise replay fails closed. This prevents target compilation from using one initial portfolio while production replay starts from another after runtime/config drift.

## V3 admission v2

The maintained V2 admission function remains unchanged. V3 uses a separate `CausalAlphaV3AdmissionRecordV2` and aggregate gate. Each holdout record persists gross/net return, turnover, cost, drawdown, trade count, execution rejection reasons, risk projection reasons, and hard-risk violation. Admission fails when aggregate gross or aggregate net is negative, a majority of symbol gross returns are negative, any hard-risk violation occurs, or any unexplained execution rejection occurs.

## Durable training-only teacher package

Admission is evaluated from the full selected batch, whose last contract is the untouched holdout. Only after admission succeeds does the pipeline build the durable teacher package. The persisted `UniversalCausalAlphaV3TeacherPackageV2` contains **training contracts and targets only**; the admission holdout contract digest is retained separately as evidence and is prohibited from appearing in any package batch. Each symbol batch is persisted with full contract and target arrays under `teacher/batches/<symbol>.json`, and `teacher/package.json` binds their artifact digests plus partition, sample, selection, admission, candidate, run, and generator identities. A fresh process can reconstruct and revalidate the package from disk.

## Responsibility boundaries

`universal_causal_alpha_v3_runner.py` is a thin compatibility facade. Runtime preparation, identity contracts, signal data contracts, chronological signal statistics, persistence, replay/admission, durable teacher artifacts, and pipeline sequencing live in focused modules. Existing public runner entrypoints remain stable for the CLI and tests. The persistence layer validates identities but does not decide statistical policy; the clustered evaluator decides signal statistics but does not perform filesystem I/O.

## Acceptance criteria

1. Signal leaf metrics are durable, strict-schema, digest-validated, run-bound artifacts.
2. Raw signal coverage and independent chronological episode count are separate explicit quantities.
3. Same-episode symbol duplication does not increase the signal gate's independent episode count.
4. Statistical clustering uses the contract interval, not local `episode_index` alone.
5. Cross-symbol clock, episode-schedule, decision-cadence, or signal-delay drift fails before pooled fitting.
6. Aggregate signal evidence rejects mixed fit configs and cluster-level pooled-fit digest drift, including direct Evidence construction outside the evaluator.
7. Valid signal leaves resume without recomputation; only missing leaves are rebuilt; corrupt or cross-run leaves fail closed.
8. A changed execution/runtime/source/dependency/Python/chronology identity cannot reuse an existing output root.
9. Replay fails before stepping when live initial weights differ from the frozen contract.
10. V3 admission rejects aggregate net-negative, hard-risk, and unexplained-rejection holdouts without changing V2/U6 admission behavior.
11. Only one active writer may operate on an output root.
12. Successful admission writes a reloadable training-only package whose batches cannot contain the admission holdout.
13. Hardened artifact mappings are immutable after digest construction.
14. The public runner is a thin orchestration facade, only the clustered Signal Gate is current, and canonical U6 invariants remain unchanged.
15. Exact final-head targeted tests, full pytest, Ruff/format, Mypy, import architecture, coverage gates, compatibility, training image, and required CI checks pass before completion is claimed.

The earlier narrow `2026-08-15-causal-alpha-v3-signal-scope-contract-fix.md` plan is historical evidence of the first symptom (`24` versus `8`). Signal Contract V2 supersedes that narrow interpretation; the architectural contract is the explicit raw/independent-unit design above.
