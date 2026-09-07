# Universal Trade RL U2 Source / FIT / Routing Identity Amendment

Status: **Normative U2 V1 amendment**  
Production: **NO-GO**  
Admission: **SEALED**  
Real U2 training: **NO-GO**

This amendment is written before any real U2 PPO training, Development numeric evaluation, Admission access, or economic result. It is a mechanics / provenance amendment discovered by implementation-time falsification review and must not be relaxed after Development results.

It supersedes only the affected source-loading, routed-environment, seed, vector-worker identity, and checkpoint-lineage statements in:

- `2026-09-03-universal-trade-rl-u2-base-ppo-selection-design.md`
- `2026-09-03-universal-trade-rl-u2-base-ppo-selection.md`
- `2026-09-03-universal-trade-rl-u2-robustness-timeout-amendment.md`

All economic thresholds, algorithm choices, architecture choices, Development / Admission isolation rules, U1 reward semantics, timeout-bootstrap rules, and fixed PPO hyperparameters remain unchanged.

---

## 1. Why this amendment exists

Implementation-time falsification review found five mechanics loopholes before any real U2 economic result:

1. `U2TrainingSource` required source and FIT to be dense 15-minute grids but did not independently require FIT to be phase-aligned to the source grid.
2. The current U2 environment builder accepts caller-supplied dataset bindings and a child-environment factory. It validates the resulting FIT child but does not own the maintained derivation from the frozen U0 source artifact to that child.
3. The generic routed environment derives episode seed from the complete `InstrumentDatasetBinding.digest`; unrelated binding metadata can therefore change episode sampling.
4. Stable-Baselines3 offsets vector-worker reset seeds as `member_seed + environment_index`, while the generic routed environment accepts only `reset(seed == run_seed)`. Workers 1..7 therefore conflict with a common immutable U2 member run seed.
5. The maintained SB3 backend probes `environment_factory()` before vector construction and uses that one environment's `environment_digest` as the training/checkpoint identity. A worker-0-specific digest therefore does not bind the complete fixed 8-worker generation.

The amendment closes these loopholes without introducing a new market-data format, persistent FIT artifact, vector framework, or checkpoint schema.

---

## 2. Quality contract

### Objective

Create one fail-closed U2 V1 chain:

```text
frozen U0 source identity
-> canonical source artifact load
-> exact FIT MarketDatasetView
-> internally derived U2 binding
-> frozen U1 environment validation
-> deterministic routed worker
-> shared U2 environment-generation identity
-> SB3 checkpoint environment identity
```

and define the exact member-seed / worker-reset-seed namespace for `n_envs=8`, `vector_environment_mode=in_process`.

### Non-goals

This amendment does **not**:

- run real PPO training;
- open Development or Admission numeric arrays;
- change PPO hyperparameters, architecture, reward, risk, execution, or Selection thresholds;
- create or publish a persistent FIT dataset artifact;
- redesign `MarketDataset`, `MarketDatasetView`, generic `EpisodeRoutedSingleInstrumentEnv`, generic SB3 vectorization, or generic checkpoint schemas;
- claim exact mid-episode / bitwise trajectory resume from an intermediate checkpoint;
- implement Development B/C/D evaluation or Selection.

### Acceptance Criteria

1. FIT bounds are provably aligned to the frozen source grid using metadata before numeric loading.
2. A canonical U0 source artifact must match the frozen source dataset identity before FIT materialization.
3. FIT is the deterministic maintained `MarketDatasetView` of that exact source and exact preregistered range.
4. Production U2 training does not accept caller-authored `InstrumentDatasetBinding` values as source of truth.
5. U2 episode sampling is unaffected by unrelated execution/descriptor binding metadata.
6. All eight in-process workers share one immutable member run seed and one run-level environment-generation digest while retaining distinct worker indices.
7. Every worker owns fresh mutable U1 runtime state; immutable FIT datasets, normalizer, and contracts may be shared.
8. The SB3-facing reset seed for worker `i` is exactly `member_seed + i`, while router `run_seed` remains the member seed.
9. Checkpoint environment identity binds U2 contract, training config, source closure, exact FIT bindings, member seed, fixed vector mode, and fixed worker set.
10. Existing U1 observation/action/reward/risk/execution semantics remain unchanged.

### Invariants

- Source artifact filesystem paths are locators, never research identity.
- `source.dataset_digest` denotes the frozen U0 source `MarketDataset.dataset_id`.
- Formal source artifacts must have verified canonical content identity.
- FIT materialization cannot escape the frozen source or preregistered FIT interval.
- `MarketDatasetView.identity` is the FIT dataset identity used by routed children.
- The same symbol-specific FIT dataset identity is used by all workers in one member run.
- `run_seed == preregistered PPO member seed` for U2 V1.
- `environment_index` is a worker coordinate and never redefines `run_seed`.
- U2 policy input remains symbol-independent with instrument and V4 context disabled.
- `ResidualMarketEnv` / U1 remains the sole Risk / Execution / Accounting authority.
- Normal U1 horizon remains `terminated=false`, `truncated=true`, no liquidation.

### Primary Failure Modes

- source/FIT grid phase mismatch;
- wrong or tampered source artifact;
- matching timestamps/count but different numeric source content;
- source artifact without canonical content identity;
- wrong FIT start/stop or off-by-one range;
- caller binding spoof/drift;
- unrelated binding metadata changing episode start sampling;
- worker 1..7 reset rejection from SB3 seed offsets;
- worker index collapse to zero;
- cross-worker BookState / pending-order / episode-state leakage;
- worker-specific runtime/data drift hidden by a worker-0 checkpoint identity;
- source closure or FIT binding drift accepted on checkpoint resume;
- timeout metadata loss at vectorization, covered by the existing timeout amendment.

### Risk

Data/source drift, hidden sampling drift, or worker-state contamination can invalidate the scientific interpretation of a seed while still producing apparently valid model artifacts. These failures are **training-blocking technical NO-GO**.

### Test Oracle

Correctness is observed through exact source dataset identity, exact source/FIT bounds, `MarketDatasetView.identity`, internally derived binding payloads, episode seed/binding, router seed/index, worker object identity/state, shared environment-generation digest, SB3 reset seeds, checkpoint environment identity, and unchanged U1 reward/runtime evidence.

### Required Test Layers

Unit/contract + canonical-artifact integration + U1/U2 integration + real `DummyVecEnv` integration + timeout/terminal-observation integration + static analysis/Ruff/format/MyPy/architecture checks + related/full suite + package/build + exact-final-HEAD CI.

### Quality Gate

U2 remains training **NO-GO** until all Acceptance Criteria have executable oracles, required layers pass on one exact final HEAD, falsification review finds no unresolved Critical/High mechanics issue, and remaining limitations are recorded. Test Green alone is insufficient.

---

## 3. Source / FIT grid alignment

Let:

```text
BAR_NS = U2_DECISION_STEP_NS = 15 minutes
```

The existing dense source/FIT grid equations remain mandatory. In addition, FIT must lie on the exact source grid:

```text
fit_offset_ns = fit_first_timestamp_ns - source_first_timestamp_ns
fit_offset_ns >= 0
fit_offset_ns % BAR_NS == 0

fit_start_index = fit_offset_ns // BAR_NS
fit_stop_index  = fit_start_index + fit_bar_count

0 <= fit_start_index < fit_stop_index <= source_row_count
```

The implied timestamps must equal FIT metadata exactly:

```text
source_first_timestamp_ns + fit_start_index * BAR_NS
    == fit_first_timestamp_ns

source_first_timestamp_ns + (fit_stop_index - 1) * BAR_NS
    == fit_last_timestamp_ns
```

A source grid `00:00, 00:15, ...` and FIT grid `00:05, 00:20, ...` is invalid even though both are independently dense 15-minute grids.

Failure is metadata-only technical NO-GO. Numeric arrays must not be opened to repair it.

---

## 4. Canonical source artifact loading

U0 source identity intentionally does not contain a filesystem path. U2 runtime may therefore receive a Train-symbol-to-artifact-locator mapping, but locator values must not enter research digests.

Before loading any source, locator keys must equal the complete Train source closure exactly: no missing, extra, Development, or Admission symbol.

For each Train source, before FIT materialization, U2 loads one canonical market-dataset artifact and requires:

```text
dataset.identity_verified == true
dataset.dataset_id         == source.dataset_digest
dataset.symbols            == (source.symbol,)
dataset.n_bars             == source.source_row_count
first timestamp            == source.source_first_timestamp_ns
last timestamp             == source.source_last_timestamp_ns
```

The loaded timestamp array must equal the exact dense 15-minute grid implied by source metadata.

Artifact/file tampering, a wrong valid artifact at the supplied locator, unverified content identity, wrong symbol/count/endpoints, or numeric content drift must fail before FIT materialization and before U1 environment creation.

### 4.1 Locator independence

Moving one byte-identical canonical source artifact to another filesystem path must not change source closure, FIT view identity, binding identity, environment-generation identity, episode sampling, or checkpoint compatibility.

---

## 5. FIT is an in-memory `MarketDatasetView`

U2 V1 reuses the maintained `MarketDatasetView` contract:

```text
view = MarketDatasetView(source_dataset, fit_start_index, fit_stop_index)
fit_dataset = view.materialize()
```

The implementation requires:

```text
fit_dataset.dataset_id == view.identity
fit_dataset.n_bars     == source.fit_bar_count
fit timestamps[0]      == source.fit_first_timestamp_ns
fit timestamps[-1]     == source.fit_last_timestamp_ns
```

The FIT child is a deterministic in-memory derived view and is **not** a second formal market-data artifact. Therefore formal-source `identity_verified=true` is required before slicing; the materialized FIT child is validated by exact `MarketDatasetView.identity` rather than being republished.

Because the formal source dataset ID content-addresses the source arrays, `MarketDatasetView.identity` transitively binds exact source content and exact absolute FIT range.

### 5.1 Sharing rule

Within one `UniversalTradeRLU2EnvironmentFactory`, each symbol's FIT dataset is materialized once and may be shared as immutable data across all eight in-process workers.

May be shared:

- FIT `MarketDataset` values;
- frozen U1 normalizer;
- immutable U0/U1/U2 contracts and source closure.

Must never be shared:

- `UniversalTradeMarketEnv` / `UniversalTradeEnvironment` instances;
- BookState / portfolio state;
- order / pending-target state;
- reward tracker;
- episode lifecycle / mutable RNG state.

---

## 6. U2 owns production binding derivation

The low-level routed-environment validator may remain for focused tests, but the production training path must not treat caller-authored `InstrumentDatasetBinding` values as authoritative.

For each verified source/FIT view, U2 derives:

```text
concrete_symbol       = source.symbol
source_dataset_id     = fit_dataset.dataset_id
symbol_dataset_digest = source.dataset_digest
split                 = "train"
```

The remaining required digests are exact U2 V1 derived values.

### 6.1 Exact execution binding digest

```text
execution_metadata_digest = content_digest({
    "schema_version": "universal_trade_rl_u2_execution_binding_v1",
    "fit_dataset_id": fit_dataset.dataset_id,
    "u1_execution_policy_digest": u1_contract.execution_policy_digest,
    "u1_pretrade_risk_digest": u1_contract.pretrade_risk_digest,
    "u1_portfolio_risk_digest": u1_contract.portfolio_risk_digest,
})
```

### 6.2 Exact disabled descriptor digest

```text
instrument_descriptor_digest = content_digest({
    "schema_version": "universal_trade_rl_u2_instrument_descriptor_disabled_v1",
    "instrument_context_enabled": false,
    "v4_context_enabled": false,
})
```

This digest does not add an observation channel. Any change to these payloads requires a new versioned U2 generation.

---

## 7. High-level U2 environment factory boundary

U2 owns source/FIT provenance but must not duplicate U1 economic configuration.

The maintained high-level factory is constructed from the semantic equivalent of:

```text
u2_contract: UniversalTradeRLU2Contract
source_closure: U2TrainingSourceClosure
source_artifact_locators: exact Train-symbol mapping
u1_contract: UniversalTradeRLU1Contract
policy_contract: UniversalTradePolicyContract
normalizer: UniversalTradeSequenceNormalizer
u1_environment_factory: Callable[[MarketDataset], UniversalTradeEnvironment]
run_seed: one member seed
```

Construction must reject unless:

```text
source_closure.u2_contract_digest == u2_contract.digest
source_closure.digest is the exact closure consumed by the factory
run_seed in u2_contract.training_seeds
u2_contract training payload fixes n_envs == 8
u2_contract training payload fixes vector_environment_mode == "in_process"
```

The factory derives `training_config_digest`, worker count, and vector mode from `u2_contract`; it must not import the later `universal_trade_rl_u2_training` orchestration module merely to obtain them.

For every worker/symbol, `u1_environment_factory(fit_dataset)` returns a **fresh mutable** U1 environment. U2 then applies the existing frozen U1 environment validator before the worker is usable.

No fallback U1 config, default risk config, alternate execution policy, or hidden context provider is permitted.

The factory protocol is:

```text
factory() == factory.for_environment_index(0)()
factory.for_environment_index(i) for i in 0..7
any other index -> reject
```

---

## 8. U2 episode-seed contract

The generic binding digest is too broad for U2 episode sampling. U2 V1 uses:

```text
payload = {
    "schema_version": "universal_trade_rl_u2_episode_seed_v1",
    "run_seed": run_seed,
    "partition_digest": source_closure.time_partition_digest,
    "environment_index": environment_index,
    "completed_episode_count": completed_episode_count,
    "fit_dataset_id": selected_fit_dataset.dataset_id,
}

episode_seed = int(content_digest(payload)[:8], 16)
```

Consequences:

- member seed, worker index, episode count, partition, or FIT data change sampling;
- unrelated execution/descriptor metadata does **not** change sampling;
- the selected symbol is transitively bound by its symbol-specific FIT dataset ID.

The seed remains an unsigned 32-bit concrete-environment seed.

---

## 9. Member seed vs SB3 worker reset seed

For one PPO member:

```text
member_seed == PPO RNG seed == U2 router run_seed
```

All eight workers share that immutable router `run_seed`.

Worker `i` has:

```text
environment_index = i
canonical_probe_seed = member_seed + i
SB3-facing reset seed = member_seed + i
```

A U2 worker accepts only:

```text
seed is None
or
seed == canonical_probe_seed
```

at its external Gymnasium/SB3 boundary. The external seed does not redefine router `run_seed` and is not the concrete episode seed. The U2 adapter validates the external seed, then delegates the routed reset under the immutable member `run_seed` and Section 8 episode-seed rule.

The generic `EpisodeRoutedSingleInstrumentEnv` reset contract is unchanged.

---

## 10. Shared run-level environment-generation identity

The SB3 backend probes `environment_factory()` before vector construction. U2 worker 0 must therefore expose an identity for the **whole fixed environment generation**, not a worker-0-only identity.

All workers 0..7 expose the same `environment_digest`, computed from this exact payload:

```text
{
    "schema_version": "universal_trade_rl_u2_environment_generation_v1",
    "u2_contract_digest": u2_contract.digest,
    "source_closure_digest": source_closure.digest,
    "training_config_digest": u2_contract.training_config_digest,
    "run_seed": run_seed,
    "n_envs": 8,
    "vector_environment_mode": "in_process",
    "environment_indices": (0, 1, 2, 3, 4, 5, 6, 7),
    "binding_digests": tuple(
        (symbol, internal_binding[symbol].digest)
        for symbol in source_closure Train-symbol order
    ),
    "router_contract_digest": u2_contract.router_contract_digest,
    "episode_sampling_contract_digest": u2_contract.episode_sampling_contract_digest,
    "episode_seed_schema": "universal_trade_rl_u2_episode_seed_v1",
}
```

`environment_index` is intentionally absent as a scalar because this is the run-level generation identity; the complete allowed worker set is bound by `environment_indices`.

Every worker must satisfy:

```text
worker.environment_digest == environment_generation_digest
worker router run_seed      == member_seed
worker environment_index    == requested index
worker router_digest        may differ by worker index
```

This shared digest is consumed by the existing framework-neutral training identity and checkpoint machinery.

### 10.1 Per-seed orchestration

`run_seed` is part of the generation digest, so seed 0/1/2 intentionally have different environment-generation digests. U2 must therefore orchestrate each preregistered seed as its own training member; it must not reuse a generic multi-seed runner contract that requires one identical environment digest across seeds.

### 10.2 Checkpoint compatibility

The existing ordinary checkpoint path already rejects an environment-digest mismatch before model/optimizer state is loaded. No U2-specific `CheckpointManifest` field is required once `environment_digest` is this run-level generation digest.

### 10.3 Exact trajectory resume is still unresolved

Environment/checkpoint identity equality proves data/configuration compatibility only. It does not prove exact continuation of partially completed vector episodes or rollout state.

This amendment does not authorize a claim of exact mid-episode PPO trajectory continuation. That remains a separate technical requirement to resolve before any exact-resume claim.

---

## 11. Required falsification tests

### Source / FIT metadata

- 5-minute phase-shifted FIT grid is rejected before numeric load;
- FIT stop beyond source count is rejected;
- aligned metadata resolves to exact integer start/stop indices.

### Source artifact provenance

- exact canonical frozen source passes;
- same symbol/timestamps/count but different numeric content fails source dataset identity;
- tampered artifact fails;
- unverified source content fails before slicing;
- changing only artifact locator leaves all U2 identities unchanged.

### FIT view

- full source -> exact preregistered `MarketDatasetView`;
- child ID equals view identity;
- first/last/count equal FIT metadata;
- no post-FIT row exists.

### Binding / sampling

- production factory derives bindings internally;
- FIT view ID is `source_dataset_id`; frozen U0 ID is `symbol_dataset_digest`;
- exact execution/descriptor digests match Section 6;
- unrelated descriptor/execution representation cannot perturb Section 8 episode seed;
- changing FIT data/range does perturb episode seed.

### Worker isolation / generation identity

- workers 0..7 own distinct mutable U1/base environments;
- all workers use the same symbol-specific FIT dataset identities and frozen normalizer generation;
- worker indices are exactly 0..7;
- all workers expose the same generation digest;
- changing source closure, binding, member seed, worker-count/vector contract changes generation identity.

### SB3 integration

Using the actual maintained `DummyVecEnv` path with eight U2 workers:

```text
vec.seed(member_seed)
-> [member_seed + 0, ..., member_seed + 7]
vec.reset()
-> succeeds for every worker
```

After reset, each router still reports the common member run seed and its distinct environment index.

### Timeout integration

The robustness/timeout amendment remains mandatory. The actual U2 in-process vector path must preserve timeout metadata, exact terminal observation, and exactly one PPO terminal-value bootstrap without changing economic reward or wealth.

---

## 12. Updated implementation order

```text
Task 4A  source/FIT grid-phase contract
Task 5A  canonical source artifact -> exact FIT MarketDatasetView
Task 5B  U2-owned binding derivation
Task 5C  U2-specific episode-seed contract
Task 5D  shared environment-generation identity
Task 5E  indexed 8-worker factory + mutable-state isolation
Task 6A  SB3 member-seed / worker-reset-seed integration
Task 6B  actual DummyVecEnv integration
Task 6C  timeout metadata + terminal observation
Task 6D  exact one-time PPO bootstrap oracle
Task 6E  minimal fixed PPO orchestration
Task 7   deterministic B/C/D evaluation
Task 8   preregistered Selection gates
```

Task 7 must not begin while any Task 4A-6D mechanics gate is unresolved.

---

## 13. Completion / training-readiness gate

The source/FIT/routing portion cannot be called complete unless one exact final HEAD proves:

- source/FIT metadata alignment;
- canonical source artifact provenance;
- deterministic `MarketDatasetView` derivation;
- U2-owned binding closure;
- sampling independence from unrelated metadata;
- common member run seed across workers;
- exact SB3 worker reset-seed translation;
- fresh mutable state for all eight workers;
- one shared run-level environment-generation identity;
- checkpoint rejection after source/FIT/generation drift;
- unchanged U1 economics;
- existing timeout/bootstrap obligations;
- related/full tests, static checks, architecture checks, package/build checks, and exact-HEAD CI.

Even after this gate passes, real U2 PPO remains separately gated by the real production-candidate U0/U1 freeze and maintained authorization requirements. Production remains **NO-GO** and Admission remains **SEALED**.
