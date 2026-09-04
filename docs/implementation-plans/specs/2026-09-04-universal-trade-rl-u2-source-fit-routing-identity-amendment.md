# Universal Trade RL U2 Source / FIT / Routing Identity Amendment

Status: **Normative U2 V1 amendment**  
Production: **NO-GO**  
Admission: **SEALED**  
Real U2 training: **NO-GO**

This amendment is written before any real U2 PPO training, Development numeric evaluation, Admission access, or economic result. It is a mechanics / provenance amendment discovered by implementation-time falsification review. It must not be relaxed after Development results.

It supersedes only the affected source-loading, routed-environment, seed, vector-worker identity, and checkpoint-lineage statements in:

- `2026-09-03-universal-trade-rl-u2-base-ppo-selection-design.md`
- `2026-09-03-universal-trade-rl-u2-base-ppo-selection.md`
- `2026-09-03-universal-trade-rl-u2-robustness-timeout-amendment.md`

All economic thresholds, algorithm choices, architecture choices, Development / Admission isolation rules, U1 reward semantics, timeout-bootstrap rules, and fixed PPO hyperparameters remain unchanged.

---

## 1. Reason for the amendment

Implementation-time falsification review found five mechanics loopholes before any real U2 economic result.

1. `U2TrainingSource` required both source and FIT to be dense 15-minute grids but did not independently require the FIT grid to be phase-aligned to the source grid.
2. The current U2 environment builder accepted caller-supplied dataset bindings and an arbitrary child-environment factory. It validated the resulting FIT child but did not own the maintained derivation from the frozen U0 source artifact to that child.
3. The generic routed environment derives the episode seed from the complete `InstrumentDatasetBinding.digest`. Therefore unrelated binding metadata can change episode sampling under otherwise identical U2 training identity.
4. Stable-Baselines3 vector seeding offsets worker reset seeds as `member_seed + environment_index`, while the generic routed environment accepts only `reset(seed == run_seed)`. With one shared member run seed this makes workers 1..7 incompatible with the current generic reset contract.
5. The maintained SB3 backend probes `environment_factory()` before vector construction and uses that single environment's `environment_digest` as the training/checkpoint identity. A worker-0-specific digest therefore does not, by itself, bind the complete fixed 8-worker U2 environment generation.

The amendment closes these loopholes without changing U1 economics or introducing a new data format, vector framework, checkpoint schema, or FIT dataset artifact.

---

## 2. Quality contract

### Objective

Create one fail-closed U2 V1 provenance chain:

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

and make the U2 member seed / worker-seed namespace explicit for `n_envs=8`, `vector_environment_mode=in_process`.

### Non-goals

This amendment does **not**:

- run real PPO training;
- open Development or Admission numeric arrays;
- change PPO hyperparameters, architecture, reward, risk, execution, or Selection thresholds;
- create a persistent FIT dataset artifact;
- redesign `MarketDataset`, `MarketDatasetView`, generic `EpisodeRoutedSingleInstrumentEnv`, generic SB3 vectorization, or generic checkpoint schemas;
- claim exact mid-episode / bitwise trajectory resume from an intermediate checkpoint;
- implement Development B/C/D evaluation or Selection.

### Acceptance Criteria

1. FIT bounds are provably aligned to the frozen source grid using metadata before numeric loading.
2. A canonical U0 source artifact must match the frozen source dataset identity before FIT materialization.
3. The FIT child is the deterministic `MarketDatasetView` of that exact source and exact FIT range.
4. Production U2 training does not accept caller-authored `InstrumentDatasetBinding` values as the source of truth.
5. U2 episode sampling is unaffected by unrelated execution/descriptor binding metadata.
6. All eight in-process workers share one immutable U2 member run seed and one run-level environment-generation digest while retaining distinct worker indices.
7. Every worker owns fresh mutable U1 runtime state; immutable FIT datasets / normalizer / contracts may be shared.
8. The SB3-facing reset seed for worker `i` is exactly `member_seed + i`, while the router run seed remains the member seed.
9. Checkpoint environment identity binds the U2 contract, training config, source closure, FIT bindings, member seed, fixed vector mode, and fixed worker count.
10. Existing U1 observation/action/reward/risk/execution semantics remain unchanged.

### Invariants

- Source artifact filesystem paths are locators, not research identity.
- `source.dataset_digest` denotes the frozen U0 source `MarketDataset.dataset_id`.
- Formal source artifacts must carry a verified canonical content identity.
- FIT materialization cannot escape the frozen source or the preregistered FIT interval.
- `MarketDatasetView.identity` is the FIT dataset identity used by the routed child.
- The same FIT dataset identity must be used by every worker for the same symbol and member run.
- `run_seed == preregistered PPO member seed` for U2 V1.
- `environment_index` is a worker coordinate and never redefines `run_seed`.
- U2 policy input remains symbol-independent and contains no instrument/V4 context.
- `ResidualMarketEnv` / U1 remains the sole Risk / Execution / Accounting authority.
- Normal U1 horizon remains `terminated=false`, `truncated=true`, no liquidation.

### Primary Failure Modes

- source/FIT grid phase mismatch;
- wrong source artifact at a valid locator;
- source artifact with matching timestamps but different numeric content;
- source artifact without canonical content identity;
- wrong FIT start/stop or off-by-one range;
- caller binding spoof / drift;
- unrelated binding metadata changing episode start sampling;
- worker 1..7 reset rejection from SB3 seed offsets;
- worker index collapse to zero;
- worker runtime object reuse / cross-worker BookState or pending-order leakage;
- worker-specific data drift hidden by a worker-0 checkpoint identity;
- source closure or FIT dataset drift accepted on checkpoint resume;
- timeout metadata loss at vectorization (covered by the existing timeout amendment).

### Risk

The highest-risk failures can invalidate the scientific interpretation of one seed run while still producing apparently valid training artifacts. Data/source drift, hidden episode-sampling drift, or worker-state contamination is therefore treated as **training-blocking technical NO-GO**, not a warning.

### Test Oracle

Correctness is observed through:

- exact source `dataset_id`, symbol, first/last timestamp, row count, and canonical identity;
- exact FIT absolute `start` / `stop` and `MarketDatasetView.identity`;
- exact internal `InstrumentDatasetBinding` payload;
- exact episode seed and `InstrumentEpisodeBinding`;
- router `run_seed`, `environment_index`, routing cycle/position;
- worker object identity / state transitions;
- shared environment-generation digest across workers;
- SB3 external reset seeds;
- checkpoint `environment_digest` and training-config digest;
- unchanged U1 reward / runtime contract evidence.

### Required Test Layers

- Unit / contract tests for source/FIT arithmetic and identities;
- artifact integration tests using canonical market-dataset artifacts;
- U1/U2 integration tests for binding and routing;
- real `DummyVecEnv` integration for all 8 workers;
- timeout / terminal-observation integration from the existing amendment;
- static analysis, Ruff, format, MyPy, architecture/import checks;
- related suite, full suite, build/package checks, and exact-final-HEAD CI before training readiness.

### Quality Gate

U2 remains training **NO-GO** until all Acceptance Criteria have executable oracles, the required test layers pass on one exact final HEAD, falsification review finds no unresolved Critical/High mechanics issue, and remaining limitations are recorded. Test Green alone is not sufficient.

---

## 3. Source / FIT grid alignment is mandatory

Let:

```text
BAR_NS = U2_DECISION_STEP_NS = 15 minutes
```

For each `U2TrainingSource`, the existing dense-grid equations remain mandatory. In addition, FIT must lie on the exact source grid:

```text
fit_offset_ns = fit_first_timestamp_ns - source_first_timestamp_ns
fit_offset_ns >= 0
fit_offset_ns % BAR_NS == 0

fit_start_index = fit_offset_ns / BAR_NS
fit_stop_index  = fit_start_index + fit_bar_count

0 <= fit_start_index < fit_stop_index <= source_row_count
```

The timestamps implied by these indices must equal the FIT metadata exactly:

```text
source_first_timestamp_ns + fit_start_index * BAR_NS
    == fit_first_timestamp_ns

source_first_timestamp_ns + (fit_stop_index - 1) * BAR_NS
    == fit_last_timestamp_ns
```

A source grid such as `00:00, 00:15, ...` and a FIT grid such as `00:05, 00:20, ...` is invalid even though each grid is independently dense at 15-minute cadence.

Failure is metadata-only technical NO-GO. Numeric arrays must not be opened to repair the mismatch.

---

## 4. Formal source artifact loading

U0 source identity intentionally does not contain a filesystem path. Therefore U2 runtime may receive a symbol-to-artifact locator mapping, but locator values are not identity and must not enter U2 digests.

For each Train source, before any FIT child is materialized, the maintained source loader must load one canonical market-dataset artifact and require:

```text
dataset.identity_verified == true
dataset.dataset_id         == source.dataset_digest
dataset.symbols            == (source.symbol,)
dataset.n_bars             == source.source_row_count
first timestamp            == source.source_first_timestamp_ns
last timestamp             == source.source_last_timestamp_ns
```

The loaded timestamps must also equal the exact dense 15-minute source grid implied by the source metadata.

Artifact/file tampering, a wrong valid artifact at the supplied path, an unverified dataset identity, wrong symbol, wrong row count, or wrong endpoints must fail before FIT materialization and before U1 environment creation.

### 4.1 Locator independence

Moving one byte-identical canonical source artifact from one filesystem path to another must not change:

- source closure digest;
- FIT view identity;
- U2 binding identity;
- U2 environment-generation identity;
- episode sampling;
- checkpoint compatibility.

---

## 5. FIT is an in-memory `MarketDatasetView`, not a new artifact

U2 V1 reuses the maintained `MarketDatasetView` contract.

For one verified source dataset:

```text
start = fit_start_index
stop  = fit_stop_index
view  = MarketDatasetView(source_dataset, start, stop)
fit_dataset = view.materialize()
```

The implementation must require:

```text
fit_dataset.dataset_id == view.identity
fit_dataset.n_bars     == source.fit_bar_count
fit timestamps[0]      == source.fit_first_timestamp_ns
fit timestamps[-1]     == source.fit_last_timestamp_ns
```

The FIT child is an in-memory deterministic derived view. U2 V1 does not publish a second market-dataset artifact for FIT.

Because formal source identity already content-addresses the source arrays, `MarketDatasetView.identity` transitively binds the exact source content and exact absolute FIT range.

### 5.1 Sharing rule

Within one U2 environment-factory instance, each symbol's FIT dataset is materialized once and may be shared as immutable data across all in-process workers.

The following may be shared across workers:

- FIT `MarketDataset` values;
- frozen U1 normalizer;
- immutable U0/U1/U2 contracts and source closure.

The following must never be shared across workers:

- `UniversalTradeMarketEnv`;
- `UniversalTradeEnvironment`;
- BookState / portfolio state;
- order / pending-target state;
- reward tracker;
- episode lifecycle / RNG state.

---

## 6. U2 owns production binding derivation

The low-level routed-environment validator may remain useful for focused tests, but the production U2 training path must not treat caller-authored `InstrumentDatasetBinding` values as authoritative.

For each verified Train source / FIT view, U2 derives the binding internally.

Required semantics:

```text
concrete_symbol       = source.symbol
source_dataset_id     = fit_dataset.dataset_id
symbol_dataset_digest = source.dataset_digest
split                 = "train"
```

`execution_metadata_digest` and `instrument_descriptor_digest` must also be deterministic U2-owned values, not arbitrary caller inputs.

For U2 V1, `execution_metadata_digest` must bind at least:

```text
FIT dataset identity
U1 execution_policy_digest
U1 pretrade_risk_digest
U1 portfolio_risk_digest
```

`instrument_descriptor_digest` must bind the fixed U2 V1 fact that instrument context and V4 context are disabled. It must not create a policy observation channel.

Changing any of these semantics requires a new versioned U2 binding schema / generation.

---

## 7. U1 runtime construction remains injected and validated

U2 owns source/FIT provenance, but it must not duplicate U1 economic configuration.

The maintained U2 environment factory receives a U1 constructor equivalent to:

```python
Callable[[MarketDataset], UniversalTradeEnvironment]
```

For every worker and concrete Train symbol, this constructor must return a fresh mutable U1 environment around the supplied verified FIT dataset.

U2 then applies the existing frozen U1 environment validator and requires exact equality for the frozen U1 policy/normalizer/runtime/execution/risk contracts before the worker can be used.

No fallback U1 config, default risk config, alternate execution policy, or hidden context provider is permitted.

---

## 8. U2 episode-seed namespace

The generic routed environment's complete binding digest is too broad for U2 V1 episode sampling because unrelated descriptor/execution metadata can perturb sampling.

U2 V1 therefore uses a versioned U2-specific episode seed derived only from sampling-relevant identity:

```text
schema_version = universal_trade_rl_u2_episode_seed_v1
run_seed
partition_digest
environment_index
completed_episode_count
fit_dataset_id
```

The resulting seed must fit the unsigned 32-bit concrete-environment seed contract.

Consequences:

- changing member seed changes sampling;
- changing worker index changes sampling;
- changing episode count changes sampling;
- changing partition/FIT data changes sampling;
- changing unrelated execution/descriptor metadata does **not** change sampling.

The selected concrete symbol is already determined by the router and is transitively bound by its symbol-specific `fit_dataset_id`.

---

## 9. Member seed and SB3 worker reset seed are distinct namespaces

For one PPO member:

```text
member_seed == PPO RNG seed == U2 router run_seed
```

All eight workers share that immutable router `run_seed`.

Worker `i` has:

```text
environment_index = i                    # i in 0..7
SB3-facing reset seed = member_seed + i
```

The SB3-facing reset seed does **not** redefine router `run_seed` and is not used directly as the U2 episode seed.

A U2 worker must accept only:

```text
seed is None
or
seed == member_seed + environment_index
```

at the external Gymnasium/SB3 boundary. Internally, the routed base reset remains bound to the immutable member `run_seed` and the U2-specific episode-seed derivation in Section 8.

The generic `EpisodeRoutedSingleInstrumentEnv` reset contract is not changed by this amendment; the translation is U2-specific.

---

## 10. Run-level environment-generation identity

The maintained SB3 backend probes `environment_factory()` before it constructs the 8-worker vector environment. Therefore U2 worker 0 must expose a digest that represents the **whole fixed U2 environment generation**, not merely worker 0.

U2 V1 defines one run-level environment-generation digest shared by workers 0..7.

The canonical payload must bind at least:

```text
schema_version = universal_trade_rl_u2_environment_generation_v1
u2_contract_digest
source_closure_digest
training_config_digest
run_seed
n_envs = 8
vector_environment_mode = in_process
ordered internal binding digests
router schema / contract identity
episode-seed schema / contract identity
```

`environment_index` is intentionally not part of this shared generation digest. Instead, the generation binds the complete fixed worker set `0..7`, and each worker exposes its own router/environment index through runtime telemetry.

Every worker returned by `for_environment_index(i)` must satisfy:

```text
worker.environment_digest == environment_generation_digest
worker.router.run_seed      == member_seed
worker.environment_index    == i
```

The factory must reject indices outside `0..7`.

This shared digest is the `environment_digest` consumed by the existing framework-neutral training identity and checkpoint machinery.

### 10.1 Why a new checkpoint schema is unnecessary

The existing ordinary checkpoint-resume path already requires exact environment-digest equality before loading the policy/optimizer state. Once `environment_digest` is the shared U2 generation digest, it transitively binds the source closure, FIT identities, worker-count/vector contract, and member seed.

No U2-specific field needs to be added to generic `CheckpointManifest` for identity-safe compatibility checking.

### 10.2 Exact trajectory resume remains a separate unresolved requirement

Environment/checkpoint identity equality proves configuration/data compatibility. It does **not** by itself prove exact continuation of a partially completed vector rollout or mid-episode environment state.

This amendment does not authorize an exact mid-episode PPO optimization resume. The existing exact-resume wording remains a separate technical requirement that must be resolved before any implementation claims exact trajectory continuation.

---

## 11. Required falsification tests

At minimum, implementation must add tests proving the following failures and invariants.

### 11.1 Source / FIT metadata

- FIT grid shifted by 5 minutes relative to source grid is rejected before numeric load.
- FIT stop beyond source row count is rejected.
- valid aligned bounds resolve to exact integer start/stop indices.

### 11.2 Source artifact provenance

- canonical source artifact with exact frozen identity passes;
- same symbol/timestamps/count but different numeric content is rejected by source dataset identity;
- tampered artifact is rejected;
- artifact with no canonical content identity is rejected;
- changing only filesystem locator does not change U2 identity.

### 11.3 FIT view

- full source -> exact preregistered FIT `MarketDatasetView`;
- child ID equals view identity;
- first/last/count equal FIT metadata;
- no post-FIT bar exists in the child.

### 11.4 Binding / episode sampling

- U2 training factory derives bindings internally;
- FIT view ID is `source_dataset_id` and frozen U0 source ID is `symbol_dataset_digest`;
- unrelated execution/descriptor metadata cannot change U2 episode seed;
- changing FIT data/range does change episode seed.

### 11.5 Worker isolation / identity

- workers 0..7 have distinct mutable U1/base environments;
- workers use the same verified FIT dataset identities and frozen normalizer generation;
- workers expose indices 0..7 exactly once;
- every worker exposes the same run-level environment-generation digest;
- changing any binding/source closure/member seed changes that generation digest.

### 11.6 SB3 seed integration

Using the actual maintained `DummyVecEnv` path with eight U2 workers:

```text
vec.seed(member_seed)
-> [member_seed + 0, ..., member_seed + 7]
vec.reset()
-> succeeds for every worker
```

After reset, every router still reports the common member `run_seed`, while each worker keeps its own environment index.

### 11.7 Timeout integration

The existing robustness/timeout amendment remains mandatory after this source/vector amendment. The actual U2 8-worker in-process path must preserve timeout metadata, exact terminal observation, and exactly one PPO terminal-value bootstrap without modifying economic reward/wealth.

---

## 12. Updated implementation order

The U2 implementation sequence is amended to:

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

The source/FIT/routing portion of U2 cannot be called complete unless one exact final HEAD proves:

- source/FIT metadata alignment contract;
- canonical source artifact provenance;
- deterministic `MarketDatasetView` FIT derivation;
- internally derived binding closure;
- U2-specific sampling seed independence from unrelated metadata;
- one common member run seed across workers;
- exact SB3 worker reset-seed translation;
- fresh mutable state for all eight workers;
- one shared run-level environment-generation identity;
- checkpoint rejection after source/FIT/generation drift;
- unchanged U1 economics;
- existing timeout/bootstrap requirements;
- related and full test suites, static checks, architecture checks, build/package checks, and exact-HEAD CI.

Even after this gate passes, real U2 PPO training remains separately gated by real production-candidate U0/U1 artifact freeze and the maintained authorization requirements. Production remains **NO-GO** and Admission remains **SEALED**.
