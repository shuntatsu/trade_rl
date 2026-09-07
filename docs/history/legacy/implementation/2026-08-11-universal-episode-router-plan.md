# Universal Single-Instrument Episode Router Implementation Plan

## Conclusion

Implement U2 as a fail-closed Gymnasium facade that routes complete episodes across immutable single-symbol environments while exposing one generic policy-facing instrument contract:

```text
symbols = ("INSTRUMENT",)
action_names = ("target_weight:INSTRUMENT",)
action_shape = (1,)
```

The implementation must not mutate `ResidualMarketEnv.dataset` after construction. Each concrete symbol remains inside its own independently constructed environment and immutable binding. The wrapper selects exactly one concrete environment at reset, delegates the whole episode to it, and switches only after termination or truncation.

This is a stacked change on top of the U1 universal-instrument artifact bundle. It does not change the current BTC training generation, existing profiles, normalization, BC, PPO, reward definitions, serving, or production authorization.

## Scope

### Included

- immutable concrete-dataset and episode binding contracts;
- deterministic balanced train-symbol routing;
- train/validation/test leakage guards;
- generic single-instrument policy-facing symbols and one-action contract;
- a lazy, cached Gymnasium facade over independent concrete environments;
- fail-closed environment schema validation;
- terminal and reset telemetry carrying concrete episode identity;
- universal policy-manifest and concrete deployment-binding identity contracts;
- focused unit, contract, and real-environment integration tests.

### Excluded

- PostgreSQL dataset payload loading;
- target-local feature selection;
- instrument descriptors as observations;
- symbol-balanced normalization;
- universal BC or critic warm start;
- PPO/Lagrangian training orchestration;
- zero-shot evaluation gates;
- checkpoint or serving integration;
- changing any maintained default or active generation.

## Design boundaries

### 1. Concrete binding ownership

`InstrumentDatasetBinding` owns the non-policy identity required to prove which concrete market data is loaded:

```text
concrete_symbol
source_dataset_id
symbol_dataset_digest
execution_metadata_digest
instrument_descriptor_digest
split
```

All digests are lowercase SHA-256 values. `split` must be one of `train`, `validation`, or `test`.

The binding validates a loaded environment before use:

- exactly one dataset symbol;
- the dataset symbol equals `concrete_symbol`;
- the dataset ID equals `source_dataset_id`;
- target-weight action mode;
- exactly one action;
- underlying concrete action name equals `target_weight:<concrete_symbol>`.

### 2. Deterministic balanced router

`DeterministicBalancedInstrumentRouter` receives:

```text
train_symbols
partition_digest
run_seed
environment_index
```

For completed episode count `k`:

```text
cycle, position = divmod(k, len(train_symbols))
```

A cycle-specific permutation is produced by sorting train symbols with canonical content-digest keys derived from the immutable routing identity and cycle number. Therefore:

- every train symbol appears exactly once per cycle;
- no symbol appears twice before all train symbols appear once;
- the same identity and completed count always return the same route;
- different environment indices can produce distinct deterministic orders;
- no process-global or mutable random-number generator is used.

### 3. Training split guard

The training facade requires exact closure:

```text
binding symbols == partition train symbols
all binding.split == "train"
```

Validation/test bindings, missing train bindings, extra bindings, duplicate bindings, or a partition mismatch fail during construction before any environment factory is called.

### 4. Independent environment state

The facade receives an environment factory and lazily creates at most one environment per concrete symbol. Constructed environments are cached and never rebound to another dataset.

Each concrete environment therefore owns independent:

- dataset;
- hybrid and shadow books;
- order books;
- pending targets;
- reward tracker;
- episode sampler;
- execution random state;
- observation runtime state.

A factory failure is propagated. The router never falls back to another symbol.

### 5. Generic policy-facing contract

The facade exposes only:

```text
policy_symbols = ("INSTRUMENT",)
action_names = ("target_weight:INSTRUMENT",)
action_space.shape = (1,)
```

Concrete symbols remain in binding and telemetry records, not in the policy-facing action layout. The observation itself is delegated unchanged because every underlying dataset is single-symbol and must satisfy the same Gymnasium space contract.

The first loaded environment establishes the canonical observation and action spaces. Every later environment must match those spaces exactly before reset is delegated. A mismatch stops the run.

### 6. Episode lifecycle

Initial reset:

1. route from completed count zero;
2. load and validate the selected environment;
3. derive a deterministic per-episode seed;
4. reset the concrete environment;
5. create an immutable `InstrumentEpisodeBinding` from returned episode boundaries;
6. attach the binding payload and digest to reset info.

During an active episode:

- `step` delegates only to the selected concrete environment;
- an early `reset` is rejected;
- `step` before reset or after completion is rejected;
- completed episode count advances only when `terminated or truncated` is true.

Terminal info preserves the ending episode binding so vector-environment auto-reset cannot erase the concrete identity of the transition.

### 7. Seed contract

`run_seed` is immutable construction identity. A reset seed may be omitted or equal `run_seed`; a different value is rejected instead of silently changing the routing sequence.

The concrete episode seed is derived from:

```text
run_seed
environment_index
completed_episode_count
selected binding digest
partition digest
```

and reduced to an unsigned 32-bit value. This separates routing determinism from process scheduling.

### 8. Universal policy and deployment identity

`UniversalSingleInstrumentPolicyManifest` contains only generic/training identity:

```text
architecture_digest
observation_schema_digest
action_schema_digest
instrument_descriptor_schema_digest
normalizer_digest
reward_environment_digest
training_catalog_digest
training_symbol_split_digest
training_symbols_digest
zero_shot_evidence_digest
```

Its canonical payload must not contain a concrete ticker. Its digest is the policy digest.

`SingleInstrumentDeploymentBinding` contains the concrete deployment identity:

```text
policy_digest
concrete_symbol
market_instrument_contract_digest
dataset_feature_schema_digest
execution_metadata_digest
instrument_descriptor_evidence_digest
seen_in_training
```

This separation permits `seen_in_training=false` while keeping concrete asset binding outside architecture identity.

## Failure behavior

The implementation fails closed for:

- fewer than one train symbol;
- invalid or duplicate train symbols;
- non-integer or negative seed/index/count values;
- invalid SHA-256 fields;
- binding closure different from partition train closure;
- any validation/test binding in the training facade;
- environment factory returning the wrong type/contract;
- multi-symbol concrete environments;
- dataset ID or concrete symbol mismatch;
- non-target-weight or multi-action environments;
- observation/action-space mismatch across symbols;
- reset before the prior episode completes;
- step before reset or after completion;
- factory/reset/step failure;
- non-boolean termination flags;
- malformed episode boundaries;
- concrete symbol leakage into the universal policy manifest.

No failing output is converted into a different route or treated as a skipped symbol.

## TDD sequence and commits

### Commit 1 — plan

```text
docs(rl): plan universal episode routing
```

Add this plan only.

### Commit 2 — RED contracts

```text
test(rl): define universal episode routing contracts
```

Add tests that import the not-yet-existing U2 modules and specify:

- immutable binding validation and digests;
- balanced cycle closure and determinism;
- split leakage rejection;
- generic one-action facade;
- lifecycle and terminal telemetry;
- factory failure without fallback;
- schema mismatch rejection;
- policy/deployment identity separation.

Expected RED is missing U2 modules only.

### Commit 3 — binding and router core

```text
feat(rl): add deterministic universal instrument router
```

Implement immutable bindings, canonical serialization, and deterministic balanced routing. Make core contract tests green while facade tests remain RED.

### Commit 4 — routed Gym facade

```text
feat(rl): route complete single-instrument episodes
```

Implement the lazy cached facade, space validation, seed derivation, lifecycle guards, and reset/terminal info enrichment.

### Commit 5 — policy/deployment identity

```text
feat(rl): separate universal policy and deployment identity
```

Implement generic universal policy manifest and concrete deployment binding contracts.

### Commit 6 — integration and review fixes

```text
test(rl): harden universal episode routing boundaries
```

Add real `ResidualMarketEnv` integration tests and any regression tests found during self-review. Update documentation only where needed to match the implemented contract.

## Focused verification

Run on each implementation head:

```text
pytest tests/rl/test_universal_instrument_binding.py -q
pytest tests/rl/test_universal_episode_router.py -q
pytest tests/rl/test_universal_single_instrument_identity.py -q
ruff check changed files
ruff format --check changed files
mypy changed source files
```

## Repository verification

The final exact head must pass:

- full pytest with branch coverage;
- critical branch-coverage ratchets;
- Ruff;
- Ruff format;
- MyPy;
- Import Linter;
- Vulture;
- frontend tests, typecheck, build, and fixed-viewport checks;
- Windows and Ubuntu compatibility;
- training image and packaged non-root runtime probe;
- PostgreSQL specialist workflow when triggered;
- package and uv identity.

## Completion criteria

U2 is software-complete only when:

1. train bindings close exactly over the U1 partition train split;
2. each cycle uses every train symbol once before repetition;
3. each environment handles exactly one concrete symbol for a complete episode;
4. no book/order/reward state is shared across concrete environments;
5. the policy-facing symbol and action contracts remain generic and scalar;
6. validation/test bindings cannot enter the training router;
7. factory, dataset, schema, and lifecycle failures stop the run without fallback;
8. reset and terminal info preserve the concrete episode binding;
9. universal policy identity contains no concrete ticker;
10. exact-head CI is fully green and the stacked PR is Ready for review.

Passing these criteria is infrastructure evidence only. It is not evidence of profitable learning, zero-shot generalization, sealed-test success, or production readiness. Production remains **NO-GO**.
