# Stage A Policy-Bound Producer Design

## Scope

This specification covers A6b-2a: proving that a completed Stage A execution replay was produced for the exact A6a request from the retained checkpoint declared by the immutable Stage A plan, with an optional validated serving bundle for framework-independent inference.

A6b-2a sits before the A6b-1 `StageAExecutionPromotionStore`. It does not perform validation selection, sealed-test authorization, evidence aggregation, or final gate evaluation. It also does not own the concrete Binance PostgreSQL slice builder or Stable-Baselines3 environment assembly; those are A6b-2b responsibilities.

## Existing boundaries

A6a already creates exact `StageAEvaluationCellRequest` values and owns validation and sealed-test orchestration.

A6b-1 already:

- binds replay bytes to every request identity;
- recomputes log growth from the verified equity curve;
- verifies execution event and promotion evidence bytes;
- prevents request rebinding;
- exposes `ArtifactBackedStageAEvaluationCellEvaluator`.

The remaining gap is provenance: a caller can currently build a valid A6b-1 replay while merely asserting that a declared checkpoint produced it. A6b-2a closes that gap before publication.

## Rejected approaches

### Merge PR #311 wholesale

PR #311 mixes A4 stage transactions, execution-promotion changes, serving manifest changes, and an alternative Stage A observation constructor. Its observation constructor still accepts caller-supplied log growth, which conflicts with A6b-1. A6b-2a therefore extracts only the provenance requirements and keeps the existing A6b-1 replay format as the sole observation identity.

### Put model loading inside the A6a evaluator

The A6a evaluator must remain a read-only verifier of immutable completed artifacts. Loading models, materializing datasets, or running episodes inside `evaluate()` would make validation and sealed-test reads depend on mutable runtime state.

### Trust a checkpoint path or model object

A filesystem path and an in-memory model are not identities. The producer must reload and validate the checkpoint manifest, bind its manifest digest to the Stage A request, and require the runtime loader to return the same source identity.

## Architecture

### 1. Policy source binding

`StageAPolicySourceBinding` is an immutable request-scoped contract containing:

- `plan_digest`;
- `request_digest`;
- `candidate_id`;
- `seed`;
- `checkpoint_digest`;
- `candidate_config_digest`;
- `checkpoint_policy_digest`;
- canonical checkpoint-manifest relative path;
- optional canonical serving-bundle relative path and bundle digest;
- schema version and content digest.

Construction is policy-only. Baseline requests never have a policy source binding.

Validation reloads the checkpoint manifest through the maintained checkpoint loader and requires:

- manifest digest equals the request checkpoint digest;
- manifest seed equals the request seed;
- manifest training-config digest equals the candidate config digest;
- manifest policy digest equals the binding policy digest;
- request, plan, candidate, and checkpoint identities match the immutable Stage A plan.

When a serving bundle is present, validation also reloads the entire bundle and requires:

- bundle digest equals the declared bundle digest;
- bundle policy digest equals the checkpoint policy digest;
- bundle environment digest equals the checkpoint environment digest;
- the bundle can be resolved by `canonical_policy_loader` using its declared architecture identity;
- all bundle files remain digest- and size-valid.

The binding does not claim that a selected-final release bundle exists for every candidate. Checkpoint-only sources remain valid for pre-selection Stage A evaluation.

### 2. Runtime policy handle

`StageAPolicyRuntimeHandle` contains:

- the loaded policy object;
- source-binding digest;
- checkpoint-manifest digest;
- checkpoint-policy digest;
- optional serving-bundle digest.

A `StageAPolicyRuntimeLoader` protocol loads one validated binding and returns this handle. The producer rechecks every handle identity before invoking the episode executor. This prevents a loader implementation from returning a model loaded from a different checkpoint.

A framework-independent serving-bundle loader is implemented in workflows using `load_serving_bundle` and `canonical_policy_loader`. A checkpoint runtime adapter is supplied by the integrations layer in A6b-2b because it needs the exact environment, algorithm configuration, policy assembly, and fresh-model identity.

### 3. Episode result contract

`StageAEvaluationEpisodeResult` is the only output accepted from a concrete episode executor. It contains:

- `request_digest`;
- nullable policy-source digest for policy cells;
- candidate-config digest;
- canonical actions;
- observation digests;
- complete positive finite equity curve;
- order events;
- terminal book;
- terminal order-book state.

Policy results require the exact source-binding digest. Baseline results require no source binding and must use the independent baseline candidate-config digest.

The result does not contain log growth. A6b-1 recomputes it.

### 4. Policy-bound producer

`StageAExecutionArtifactProducer` receives:

- immutable Stage A plan;
- A6b-1 promotion store;
- baseline candidate-config digest;
- policy-source registry;
- runtime policy loader;
- episode executor.

For a policy request it:

1. validates the request against the plan;
2. resolves and reloads the exact source binding by request digest;
3. loads a runtime handle and verifies source, checkpoint, and policy identities;
4. executes the exact episode request with that handle;
5. validates the returned result identities;
6. builds the A6b-1 replay artifact using the maintained execution-event and promotion evidence contracts;
7. publishes it through `StageAExecutionPromotionStore`;
8. reloads the published artifact and returns its digest.

For a baseline request it skips policy-source lookup, executes through the baseline path, requires the independent baseline config digest, and publishes the same A6b-1 artifact shape.

The producer is idempotent only when the request already resolves to identical immutable bytes. It cannot replace an existing request with different execution results.

## Storage layout

```text
policy-source-root/
  checkpoints/<candidate>/<seed>/checkpoint.json
  bundles/<bundle-digest>/...
  bindings/<request-digest>/<binding-digest>.json
  by-request/<request-digest>.json
```

The request index is the only lookup entry point. All paths are normalized relative paths beneath the configured root. Symlinks, absolute paths, `..`, undeclared bundle files, and non-canonical JSON are rejected.

## Error handling

The implementation fails closed on:

- baseline requests with a policy binding;
- policy requests without a binding;
- request, plan, candidate, seed, checkpoint, config, or policy mismatch;
- checkpoint manifest digest or file mismatch;
- checkpoint policy substitution;
- bundle digest, environment, or policy mismatch;
- unsafe paths or symlinks;
- runtime handle source substitution;
- executor result request or source substitution;
- caller-supplied growth;
- publication of bytes that differ from an existing request binding.

No fallback to a nearby checkpoint, latest checkpoint, highest score, different seed, different bundle, or mutable path is allowed.

## Testing strategy

Tests must prove:

- valid checkpoint-only and checkpoint-plus-bundle bindings reload successfully;
- the plan candidate and seed select exactly one checkpoint digest;
- checkpoint manifest, policy file, binding, index, and bundle tampering are rejected;
- candidate config, policy digest, environment, seed, and checkpoint substitution are rejected;
- a runtime loader cannot return a handle for another source;
- a policy executor cannot return a result for another request or source;
- baseline execution never reads a policy source;
- producer retries are idempotent only for identical bytes;
- produced artifacts are accepted by `ArtifactBackedStageAEvaluationCellEvaluator` and A6a validation orchestration.

## Follow-on A6b-2 work

A6b-2b will implement the exact triplet/fold PostgreSQL dataset materializer, SB3 checkpoint runtime adapter, and maintained conservative execution episode executor.

A6b-2c will build `StageATestSchedule` from maintained evaluation folds, construct the PostgreSQL one-shot sealed-test ledger, and expose validation and sealed-test CLI commands.
