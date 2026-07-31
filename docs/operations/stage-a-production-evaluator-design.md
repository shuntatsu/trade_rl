# Stage A Production Evaluator Design

## Scope

This specification covers A6b-1: the fail-closed trust boundary between completed conservative execution artifacts and the A6a `StageAEvaluationCellEvaluator` protocol.

It does not load a checkpoint or execute a market episode. Checkpoint loading, dataset materialization, episode execution, CLI wiring, and PostgreSQL construction remain in A6b-2. A6b-1 makes those later producers prove that every published result belongs to the exact Stage A evaluation cell requested by A6a.

## Existing context

A6a already owns:

- the immutable Stage A evaluation plan;
- deterministic validation and selected-only sealed-test iteration;
- shared baseline reuse per triplet, fold, and seed;
- validation selection recomputation before test access;
- one-shot fold authorization;
- v2 evidence and gate construction.

A6b-1 must therefore implement only `StageAEvaluationCellEvaluator.evaluate(request)` and must not duplicate A6a selection or ledger logic.

## Rejected approaches

### Complete PR #311 as one unit

PR #311 mixes execution evidence, symbol-triplet stage state, structured serving manifests, training CLI changes, and Stage A observation construction. Its 33-file scope makes a failure difficult to attribute and combines A4 and A6 concerns.

### Trust caller-supplied log growth and cell labels

A wrapper that accepts `policy_log_growth`, `triplet_id`, and `checkpoint_digest` as independent arguments can combine a valid replay with forged evaluation labels. A6b-1 instead derives growth from verified replay bytes and requires the replay identity to equal the complete A6a request.

### Generate executions inside the evaluator

Loading checkpoints and running market episodes inside `evaluate()` would couple the trust boundary to data access, GPU runtime, and serving implementation details. The evaluator consumes immutable completed artifacts. A6b-2 will provide the producer.

## Architecture

### 1. Exact execution-cell identity

`StageAExecutionCellIdentity` binds one execution replay to:

- `request_digest`;
- `plan_digest`;
- `split`;
- `triplet_id`;
- `fold`;
- `seed`;
- nullable `candidate_id` and `checkpoint_digest` for baseline requests;
- `candidate_config_digest`;
- `dataset_identity`;
- `feature_identity`;
- `execution_identity`;
- `evaluation_identity`.

Policy cells require candidate and checkpoint identities. Baseline cells require both to be null. The identity digest is content-addressed and all SHA-256 fields are validated.

The baseline candidate configuration digest is not inferred from a policy candidate. It is a separately configured immutable digest, so one shared baseline can be reused across all candidates for the same triplet, fold, and seed.

### 2. Strict replay artifact

`StageAExecutionReplayArtifact` contains:

- the exact cell identity;
- canonical action vectors;
- canonical observation digests;
- the complete equity curve;
- the existing execution-event artifact bytes by digest and size;
- the existing execution-promotion evidence bytes by digest and size;
- a schema version and content digest.

The replay loader:

1. opens only regular non-symlink files;
2. enforces strict JSON field closure and canonical encoding;
3. verifies the referenced event and promotion files by size and SHA-256;
4. loads and validates existing execution evidence;
5. validates the event artifact against that evidence;
6. confirms dataset and execution identities match the cell identity;
7. requires at least two positive finite equity values;
8. recomputes log growth as `log(last / first)`.

Caller-supplied growth is never accepted.

### 3. Content-addressed promotion root

`StageAExecutionPromotionStore` publishes and loads one immutable replay per request digest.

Canonical layout:

```text
root/
  events/<sha256>.order-events.json
  evidence/<sha256>.execution-evidence.json
  cells/<request-digest>/<artifact-digest>.stage-a-cell.json
  by-request/<request-digest>.json
```

`by-request/<request-digest>.json` is the only lookup entry point. It binds the request digest to one artifact digest and canonical relative path. Publication is exclusive and idempotent only when existing bytes are identical. A request digest cannot be rebound to different execution bytes.

### 4. Artifact-backed evaluator

`ArtifactBackedStageAEvaluationCellEvaluator` accepts:

- a promotion-store root;
- the immutable baseline candidate configuration digest.

For each request it:

1. loads the replay through the request index;
2. compares every cell identity field to the request;
3. requires a policy cell's candidate configuration digest to equal `plan.candidate(candidate_id).candidate_config_digest` through a supplied candidate-config resolver;
4. requires a baseline cell's candidate configuration digest to equal the configured baseline digest;
5. returns `StageAEvaluationCellResult` using the request digest, verified execution-evidence digest, and recomputed log growth.

The evaluator does not cache mutable filesystem state. Repeated loads revalidate the immutable bytes.

## Interfaces

```python
class StageACandidateConfigResolver(Protocol):
    def candidate_config_digest(self, candidate_id: str) -> str: ...


class ArtifactBackedStageAEvaluationCellEvaluator:
    def evaluate(
        self, request: StageAEvaluationCellRequest
    ) -> StageAEvaluationCellResult: ...
```

The Stage A plan itself can satisfy the resolver through a small adapter. The evaluator module must depend on the public A6a contracts, not private gate helpers.

## Error handling

The implementation fails closed on:

- unsafe paths or symlinks;
- non-canonical JSON;
- missing or extra fields;
- artifact digest or size mismatch;
- request-index rebinding;
- event/evidence mismatch;
- dataset, feature, execution, evaluation, plan, split, triplet, fold, seed, candidate, checkpoint, or request mismatch;
- candidate-dependent baseline configuration;
- non-positive or non-finite equity;
- an equity curve with fewer than two values.

No fallback to caller-supplied growth, unverified paths, or a different request index is allowed.

## Testing strategy

Tests must prove:

- a valid policy and baseline replay return recomputed growth;
- request lookup is content-addressed and idempotent;
- a request cannot be rebound to different bytes;
- policy candidate and checkpoint substitution are rejected;
- split, triplet, fold, seed, feature, and evaluation substitution are rejected;
- baseline configuration substitution is rejected;
- event, evidence, equity, manifest, and index byte tampering are rejected;
- unsafe paths and symlinks are rejected;
- A6a validation can consume the evaluator while retaining one shared baseline result per cell.

## A6b-2 boundary

A6b-2 will:

- resolve retained serving bundles and checkpoints through the canonical policy loader;
- materialize the exact triplet and fold dataset;
- run the maintained conservative execution model;
- publish the event, promotion, and Stage A cell artifacts defined here;
- build `StageATestSchedule` from the maintained evaluation source;
- construct the PostgreSQL sealed-test ledger;
- expose validation and sealed-test CLI commands.
