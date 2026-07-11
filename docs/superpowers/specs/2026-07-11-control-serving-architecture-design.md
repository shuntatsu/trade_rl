# Control Plane / Serving Plane Architecture Redesign

Date: 2026-07-11  
Status: Approved design; implementation not started  
Target: PR #7 (`agent/fix-m7-production-blockers-clean`)

## 1. Purpose

This design removes the current production architecture contradictions around serving, model activation, account state, guardrails, preprocessing parity, API trust boundaries, and documentation authority.

The target outcome is a system with:

- one offline control plane for training, evaluation, evidence, registration, promotion, and rollback;
- one online serving plane for authenticated, read-only signal delivery;
- one authoritative model registry;
- one immutable serving bundle contract;
- one stateful observation-building contract shared by evaluation and serving;
- one decision pipeline and one pre-trade risk contract;
- fail-closed behavior at every production boundary.

Production remains **NO-GO** until the implementation and the owner-operated readiness items are complete.

## 2. Scope

### In scope

- split control-plane and serving-plane runtime boundaries;
- replace the competing model lifecycle implementations with one registry;
- define immutable `ServingBundle` artifacts and atomic activation;
- connect deployment evidence to registry activation;
- inject real account and portfolio state before policy inference;
- persist and validate the complete preprocessing and observation schema;
- make guardrails stateful and correctly parameterized;
- cache serving artifacts and support safe hot-swap;
- add serving authentication and remove destructive endpoints from the serving process;
- produce a deterministic pre-trade risk verdict for mandatory enforcement by the Trade Platform;
- replace obsolete tests with tests for the new contracts;
- replace all project-authored Markdown documentation with a coherent, minimal documentation set.

### Out of scope

- implementation of a real exchange-specific `EmergencyExecutionAdapter` without an existing exchange contract;
- choosing real on-call contacts, legal retention periods, or production reviewers;
- inventing secrets, infrastructure endpoints, or organization-specific deployment credentials;
- changing the trading strategy solely to improve profitability;
- redesigning unrelated feature engineering or PPO internals;
- building a new control-plane web UI or management API. Control-plane operations remain CLI and GitHub Actions for this redesign.

## 3. Architectural principles

1. **One authoritative path per production responsibility.** No parallel registries, active pointers, or serving entrypoints.
2. **Fail closed.** Invalid state, schema, digest, account data, authentication, or market data yields no signal.
3. **Evidence binds to the exact artifact.** Reports, Git SHA, model files, preprocessing schema, and bundle digest must agree.
4. **Training and serving share pure contracts.** Observation construction and target-weight transformation are shared code, not duplicated logic.
5. **Online serving is read-only.** Training, deletion, promotion, and rollback are not exposed by the serving process.
6. **Activation is atomic.** A failed registration, validation, or load never replaces the known-good active version.
7. **Runtime state is explicit.** Current positions and account state are authenticated request inputs, never implicit defaults.
8. **Execution ownership is explicit.** Serving computes a risk verdict; the Trade Platform is the authoritative order-enforcement boundary.
9. **Documentation describes the executable system.** Research results and historical experiments cannot authorize Production behavior.

## 4. System boundaries

### 4.1 Control Plane

The control plane is offline and operationally privileged. It owns:

- data preparation;
- training;
- walk-forward and holdout evaluation;
- evidence production;
- serving-bundle construction;
- registry registration;
- promotion and rollback;
- incident and GameDay tooling.

The interface is CLI and GitHub Actions. No control-plane HTTP management service is added in this redesign.

The control plane does not serve live trading signals.

### 4.2 Serving Plane

The serving plane is an online, read-only service. It exposes only:

- `GET /health`;
- `GET /ready`;
- `POST /api/signal/latest`.

It does not expose:

- model deletion;
- training start/stop;
- promotion;
- rollback;
- registry mutation;
- arbitrary filesystem access;
- dashboard or training-management endpoints.

The service loads one validated active `ServingBundle`, caches it, and swaps versions only after complete readiness validation.

The legacy metrics/dashboard server may remain available as an explicitly non-production development process, started through a separate entrypoint. It is not the production signal path.

## 5. Authoritative components

### 5.1 ModelRegistry

There will be one registry implementation and one on-disk contract:

```text
registry/
  versions/
    <version>/
      manifest.json
      model.zip | ensemble/
      metadata.json
      preprocessing.json
      risk.json
  active.json
  activation-history.jsonl
```

`versions/<version>` is immutable after registration.

`active.json` contains only the active version, bundle digest, activation timestamp, and activation evidence identity. It is updated through write-to-temp, flush, and atomic replace.

Registration and activation are separate operations.

Rollback validates a prior immutable version and atomically rewrites `active.json`. It does not rename or copy model files.

The existing `server.model_registry` and `serving.model_store` contracts are migrated into this implementation. Compatibility helpers may exist temporarily, but they must delegate to the one registry and must not maintain another active pointer.

### 5.2 ServingBundle

A `ServingBundle` is the complete unit required for deterministic inference. It includes:

- model or ensemble artifacts;
- ordered symbol universe;
- feature schema and feature names;
- preprocessing configuration, including rank-Gaussian normalization when used;
- feature mask and expected post-mask dimension;
- observation schema version;
- environment inference settings relevant to observation shape;
- post-processing configuration;
- guardrail and pre-trade risk configuration;
- training Git SHA;
- model version;
- evaluation identity;
- file digests and bundle digest.

A bundle is rejected when any file, schema, dimension, symbol order, or digest is inconsistent.

### 5.3 Observation builder

A pure function such as:

```python
build_observation(
    feature_snapshot: FeatureSnapshot,
    inference_state: InferenceState,
    observation_schema: ObservationSchema,
) -> np.ndarray
```

must be used by both evaluation and serving.

`InferenceState` includes at minimum:

- current weights in bundle symbol order;
- portfolio value;
- day-start value;
- peak value;
- disagreement state when applicable;
- pending orders;
- consecutive-loss state;
- turnover mean and standard deviation;
- request ID and market snapshot ID.

Drawdown and gross exposure are derived from these validated inputs rather than independently supplied.

The policy must receive real current positions before `predict()` is called. Applying `prev_weights` only after inference is forbidden.

### 5.4 DecisionPipeline

The decision pipeline remains the shared path for:

1. action projection;
2. post-processing;
3. volatility targeting;
4. drawdown scaling;
5. disagreement scaling;
6. no-trade logic;
7. higher-timeframe gating.

It accepts explicit portfolio and market state. No serving-only duplicate implementation is permitted.

### 5.5 Guardrails and pre-trade risk

Guardrails receive authenticated account state and use:

- actual portfolio value;
- day-start value;
- peak value;
- consecutive losses;
- turnover mean and standard deviation;
- actual proposed turnover: `abs(target - current).sum()`;
- data age and feature health.

`PreTradeRiskVerifier` evaluates current positions, proposed deltas, pending orders, symbol restrictions, liquidity, and reduce-only behavior.

The serving response includes the deterministic pre-trade verdict and all reason codes. The Trade Platform is the authoritative execution boundary and must reject any order plan that is not accompanied by an accepted verdict for the same request ID, market snapshot ID, active version, and bundle digest.

The repository will define and test this interface contract. Until the external Trade Platform enforces it end to end, Production readiness remains NO-GO. Serving never claims an order was executed.

## 6. Model lifecycle and deployment data flow

### 6.1 Training and candidate construction

The training job produces a candidate bundle containing the model and every inference-relevant setting. The bundle manifest contains SHA-256 digests for all files and a deterministic bundle digest.

### 6.2 Evidence generation

Shadow and Canary evaluations run against the exact candidate bundle. Every report records:

- model version;
- bundle digest;
- Git SHA;
- source run ID;
- parent evidence identity where applicable.

Evidence from another model or bundle cannot be reused.

### 6.3 Gate evaluation

Production thresholds are fixed in trusted code or controlled configuration, not supplied by candidate evidence.

The gate rejects:

- non-finite metrics;
- invalid ranges;
- digest mismatch;
- Git SHA mismatch;
- evidence lineage mismatch;
- path traversal;
- active incidents;
- failed source runs;
- invalid approval evidence.

### 6.4 Registration

After gate success, the control plane:

1. recalculates the bundle digest;
2. validates schema compatibility;
3. writes a new immutable version directory;
4. flushes and verifies the registered content;
5. leaves the current active pointer unchanged.

A partial registration is removed or left unreachable and cannot become active.

### 6.5 Activation

Activation occurs only after registration and serving compatibility checks.

The control plane writes a temporary active pointer and atomically replaces `active.json`. Failure preserves the previous active pointer.

### 6.6 Serving load and hot-swap

At startup, serving:

1. reads `active.json`;
2. loads the referenced bundle;
3. verifies digest and schemas;
4. loads the model into memory;
5. builds preprocessing and inference objects;
6. runs a readiness check;
7. exposes readiness only after success.

A background pointer watcher polls `active.json` at a configurable interval. On change, it loads the new bundle beside the current bundle. Only a fully validated, ready bundle replaces the in-memory reference. A failed replacement leaves the old bundle active and sets readiness to degraded.

### 6.7 Rollback

Rollback selects a known immutable prior version, validates it, and atomically changes `active.json`. Serving uses the same hot-swap path to return to the prior version.

## 7. Online inference flow

The authenticated Trade Platform submits a JSON request containing:

- request ID;
- market snapshot ID;
- current weights as a symbol map;
- portfolio value;
- day-start value;
- peak value;
- pending orders;
- consecutive-loss count;
- turnover mean and standard deviation;
- optional disagreement state.

The current weights must cover exactly the bundle symbol set. The server normalizes them into bundle order and rejects missing, duplicate, or extra symbols.

The serving flow is:

```text
Authenticate request
  -> validate request/state/schema
  -> enforce request idempotency
  -> obtain immutable feature snapshot
  -> preprocess according to ServingBundle
  -> build_observation(feature_snapshot, inference_state, schema)
  -> policy.predict()
  -> DecisionPipeline
  -> Guardrails
  -> PreTradeRiskVerifier
  -> response and audit event
```

The response includes:

- status (`ok`, `no_signal`, `rejected`);
- target weights;
- raw and processed weights when policy permits disclosure;
- rejection and scaling reasons;
- active model version;
- bundle digest;
- market snapshot ID;
- request ID;
- data age;
- guardrail result;
- pre-trade risk result.

## 8. Authentication, idempotency, and runtime state

### 8.1 Authentication

The serving plane uses bearer-token machine authentication for the initial implementation:

```text
Authorization: Bearer <TRADE_RL_SERVING_TOKEN>
```

The token is supplied through the environment or deployment secret management and compared using a constant-time comparison.

The service refuses to start without a token unless an explicit insecure-development flag is set. The insecure flag is forbidden in Production documentation and deployment examples.

### 8.2 Idempotency and replay handling

The serving service maintains a small SQLite runtime database for request identity and audit metadata. It is not the authoritative portfolio state store.

For each request ID it stores a canonical request hash and response identity:

- repeating the same request ID with the same canonical hash returns the original response identity or an equivalent cached response;
- repeating the same request ID with a different hash returns `409 Conflict`;
- expired entries are removed according to a configured retention window.

This prevents accidental or malicious reuse of a request ID for different account or market state.

### 8.3 Portfolio and guardrail state ownership

The Trade Platform is the authoritative source of current positions, day-start value, peak value, pending orders, consecutive losses, and turnover statistics. These values are provided on every authenticated request.

Serving validates and consumes them but does not silently reconstruct account state from prior signal requests. This avoids hidden state divergence across restarts or multiple serving instances.

The SQLite runtime database stores audit and idempotency data only.

## 9. Failure handling

### 9.1 Bundle failures

Digest, schema, feature-mask, model-load, or preprocessing failures prevent activation.

If a previous bundle is healthy, serving continues with it and reports degraded readiness. If no healthy bundle exists, the signal endpoint returns `503` with `no_signal`.

### 9.2 Request and state failures

Invalid authentication, symbol sets, non-finite values, inconsistent account values, malformed pending orders, stale data, NaNs, all-zero features, or conflicting request IDs are rejected before actionable weights are returned.

### 9.3 Guardrail failures

Flatten or scale outcomes are explicit. A flatten recommendation is not reported as executed unless a real execution adapter confirms cancellation, reconciliation, reduce-only closure, and zero residual exposure.

### 9.4 Auditability

Every inference, activation, rollback, and rejection emits structured audit data containing request ID, version, digest, market snapshot, reason codes, and timestamps. Secrets and bearer tokens are never logged.

## 10. Security model

- Serving and control-plane processes use separate credentials and deployment identities.
- Serving uses an allowlisted origin policy rather than wildcard CORS. CORS may be disabled entirely for server-to-server deployment.
- Serving binds according to deployment configuration, not an unconditional public default.
- Signal access requires the bearer-token machine identity described above.
- Destructive and management operations are absent from serving.
- Registry writes require control-plane authorization.
- Secrets are supplied through deployment secret management and are never stored in the repository.
- Path traversal, manifest tampering, request-ID conflict, and unauthorized requests are covered by tests.

## 11. Testing strategy

### 11.1 Unit tests

- bundle manifest, digest, and schema validation;
- feature-mask and symbol-order validation;
- preprocessing serialization and restoration;
- stateful `build_observation()` parity;
- correct turnover and guardrail calculations;
- pending-order worst-case risk;
- atomic register, activate, and rollback behavior;
- old active version preservation on failures;
- authentication and request-ID conflict behavior.

### 11.2 Integration tests

- train artifact -> construct bundle -> register -> activate -> serve;
- active bundle version and digest equal served version and digest;
- real current weights alter the policy observation before inference;
- rank-Gaussian normalization and feature masks match training behavior;
- corrupted new bundle leaves the old version serving;
- rollback changes the served version through normal hot-swap;
- serving process exposes no destructive routes;
- Trade Platform contract inputs produce the expected pre-trade verdict.

### 11.3 End-to-end tests

- evidence production -> deployment gate -> registration -> activation -> served identity;
- rollback -> served identity restoration;
- authenticated Trade Platform request -> stateful inference -> risk verdict;
- invalid credentials, tampered artifacts, stale data, and conflicting request IDs fail closed.

### 11.4 Regression gates

The repository must continue to pass:

- Ruff lint;
- Ruff format check;
- mypy;
- full pytest suite;
- coverage threshold of at least 70%;
- focused M7 and adversarial tests;
- P0 and walk-forward contracts where the test environment supports them.

Obsolete tests are replaced with tests for the current contract rather than deleted without equivalent coverage.

## 12. Runtime entrypoints

The final architecture uses explicit entrypoints:

- control-plane training/evaluation CLI;
- control-plane registry/deployment CLI;
- serving-plane startup command that starts only the read-only signal service;
- development-only metrics/dashboard startup command.

`scripts/run_server.py` starts the production read-only serving plane. The legacy mixed-responsibility metrics server moves to an explicitly named development/dashboard entrypoint.

## 13. Documentation redesign

The project-authored Markdown documentation is consolidated into this normative set:

- `README.md` — repository entrypoint, current status, minimal commands;
- `docs/ARCHITECTURE.md` — only normative current-state architecture;
- `docs/OPERATIONS.md` — deployment, rollback, incident response, and GameDay;
- `docs/SECURITY.md` — authentication, authorization, secrets, and threat model;
- `docs/MODEL_LIFECYCLE.md` — training, evidence, registration, activation, rollback;
- `docs/TESTING.md` — test layers, CI, and acceptance gates;
- `docs/PRODUCTION_READINESS.md` — GO/NO-GO evidence checklist;
- `docs/DECISIONS.md` — concise ADR-style decisions;
- `docs/RESEARCH_HISTORY.md` — historical experiments and non-authoritative findings.

All other project-authored Markdown files are reviewed, merged into the normative set when still valid, and deleted when redundant or obsolete. Historical benchmark claims may remain only in `RESEARCH_HISTORY.md` with scope, dataset, date, and limitations. They cannot be presented as Production authorization.

Licenses, generated third-party documentation, dependency metadata, contribution templates required by GitHub, and files required by external tooling are not deleted solely for consolidation.

## 14. Migration sequence

1. Introduce bundle and registry contracts with failing tests.
2. Consolidate registry implementations and migrate existing model artifacts.
3. Extract and test the stateful observation builder.
4. Build the read-only serving application around a cached active bundle.
5. Correct guardrail and pre-trade risk inputs.
6. Add authentication, idempotency, and audit persistence.
7. Remove management routes from serving and separate the development dashboard entrypoint.
8. Connect deployment gate to registration and atomic activation.
9. Add hot-swap and rollback integration tests.
10. Switch the production server entrypoint.
11. Replace the Markdown documentation set.
12. Run full CI and architecture acceptance checks.

Each migration step must preserve a runnable branch and must not activate a new production path before its integration tests pass.

## 15. Acceptance criteria

The redesign is complete only when all of the following are true:

- one registry and one active pointer exist;
- the deployed, active, and served version/digest are provably identical;
- serving policy inference uses real current positions and account state;
- complete preprocessing is restored from the active bundle;
- bundle incompatibility fails closed;
- serving is read-only and authenticated;
- management and serving trust boundaries are separate;
- guardrails use real state and correct turnover;
- serving produces a deterministic pre-trade verdict and the external enforcement contract is tested;
- hot-swap and rollback preserve availability and identity correctness;
- duplicate request IDs cannot represent different input state;
- obsolete tests are replaced by contract-level tests;
- all CI gates pass;
- the new documentation set contains no conflicting source of architectural authority;
- Production readiness remains NO-GO until external owner actions have attached evidence.

## 16. External owner actions

The following cannot be completed from repository code alone and remain explicit Production blockers:

- real exchange/platform `EmergencyExecutionAdapter`;
- Trade Platform enforcement of the pre-trade verdict contract;
- GitHub Environment protection and required reviewers;
- trusted evidence-producing workflow identity and branch restrictions;
- production secrets and machine identity configuration;
- real on-call and compliance contacts;
- legal determination of applicable jurisdiction and retention periods;
- testnet GameDay and emergency-flatten evidence;
- final operational approval.
