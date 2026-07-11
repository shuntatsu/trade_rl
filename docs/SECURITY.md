# Security

## Trust boundaries

The Control Plane and Serving Plane are separate processes with separate identities.

- Control Plane: privileged training, evidence, Registry writes, activation, rollback.
- Serving Plane: authenticated, read-only signal delivery.
- Trade Platform: authoritative account state, order placement, and final pre-trade enforcement.

## Authentication and network exposure

`POST /api/signal/latest` requires `Authorization: Bearer <token>`. The expected token is supplied through `TRADE_RL_SERVING_TOKEN` and compared in constant time. Missing credentials return `401`; incorrect credentials return `403`.

Serving binds to `127.0.0.1` by default. CORS uses an explicit allowlist. Wildcard origins are not enabled with credentials.

## Artifact integrity

Every ServingBundle contains SHA-256 digests for all files and a canonical bundle digest. Registration, activation, startup, and hot-swap revalidate the bundle. Version directories are immutable.

Evidence must bind the model version, Git SHA, bundle identity, source run, and evaluation lineage. Candidate-provided thresholds are not trusted.

## Request integrity

The Trade Platform supplies a unique request ID and market snapshot identity. SQLite stores claimed request IDs and immutable audit events. Reuse of a request ID is rejected; reuse with a different payload is treated as an integrity violation.

SQLite is not an account-state source. Current positions and risk state must arrive from the authenticated Trade Platform on every request.

## Fail-closed conditions

No actionable weights are returned for:

- invalid bearer credentials;
- missing or unhealthy active bundle;
- digest, schema, symbol-order, feature-order, or observation-dimension mismatch;
- stale, NaN, all-zero, or malformed market data;
- invalid account values or pending orders;
- request replay;
- guardrail flatten/rejection;
- failed pre-trade risk verdict.

A failed bundle refresh preserves the prior healthy in-memory bundle and marks readiness degraded.

## Secrets

Secrets, live endpoints, private keys, API keys, and operational contact details must not be committed. Use deployment secret management and stage-specific GitHub Environments.

## Threats covered by tests

- path traversal and manifest tampering;
- cross-model evidence reuse;
- non-finite and out-of-range metrics;
- invalid feature masks and schema dimensions;
- replayed request IDs;
- unauthorized signal requests;
- destructive route exposure;
- failed atomic activation and corrupted hot-swap candidates.

## External security actions

Production approval still requires deployment-specific network policy, secret rotation policy, machine identity, reviewer policy, audit retention determination, incident contacts, and GameDay evidence.
