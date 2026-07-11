# Architecture

This document is the only normative description of the current Trade RL architecture. Code and tests take precedence when a discrepancy is discovered; the discrepancy must then be corrected here.

## Status

Production is **NO-GO** until [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) is complete with evidence.

## Principles

1. One authoritative implementation per production responsibility.
2. Invalid artifacts, state, schemas, credentials, or market data fail closed.
3. Evidence is bound to one exact bundle digest and Git commit.
4. Training evaluation and serving share observation and decision contracts.
5. Online serving is authenticated and read-only.
6. Registry versions are immutable; activation is an atomic pointer update.
7. Account state is explicit and supplied by the Trade Platform on every request.
8. Research findings do not authorize Production behavior.

## System boundaries

### Control Plane

The offline Control Plane owns:

- data preparation and quality gates;
- training, PBT, walk-forward, sealed-holdout evaluation, and statistical checks;
- complete `ServingBundle` construction;
- deployment evidence production and verification;
- immutable registry registration;
- activation and rollback through CLI and GitHub Actions.

The Control Plane does not serve live signals. Its supported interfaces are CLI commands and trusted GitHub Actions workflows.

### Serving Plane

The online Serving Plane exposes only:

- `GET /health`;
- `GET /ready`;
- authenticated `POST /api/signal/latest`.

It does not expose training, model deletion, registry mutation, promotion, or rollback.

## ServingBundle

A bundle is the complete deterministic inference unit:

```text
serving_candidate/
  manifest.json
  model.zip | ensemble/
  metadata.json
  preprocessing.json
  risk.json
```

The bundle contains:

- model version and training Git SHA;
- ordered symbols;
- ordered per-symbol and global feature names;
- feature normalization and zero-mask configuration;
- observation schema and dimension;
- serving-compatible progress mode;
- environment inference settings;
- post-processing configuration;
- guardrail and pre-trade risk configuration;
- evaluation summaries;
- SHA-256 for every file and a canonical bundle digest.

Any digest, file set, dimension, symbol order, feature order, or schema mismatch rejects the bundle.

## Registry

The only registry implementation is `mars_lite.serving.registry.ModelRegistry`.

```text
registry/
  versions/<version>/...   immutable bundles
  active.json              atomic active identity
  activation-history.jsonl
```

Registration copies and revalidates a candidate into an immutable version directory. It never activates the version. Activation validates the registered bundle and atomically replaces `active.json`. Failed registration or activation preserves the previous active version.

## Training and promotion flow

```text
build data
  -> quality and leak checks
  -> P0
  -> PBT on development data
  -> multi-fold walk-forward and cost sensitivity
  -> final training
  -> sealed holdout and baseline gate
  -> complete ServingBundle candidate
  -> immutable registration
  -> Shadow/Canary evidence bound to bundle digest
  -> deployment gate
  -> environment approval
  -> atomic activation
```

A successful training run registers a candidate but does not activate it.

## Online inference flow

The Trade Platform sends:

- request ID and market snapshot identity;
- current weights in bundle symbol order;
- portfolio, day-start, and peak values;
- consecutive-loss and turnover state;
- pending orders;
- optional disagreement and risk-state values required by the model schema.

The Serving Plane executes:

```text
authenticate
  -> claim request ID and reject replay
  -> obtain cached immutable feature snapshot
  -> validate symbols and feature schemas
  -> restore preprocessing
  -> build_observation with real current positions
  -> policy.predict
  -> DecisionPipeline
  -> stateful guardrails using actual order turnover
  -> pending-order-aware PreTradeRiskVerifier
  -> response and structured audit event
```

The Trade Platform is the final execution and risk-enforcement boundary. It must refuse execution unless the response is valid and the risk verdict is approved.

## Shared contracts

- `mars_lite.env.observation.build_observation` is the shared observation builder.
- `mars_lite.trading.pipeline.DecisionPipeline` is the shared action-to-target path.
- `mars_lite.trading.guardrails.evaluate_guardrails` evaluates real account state.
- `mars_lite.trading.pre_trade_risk.PreTradeRiskVerifier` evaluates deltas, pending orders, liquidity, restrictions, and reduce-only behavior.
- `mars_lite.serving.runtime.ServingRuntime` owns cached loading, safe hot-swap, inference orchestration, and readiness.

## Availability and hot-swap

Serving loads the active bundle beside the currently loaded bundle. The new bundle becomes visible only after digest, schema, preprocessing, model-load, and readiness checks succeed. A bad new bundle leaves the old in-memory bundle serving and reports degraded readiness. With no healthy bundle, the signal route returns `503` and no actionable weights.

## Trust boundaries

Control and Serving processes use different credentials. Serving uses a bearer token, origin allowlist, local bind by default, audit logging, and request replay protection. Registry writes require a Control Plane identity. Secrets are supplied by deployment secret management and are not stored in the repository.

## Package map

- `mars_lite/data`, `mars_lite/features` — data and feature construction
- `mars_lite/env`, `mars_lite/learning` — RL environment and training
- `mars_lite/eval` — evaluation and replay simulation
- `mars_lite/pipeline` — offline orchestration
- `mars_lite/serving` — bundle, registry, contracts, runtime, audit, feature snapshots
- `mars_lite/server` — read-only serving HTTP boundary and deployment gate
- `mars_lite/trading` — execution costs, decision processing, guardrails, and pre-trade risk

## Explicit non-guarantees

Passing CI proves that tested contracts hold; it does not prove profitability or Production readiness. Synthetic results, historical backtests, and one-time experiments are not live-trading authorization.
