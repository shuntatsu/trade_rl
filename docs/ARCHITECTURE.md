# Architecture

This document is the only normative description of the current Trade RL architecture. Code and tests take precedence when a discrepancy is discovered; the discrepancy must then be corrected here.

## Status

Production is **NO-GO** until [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) is complete with evidence.

## Principles

1. One authoritative implementation per production responsibility.
2. Invalid artifacts, state, schemas, credentials, market data, or release identities fail closed.
3. Evidence is bound to one exact bundle digest and Git commit.
4. Training evaluation and serving share observation and decision contracts.
5. Online serving is authenticated and read-only.
6. Registry versions are immutable; activation is an atomic pointer update.
7. Account state is explicit and supplied by the Trade Platform on every request.
8. Research findings and override runs do not authorize Production behavior.
9. Deployment is incomplete until the live Serving Plane reports the approved version, bundle digest, and running release Git SHA.

## System boundaries

### Control Plane

The offline Control Plane owns:

- data preparation and quality gates;
- training, PBT, walk-forward, sealed-holdout evaluation, and statistical checks;
- release-eligibility derivation from resolved pipeline state;
- complete risk-policy validation;
- complete `ServingBundle` construction;
- deployment evidence production and verification;
- immutable registry registration;
- activation and rollback through CLI and trusted GitHub Actions.

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
- immutable release eligibility, including override and mandatory-gate state;
- proof that the sealed holdout was used;
- ordered symbols;
- ordered per-symbol and global feature names;
- feature normalization and zero-mask configuration;
- observation schema and dimension;
- serving-compatible progress mode;
- environment inference settings;
- post-processing configuration;
- complete guardrail and pre-trade risk configuration;
- evaluation summaries;
- SHA-256 for every file and a canonical bundle digest.

A release bundle is rejected when it was forced, skipped a mandatory gate, lacks sealed-holdout proof, contains a failed mandatory gate, or lacks any required risk limit. Risk liquidity caps must exactly cover the ordered bundle symbols.

Any digest, file set, dimension, symbol order, feature order, schema, release eligibility, or risk-policy mismatch rejects the bundle.

## Registry

The only registry implementation is `mars_lite.serving.registry.ModelRegistry`.

```text
registry/
  versions/<version>/...   immutable bundles
  active.json              atomic active identity
  activation-history.jsonl
```

Registration copies and revalidates a candidate into an immutable version directory. It never activates the version. Activation validates the registered bundle and atomically replaces `active.json`. Failed registration or activation preserves the previous active version.

The deployment runner must access the same persistent Registry storage as the stage Serving Plane. A GitHub-hosted ephemeral filesystem is not a deployment target.

## Training and promotion flow

```text
build data
  -> quality and leak checks
  -> sealed development/holdout split
  -> P0
  -> optional PBT on development data
  -> mandatory multi-fold walk-forward and cost sensitivity
  -> final training on development data
  -> Gate 2 on sealed holdout
  -> derive immutable release eligibility
  -> validate explicit release risk policy
  -> complete ServingBundle candidate
  -> immutable registration
  -> Shadow/Canary evidence bound to bundle digest
  -> deployment gate and environment approval
  -> atomic activation in persistent stage Registry
  -> live /ready identity verification
```

A successful eligible training run registers a candidate but does not activate it. `--force`, `--skip-p0`, `--skip-wf`, or `--skip-gate` makes a run ineligible for candidate construction and registration. `--skip-pbt` is recorded but does not by itself disqualify a release because PBT is an optimization step rather than a safety gate.

## Residual Walk-Forward research boundary

Residual Walk-Forward is a research-only Control Plane orchestration path. `mars_lite.pipeline.residual_walk_forward` owns dataset construction, immutable configuration resolution, fold-local alpha fitting, A/B/C/D candidate training and selection, outer-OOS execution, artifact staging, and atomic publication. `mars_lite.eval.residual_walk_forward` contains only pure evaluation primitives for fold construction, stitched OOS statistics, and strict report shaping; it must not import the pipeline layer.

Each executable fold separates policy training, checkpoint validation, configuration selection, and outer OOS with purge gaps. Checkpoint validation and configuration selection never overlap. A run must have at least two completed folds or fail closed without publishing a new success report.

All artifacts are written below a run-specific staging directory. A successful run is moved atomically into `residual_wf_runs/<run_id>/`, and only then is `residual_walk_forward.json` atomically replaced. Failed partial runs are isolated below `failed/<run_id>/`; they never mix with a prior successful run.

The authoritative aggregate is a stitched OOS path built from chronological, non-overlapping base-bar hybrid and shadow return series. Fold means and medians are supplemental, not total Walk-Forward performance. Model provenance is content-bound by SHA-256, and the 1x and 2x cost evaluations must reference the same selected model digest.

Residual Walk-Forward output remains research-only. It is not sealed release evidence, cannot register a Registry version, and does not change Production from **NO-GO**.

## Deployment identity handshake

For Canary and Production, the deployment workflow must:

1. validate evidence against the exact immutable ServingBundle;
2. use a self-hosted deployment runner with access to persistent stage Registry storage;
3. register and atomically activate the approved version;
4. poll the configured stage `/ready` endpoint;
5. require the reported `active_version`, `bundle_digest`, and `release_git_sha` to equal the approved identity.

A `degraded` response is acceptable only when it reports the newly approved identity. A degraded response still serving a previous bundle fails deployment. An unreachable or mismatched endpoint also fails deployment and requires an explicit operator rollback decision.

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
- `mars_lite.pipeline.release_eligibility.derive_release_eligibility` is the only release-classification path.
- `mars_lite.pipeline.release_risk.load_release_risk_policy` validates release risk policy files.
- `mars_lite.serving.runtime.ServingRuntime` owns cached loading, safe hot-swap, Git-SHA binding, inference orchestration, and readiness.

## Availability and hot-swap

Production serving starts with `TRADE_RL_RELEASE_GIT_SHA` and strict release binding enabled. The active bundle Git SHA must equal the running release SHA.

Serving loads the active bundle beside the currently loaded bundle. The new bundle becomes visible only after digest, release eligibility, risk policy, schema, preprocessing, Git-SHA binding, model-load, and readiness checks succeed. A bad new bundle leaves the old in-memory bundle serving and reports degraded readiness. With no healthy bundle, the signal route returns `503` and no actionable weights.

## Trust boundaries

Control and Serving processes use different credentials. Serving uses a bearer token, origin allowlist, local bind by default, audit logging, and request replay protection. Registry writes require a Control Plane identity. Deployment uses a dedicated self-hosted runner label and stage-scoped GitHub Environment variables. Secrets are supplied by deployment secret management and are not stored in the repository.

## Package map

- `mars_lite/data`, `mars_lite/features` — data and feature construction
- `mars_lite/env`, `mars_lite/learning` — RL environment and training
- `mars_lite/eval` — pure evaluation primitives and replay statistics
- `mars_lite/pipeline` — offline orchestration and release eligibility/risk validation
- `mars_lite/serving` — bundle, registry, contracts, runtime, audit, feature snapshots
- `mars_lite/server` — read-only serving HTTP boundary and deployment gate
- `mars_lite/trading` — execution costs, decision processing, guardrails, and pre-trade risk

## Explicit non-guarantees

Passing CI proves that tested contracts hold; it does not prove profitability or Production readiness. Synthetic results, historical backtests, and one-time experiments are not live-trading authorization.

## Local validation boundaries

The Control Plane's P0 gate preserves the candidate `horizon` and `decision_every`; `--p0-days` controls only synthetic runtime. The Serving Plane creates a content-addressed snapshot from the exact inference inputs after selecting the latest completed bar and computes staleness from bar close.

`mars_lite.server.signal_server` remains authoritative. The legacy dashboard is gated by `TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`. The filesystem Registry is deliberately single-node and is not a distributed coordination mechanism.
