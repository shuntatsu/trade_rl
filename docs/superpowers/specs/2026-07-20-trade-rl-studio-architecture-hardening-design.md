# Trade RL Studio Architecture Hardening Design

## Status

Approved by the user's instruction to fix the findings from the Studio architecture audit.

## Goal

Turn Trade RL Studio from a trusted single-process prototype into a fail-closed local research console whose resource identities, job lifecycle, comparison semantics, and frontend state remain correct across restarts, duplicate roots, malformed artifacts, and slow or out-of-order requests.

## Safety boundary

- Studio remains loopback-only. Remote binding is removed rather than protected by a partial authentication design.
- No exchange order routing, API-key handling, live activation, or fund mutation is added.
- All research and serving states remain `NO-GO`.

## Backend boundaries

`StudioCatalog` remains the public facade but delegates to focused dataset, config, run, system, and overview services. Catalog entries use opaque resource IDs derived from relative location and canonical identity, so duplicate human run IDs across roots cannot resolve ambiguously.

Dataset and run validation results are cached only while a stat-based artifact fingerprint remains unchanged. A cache miss performs the existing canonical full validation. Audit endpoints always validate the selected immutable artifact before returning evidence or comparison data.

Training requests contain `configResourceId`, `datasetResourceId`, and `runId`; callers cannot submit filesystem paths or choose arbitrary output roots. The server resolves the IDs through the validated catalog and revalidates the canonical config and dataset before spawning the fixed CLI argument vector.

## Job lifecycle

A `JobStore` owns atomic JSON records, per-record locks, run reservations, schema versioning, owner instance IDs, and legal state transitions. Reservation creation uses exclusive filesystem creation, preventing duplicate run submission across processes. Temporary record names are unique and fsynced before replacement.

A restarted Studio process may observe a detached live worker but never terminate it. Detached jobs remain `running` with `cancellable=false`; cancellation raises a typed ownership error without first writing `cancelling`. PID start tokens prevent a reused PID from being treated as the original worker when `/proc` is available.

## Comparison semantics

Every comparison returns an eligibility object with `COMPARABLE`, `PARTIALLY_COMPARABLE`, or `NOT_COMPARABLE` and explicit reasons. Dataset identity, walk-forward schema, stitch mode, fold indices, test ranges, and return lengths define alignment. A not-comparable pair returns configuration differences but no decision metrics, fold deltas, or wealth curve. Series labels use sealed test indices when available.

## Evidence and serving integrity

Evidence inspection validates proposal and authorization payload identities against the run manifest and checks the proposal's dataset, walk-forward, and gate bindings. Paper snapshot content is canonical-digested and compared with `snapshotDigest`; a mismatched snapshot is reported as failed telemetry without changing bundle state.

## API errors

Studio-specific exceptions distinguish missing resources, invalid client requests, artifact failures, identity conflicts, and lost job ownership. FastAPI maps these exceptions explicitly instead of inferring status from broad built-in exception classes.

## Frontend contracts and state

All endpoint payloads receive runtime structural validation. Dashboard startup distinguishes `LIVE`, `OFFLINE`, and explicit `DEMO` modes; backend failure never silently becomes demo research data. Fixed telemetry such as the synthetic `42°C` value is removed.

Workspace, selected run pair, evidence run, and selected job are stored in URL search parameters. Request sequence guards prevent older comparison or log responses from overwriting a newer selection.

## Verification

Tests cover cross-instance duplicate submission, restart-safe cancellation, stale PID tokens, canonical config and dataset rejection, resource collisions, cache reuse and invalidation, comparison eligibility, evidence binding, snapshot digests, typed API errors, runtime frontend guards, offline/demo separation, URL restoration, and stale-response suppression. Existing Python, frontend, type, build, layout, architecture, compatibility, and training-image checks remain required.
