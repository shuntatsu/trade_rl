# Studio Data Truthfulness Design

## Problem

Trade RL Studio is a read-only research console. Its most important contract is not visual polish but semantic honesty: two series shown on one chart must share one scale, action markers must describe the displayed asset, polling must publish successful responses even on slow systems, and an automatically selected job must automatically load its log.

The current `main` implementation violates those contracts in four places:

1. `LineChart` derives a separate Y-domain for RL and Baseline.
2. `MarketReplayChart` derives BUY/SELL from the signed sum of all asset-weight changes while displaying the first asset's price.
3. `useTrainingTelemetry` and `useTrainingMetrics` use fixed `setInterval` polling; a later request invalidates an earlier still-valid response.
4. `RunCenterPage` selects the first job after loading the list but does not load its log.

## Scope

This change fixes only those four high-priority defects and adds regression coverage. It does not add Serving Monitor stale-state presentation, implement Settings, wire the Dashboard detail button, change typography or responsive layout, or update the documented port.

## Considered approaches

### A. Local minimal fixes in the existing components and hooks

Compute one Dashboard domain, use the primary asset weight delta, replace each interval with a completion-scheduled timeout, and invoke `loadLog` after resolving the selected job.

Advantages: smallest review surface, no API contract changes, low conflict risk with the active Stage A work, and direct regression tests.

Disadvantages: telemetry and metrics retain two similar polling loops.

### B. Introduce a generic serial-polling hook

Extract a reusable `useSerialPolling` hook and migrate both telemetry and metric polling.

Advantages: removes duplication and centralizes lifecycle behavior.

Disadvantages: expands scope, creates a new abstraction before a third consumer exists, and increases the chance of lifecycle regressions.

### C. Move chart direction and domains into backend payloads

Add explicit `primarySymbolWeightDelta` or `positionDirection` fields and precomputed chart-domain metadata to API responses.

Advantages: strongest long-term semantic contract.

Disadvantages: requires backend schemas, guards, serializers, compatibility handling, and coordinated rollout. It is larger than the approved frontend-only repair.

## Decision

Use approach A. It directly repairs the four audited defects without changing backend contracts or introducing unrelated abstraction. Backend-explicit direction remains a valid later improvement.

## Detailed design

### Dashboard shared Y-domain

`LineChart` will derive one domain from every finite `rl` and `baseline` value in `points`, while preserving the existing 0.8 lower anchor and 2.0 upper anchor. Both paths receive that same domain. Empty data behavior and chart dimensions remain unchanged.

A regression test renders an out-of-range RL series together with a smaller Baseline series and verifies that both SVG paths use coordinates from the same domain. This catches any future return to per-series normalization.

### Primary-asset BUY/SELL direction

`MarketReplayChart` will calculate direction from index 0 only:

```ts
(record.weightsAfter[0] ?? 0) - (record.weightsBefore[0] ?? 0)
```

This matches the chart's displayed primary symbol instead of netting unrelated assets. A focused component test uses offsetting multi-asset changes where the total delta is zero but the first asset decreases; the marker must be SELL.

### Completion-scheduled polling

Each polling effect will run one refresh immediately. Only after that promise settles will it schedule the next run with `window.setTimeout`. Cleanup marks the loop inactive, clears the pending timeout, and increments the existing request generation so late responses cannot publish after unmount or identity changes.

The public `refresh` function and existing request/generation validation remain unchanged. This design removes automatic overlap without changing API signatures or adding abort support.

Regression tests use fake timers and deferred promises. Advancing beyond the interval while the first request is unresolved must not start a second request. After the first request resolves, advancing one full interval must start the next request.

### Run Center initial log

After `loadJobs` returns, `refreshJobs` will resolve `next` from the current selection and returned jobs, update the selection and URL, and immediately await `loadLog(next)`. If there is no job, it will clear the selection and log lines and invalidate any pending log request.

The existing test will no longer click the selected row before asserting the log. This proves the initial selection is operational rather than cosmetic. The cancellation assertion remains to protect the existing behavior.

## Error and race handling

The existing request counters remain authoritative for stale-response rejection. Serial polling prevents interval-driven requests from invalidating one another. Job-log requests continue to use `logRequest`; a later explicit selection still wins over an earlier automatic load.

No stale backend response is accepted after job, seed, tag, stream-generation, or component lifecycle changes.

## Verification

The change requires:

- focused Vitest regression tests for all four defects;
- the complete Studio Vitest suite;
- Studio TypeScript typecheck;
- production build;
- fixed viewport layout checks;
- repository CI on one unchanged final head.

The PR remains draft until those checks are green.