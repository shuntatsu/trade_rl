# Local Validation Hardening Design

Status: approved direction from the architecture follow-up.

## Goal

Make local end-to-end validation trustworthy without expanding the project into a distributed production platform.

This increment must ensure that:

- P0 validates the actual candidate timing configuration rather than a hidden fixed configuration;
- a feature snapshot identity changes whenever the inference-relevant content changes;
- staleness is computed from the close time of the latest completed bar for the bundled base timeframe;
- the legacy metrics server cannot be mistaken for the authoritative Serving Plane;
- an offline local GameDay can demonstrate the critical fail-closed behaviors with no exchange access;
- the existing filesystem Registry remains the supported local single-node implementation.

Production remains **NO-GO** after this increment.

## Scope

### Included

1. Candidate-aligned P0 configuration.
2. Content-addressed `FeatureSnapshot.snapshot_id`.
3. Timeframe-aware completed-bar selection and staleness.
4. Explicit development-only isolation for `mars_lite.server.metrics_server`.
5. Offline local GameDay command and tests.
6. English and Japanese documentation updates.

### Excluded

- distributed or remotely coordinated Registry implementations;
- Kubernetes, database-backed activation, consensus, or multi-writer locking;
- a real exchange `EmergencyExecutionAdapter`;
- testnet order submission;
- Production GO evidence or operational sign-off;
- replacement of the legacy training dashboard.

## Considered approaches

### Approach A: documentation-only clarification

Explain the six findings without changing code.

This is insufficient because P0 and staleness currently produce materially misleading local results, and the snapshot identity cannot prove which values were used for inference.

### Approach B: minimal local correctness hardening — selected

Fix the candidate/P0 mismatch, content-address the in-memory snapshot, make freshness timeframe-aware, isolate the legacy server, and add an offline GameDay harness. Keep the single-node Registry unchanged.

This gives a trustworthy local verification path with bounded scope and preserves the current architecture.

### Approach C: full production infrastructure expansion

Replace the Registry with a networked service, add testnet execution, deploy multiple Serving instances, and automate operational GameDays.

This is premature and would mix local correctness work with infrastructure and ownership decisions that require a real deployment environment.

## Design

## 1. Candidate-aligned P0

`production_pipeline.run()` must stop overwriting `args.horizon` and `args.decision_every` with `4` and `1`.

P0 will use the resolved candidate values for:

- signal-check horizon;
- post-processor construction;
- environment `decision_every`;
- training horizon;
- manifest metadata.

The synthetic sample duration remains separately bounded for runtime control. A new P0-specific days option may be introduced only if it is explicit and recorded. Hidden mutation of candidate timing parameters is forbidden.

The P0 report must include the effective configuration:

```json
{
  "config": {
    "horizon": 12,
    "decision_every": 4,
    "days": 240
  }
}
```

Release eligibility continues to depend on `P0_PASSED`.

## 2. Content-addressed feature snapshots

The snapshot ID must be the SHA-256 of a canonical byte stream containing all inference-relevant snapshot content:

- schema/version marker;
- active bundle digest;
- base timeframe;
- ordered symbols;
- ordered feature names;
- ordered global feature names;
- selected timestamps;
- `feature_history` values, shape, and dtype normalized to little-endian float64;
- `global_features` values, shape, and dtype normalized to little-endian float64;
- `close_history` values, shape, and dtype normalized to little-endian float64.

The hash must not depend on Python object identity, CSV filename order, platform-native endianness, or non-canonical JSON formatting.

Changing any inference-relevant value while keeping timestamp and row count unchanged must change `snapshot_id`.

A small pure helper will build the canonical hash so it can be unit-tested independently.

## 3. Timeframe-aware completed-bar freshness

The provider must treat feature timestamps as bar-open timestamps.

For base timeframe `T`:

- `bar_close = bar_open + duration(T)`;
- only rows with `bar_close <= now` are completed;
- the most recent completed row is the inference endpoint;
- `data_age_hours = now - latest_completed_bar_close`.

The provider must reject snapshots when no completed bar exists.

Supported durations remain `15m`, `1h`, `4h`, and `1d`. The duration mapping must be derived from the existing timeframe utilities rather than duplicated with unrelated constants.

The provider will accept an injectable UTC clock for deterministic tests. Runtime behavior uses the current UTC time by default.

This design avoids falsely marking a currently forming 4h or 1d bar as many hours stale while also preventing incomplete-bar inference.

The existing risk bundle remains the authority for the maximum permitted `data_age_hours`; no new global stale threshold is introduced.

## 4. Legacy metrics server isolation

`mars_lite.server.signal_server` remains the only authoritative Serving Plane.

The legacy `metrics_server.py` remains available for local training-dashboard compatibility but will be explicitly marked and guarded as development-only:

- module documentation states that it is not a Production or signal-serving entrypoint;
- startup requires an explicit development opt-in, such as `TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`, or a direct `development_only=True` factory argument used by tests;
- without opt-in, app creation or server startup fails closed with a clear message;
- docs must not present its `/api/signal/latest` route as supported Serving behavior;
- the legacy server is not imported by `scripts/run_server.py`.

This increment does not rewrite or secure every legacy dashboard endpoint. It prevents accidental use as the current Serving Plane.

## 5. Offline local GameDay

Add `scripts/run_local_gameday.py` as a deterministic, exchange-free validation harness.

It will create temporary local resources and exercise the real Registry, bundle loader, runtime, feature provider, signal server, and audit store where practical.

Required scenarios:

1. **Healthy activation**
   - register and activate a valid local bundle;
   - load it into Serving;
   - verify readiness identity.

2. **Content mutation identity**
   - change an inference-relevant market value without changing row count or final timestamp;
   - verify the snapshot ID changes.

3. **Timeframe freshness**
   - verify completed-bar selection and age for at least `1h`, `4h`, and `1d`;
   - verify an incomplete latest bar is excluded.

4. **Stale-data fail closed**
   - provide a snapshot older than the bundle guardrail limit;
   - verify no actionable target weights are returned.

5. **Replay rejection**
   - submit the same request ID twice;
   - verify the replay is rejected and audited.

6. **Bundle rejection and healthy-runtime preservation**
   - activate a corrupted or Git-SHA-mismatched bundle;
   - verify refresh fails and the previous healthy in-memory bundle remains served as degraded.

7. **Rollback**
   - activate a second valid version;
   - roll back;
   - verify Registry and Serving identities return to the expected version and digest.

The script prints a JSON summary and exits non-zero if any scenario fails. It does not claim to test real exchange cancellation, fills, reconciliation, network partitions, or multi-node behavior.

## 6. Registry decision

No Registry implementation change is required for local validation.

The existing filesystem Registry remains supported only for one administrative filesystem domain with one authoritative active pointer. Documentation will explicitly state that local success does not establish multi-node safety.

## Error handling

All new paths fail closed:

- invalid P0 config causes a validation error before training;
- no completed market bar causes snapshot creation to fail;
- non-finite snapshot values fail validation before hashing/inference;
- legacy metrics server without explicit development opt-in refuses to start;
- any failed GameDay scenario produces a non-zero exit code and identifies the failed invariant.

## Testing

Required tests:

- Production Pipeline passes candidate `horizon` and `decision_every` unchanged into P0.
- P0 report records effective timing configuration.
- Snapshot hashes are deterministic across equivalent array representations.
- A single value mutation changes the snapshot hash.
- Incomplete bars are excluded for `1h`, `4h`, and `1d`.
- Staleness is measured from bar close, not bar open.
- No-completed-bar input is rejected.
- Legacy metrics server requires explicit development opt-in.
- `scripts/run_server.py` continues to use only the authoritative signal server.
- Local GameDay success and deliberately failed invariant paths are covered.
- Ruff lint, Ruff format, mypy, focused tests, and full pytest/coverage pass before merge.

## Acceptance criteria

1. A candidate configured with `horizon=12` and `decision_every=4` runs P0 with exactly those values.
2. Snapshot identity changes when inference values change, even if timestamp and row count do not.
3. A completed 1d bar is not falsely treated as 24 hours stale merely because its timestamp is the bar open.
4. An incomplete current bar is never used for inference.
5. The legacy metrics server cannot start accidentally without explicit development opt-in.
6. The offline GameDay validates the seven required local scenarios and returns machine-readable results.
7. No distributed Registry or real exchange adapter is introduced.
8. Production documentation remains **NO-GO**.
