# Local Validation Hardening Design

Status: approved direction from the architecture follow-up.

## Goal

Make local end-to-end validation trustworthy without expanding the project into a distributed production platform.

This increment must ensure that:

- P0 validates the actual candidate timing configuration rather than a hidden fixed configuration;
- a feature snapshot identity changes whenever inference-relevant content changes;
- staleness is computed from the close time of the latest completed bar for the bundled base timeframe;
- the legacy metrics server cannot be mistaken for the authoritative Serving Plane;
- an offline local GameDay demonstrates critical fail-closed behavior without exchange access;
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

### Approach A: documentation only

Explain the findings without changing code. This is insufficient because P0 and staleness currently produce misleading local results, and the current snapshot identity cannot prove which values were used for inference.

### Approach B: minimal local correctness hardening — selected

Fix the candidate/P0 mismatch, content-address the in-memory snapshot, make freshness timeframe-aware, isolate the legacy server, and add an offline GameDay harness. Keep the single-node Registry unchanged.

### Approach C: full production infrastructure expansion

Replace the Registry with a networked service, add testnet execution, deploy multiple Serving instances, and automate operational GameDays. This is premature and mixes local correctness with infrastructure and owner decisions.

## 1. Candidate-aligned P0

`production_pipeline.run()` must stop overwriting `args.horizon` and `args.decision_every` with `4` and `1`.

P0 uses the resolved candidate values for signal checking, post-processing, environment construction, training, and report metadata.

Synthetic P0 duration is controlled separately by a new explicit CLI option:

```text
--p0-days 240
```

`--p0-days` must be a positive integer, defaults to `240`, and does not mutate the candidate dataset `--days` value. Hidden mutation of candidate timing parameters is forbidden.

The P0 report records:

```json
{
  "config": {
    "horizon": 12,
    "decision_every": 4,
    "p0_days": 240
  }
}
```

Release eligibility continues to depend on `P0_PASSED`.

## 2. Content-addressed feature snapshots

The snapshot ID is the SHA-256 of a canonical byte stream containing all inference-relevant content:

- schema/version marker;
- active bundle digest;
- base timeframe;
- ordered symbols;
- ordered feature names;
- ordered global feature names;
- selected timestamps normalized to nanosecond integers;
- `feature_history`, `global_features`, and `close_history`, including shapes and values normalized to contiguous little-endian float64.

The hash must not depend on Python object identity, platform endianness, CSV filename order, or non-canonical JSON formatting.

Changing any inference-relevant value while keeping timestamp and row count unchanged must change `snapshot_id`. A pure helper builds the canonical hash and is tested independently.

## 3. Timeframe-aware completed-bar freshness

Feature timestamps are bar-open timestamps.

For base timeframe `T`:

- `bar_close = bar_open + duration(T)`;
- only rows with `bar_close <= now` are completed;
- the most recent completed row is the inference endpoint;
- `data_age_hours = now - latest_completed_bar_close`.

The provider rejects snapshots when no completed bar exists.

Supported durations remain `15m`, `1h`, `4h`, and `1d`. Duration is derived from the existing `TF_TO_MINUTES` utility. The provider accepts an injectable UTC clock for deterministic tests and uses current UTC by default.

The existing bundle guardrail remains the authority for maximum permitted data age. No new global stale threshold is introduced.

## 4. Legacy metrics server isolation

`mars_lite.server.signal_server` remains the only authoritative Serving Plane.

The legacy server remains available only for local training-dashboard compatibility:

- module and API documentation label it development-only and non-authoritative;
- `create_app(..., development_only=False)`, `run_server(..., development_only=False)`, and `run_server_async(..., development_only=False)` fail closed unless `development_only=True`;
- the module CLI requires an explicit `--development-only` switch;
- existing training-dashboard callers must pass that explicit switch or argument;
- documentation must not present the legacy `/api/signal/latest` route as supported Serving behavior;
- `scripts/run_server.py` continues to import only the authoritative signal server.

This increment does not secure or redesign every legacy dashboard endpoint. It prevents accidental use as the current Serving Plane.

## 5. Offline local GameDay

Add `scripts/run_local_gameday.py` as a deterministic, exchange-free validation harness. It creates temporary resources and exercises the real Registry, bundle loader, runtime, feature provider, signal server, and audit store where practical.

Required scenarios:

1. Healthy valid-bundle registration, activation, runtime load, and readiness identity.
2. Market-value mutation changes snapshot identity without changing row count or final timestamp.
3. Completed-bar selection and age for `1h`, `4h`, and `1d`, including exclusion of an incomplete latest bar.
4. Stale data yields no actionable target weights.
5. Reusing a request ID is rejected and audited.
6. Corrupted or Git-SHA-mismatched activation is rejected while the previous healthy runtime remains available as degraded.
7. Activation of a second valid version followed by rollback restores the expected Registry and Serving version/digest.

The script prints one JSON document and exits non-zero if any scenario fails. It does not claim to test real exchange cancellation, fills, reconciliation, network partitions, or multi-node behavior.

## 6. Registry decision

No Registry implementation change is required for local validation.

The filesystem Registry is supported only for one administrative filesystem domain with one authoritative active pointer. Documentation explicitly states that local success does not establish multi-node safety.

## Error handling

All new paths fail closed:

- invalid P0 configuration fails before training;
- no completed market bar fails snapshot creation;
- non-finite snapshot content fails validation before hashing or inference;
- the legacy metrics server refuses startup without explicit development opt-in;
- any failed GameDay invariant produces a non-zero exit code and identifies the failed scenario.

## Testing

Required tests:

- Production Pipeline passes candidate `horizon` and `decision_every` unchanged into P0.
- `--p0-days` is validated and the P0 report records effective configuration.
- Snapshot hashes are deterministic across equivalent array representations.
- A single content mutation changes the snapshot hash.
- Incomplete bars are excluded for `1h`, `4h`, and `1d`.
- Staleness is measured from bar close, not bar open.
- No-completed-bar input is rejected.
- Legacy metrics server requires explicit development opt-in.
- `scripts/run_server.py` remains independent from the legacy server.
- Local GameDay success and deliberately failed invariant paths are covered.
- Ruff lint, Ruff format, mypy, focused tests, and full pytest/coverage pass before merge.

## Acceptance criteria

1. A candidate configured with `horizon=12` and `decision_every=4` runs P0 with exactly those values.
2. Snapshot identity changes when inference values change even if timestamp and row count do not.
3. A completed 1d bar is not falsely treated as 24 hours stale because its timestamp is the bar open.
4. An incomplete current bar is never used for inference.
5. The legacy metrics server cannot start without explicit development opt-in.
6. The offline GameDay validates all seven scenarios and returns machine-readable results.
7. No distributed Registry or real exchange adapter is introduced.
8. Production documentation remains **NO-GO**.
