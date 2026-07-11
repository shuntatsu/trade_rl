# Operations

Production remains **NO-GO** until [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) is complete.

## Control Plane run

```bash
uv run python scripts/run_pipeline.py \
  --source postgres \
  --git-sha "$(git rev-parse HEAD)" \
  --model-version model-YYYYMMDD-N
```

The pipeline must stop on failed P0, walk-forward, final baseline, or configured significance gates unless an explicitly documented research-only `--force` run is being performed. A forced run must not be promoted.

The output candidate is placed under `output/.../candidates/<version>` and registered immutably. Training never activates it.

## Evidence and deployment

Canary and Production deployment require a successful prior GitHub Actions run containing one `deployment-evidence` artifact with:

- `candidate.json`;
- the model/report files required by the deployment gate;
- `serving_candidate/` containing the exact immutable ServingBundle;
- Shadow, drift, incident, and, for Production, Canary reports.

The deployment workflow verifies source-run success, Git SHA, model version, report hashes, bundle digest, evidence lineage, incidents, approval ticket, and Environment approval. It then registers and atomically activates the exact `serving_candidate/` in the stage Registry configured by `TRADE_RL_REGISTRY_DIR`.

## Start serving

Required secrets and configuration:

```text
TRADE_RL_SERVING_TOKEN
TRADE_RL_REGISTRY_DIR
TRADE_RL_AUDIT_DB
TRADE_RL_DATA_DIR
TRADE_RL_ALLOWED_ORIGINS
```

Start:

```bash
uv run python scripts/run_server.py
```

Check:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/ready
```

`/ready` may be `degraded` while the last healthy in-memory bundle continues serving. `unavailable` means no actionable signal can be returned.

## Rollback

Inspect the Registry:

```bash
uv run python scripts/manage_registry.py \
  --registry-dir "$TRADE_RL_REGISTRY_DIR" list
uv run python scripts/manage_registry.py \
  --registry-dir "$TRADE_RL_REGISTRY_DIR" show-active
```

Rollback:

```bash
uv run python scripts/manage_registry.py \
  --registry-dir "$TRADE_RL_REGISTRY_DIR" rollback \
  --target-version <known-good-version>
```

Confirm that `/ready` reports the expected version and digest. The Serving Plane uses its normal validated hot-swap path; operators do not copy or rename model files.

## Incident response

1. Block new live risk at the Trade Platform.
2. Preserve request IDs, bundle digest, active version, market snapshot IDs, Registry state, and audit database.
3. If exposure exists, invoke the real platform-specific emergency adapter with a unique idempotency key.
4. Require cancellation, reconciliation, reduce-only closure, and verified zero residual exposure before reporting flatten success.
5. Roll back only to a registered, digest-valid known-good bundle.
6. Keep Production disabled until the root cause, evidence, and recovery validation are documented.

Repository code intentionally contains no invented exchange adapter or operational contact destination.

## GameDay minimum

Before GO approval, run a testnet exercise that proves:

- stale data returns no actionable signal;
- invalid or replayed requests fail closed;
- a corrupted candidate does not replace the healthy bundle;
- activation and rollback change the served version/digest as expected;
- emergency cancellation/flatten is idempotent and reconciled;
- audit records are sufficient to reconstruct the event.

Attach commands, timestamps, identities, outputs, and reviewer approval to the readiness checklist.
