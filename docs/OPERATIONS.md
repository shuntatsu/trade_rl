# Operations

Production remains **NO-GO** until [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) is complete.

## Control Plane run

Copy and edit the release risk example so that `symbol_liquidity_caps` exactly matches the resolved bundle symbols:

```bash
cp config/release-risk.example.json config/release-risk.local.json
```

Run an eligible release pipeline:

```bash
uv run python scripts/run_pipeline.py \
  --source postgres \
  --git-sha "$(git rev-parse HEAD)" \
  --model-version model-YYYYMMDD-N \
  --risk-config config/release-risk.local.json
```

A release-capable run must have a non-empty sealed holdout, pass P0, walk-forward, Gate 2, and any configured significance gate, and load a complete release risk policy.

`--force`, `--skip-p0`, `--skip-wf`, or `--skip-gate` makes the run research-only. The run may continue and write reports, but code prevents candidate construction and Registry registration. Use `--no-register` for intentional research runs. `--skip-pbt` is recorded but is not by itself release-disqualifying.

An eligible output candidate is placed under `output/.../candidates/<version>` and registered immutably. Training never activates it.

## Evidence and deployment

Canary and Production deployment require a successful prior GitHub Actions run containing one `deployment-evidence` artifact with:

- `candidate.json`;
- the model/report files required by the deployment gate;
- `serving_candidate/` containing the exact immutable ServingBundle;
- Shadow, drift, incident, and, for Production, Canary reports.

Configure each Canary or Production GitHub Environment with:

```text
TRADE_RL_REGISTRY_DIR       absolute persistent path shared with stage Serving
TRADE_RL_SERVING_READY_URL  full stage readiness URL, including /ready
```

The deployment job requires a self-hosted runner with the `trade-rl-deploy` label and access to the same persistent Registry storage used by the stage Serving process. Do not use an ephemeral GitHub-hosted runner as the deployment target.

The deployment workflow verifies source-run success, Git SHA, model version, report hashes, bundle digest, release eligibility, risk policy, evidence lineage, incidents, approval ticket, and Environment approval. It then registers and atomically activates the exact `serving_candidate/`.

Activation is not deployment success. The workflow polls `TRADE_RL_SERVING_READY_URL` and succeeds only when `/ready` reports the approved model version, bundle digest, and release Git SHA. `degraded` is accepted only when it reports the newly approved identity. A previous identity or unreachable endpoint fails the workflow.

## Start serving

Required secrets and configuration:

```text
TRADE_RL_SERVING_TOKEN
TRADE_RL_RELEASE_GIT_SHA
TRADE_RL_REGISTRY_DIR
TRADE_RL_AUDIT_DB
TRADE_RL_DATA_DIR
TRADE_RL_ALLOWED_ORIGINS
```

`TRADE_RL_RELEASE_GIT_SHA` must be the exact 40-character Git SHA of the running Serving release. Strict Production serving rejects a bundle built from another revision.

Start:

```bash
export TRADE_RL_RELEASE_GIT_SHA="$(git rev-parse HEAD)"
uv run python scripts/run_server.py
```

Check:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/ready
```

`/ready` reports `status`, `active_version`, `bundle_digest`, `release_git_sha`, and an optional failure reason. It may be `degraded` while the last healthy in-memory bundle continues serving. `unavailable` means no actionable signal can be returned.

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

Confirm that `/ready` reports the expected version, bundle digest, and release Git SHA. The rollback target must have been built from the currently running release SHA; otherwise strict binding rejects it and preserves the prior in-memory bundle. Operators do not copy or rename model files.

Automatic rollback after a failed post-activation handshake is intentionally disabled. A mismatch is an incident requiring an explicit operator decision based on Registry state, Serving readiness, and audit evidence.

## Incident response

1. Block new live risk at the Trade Platform.
2. Preserve request IDs, bundle digest, active version, running release SHA, market snapshot IDs, Registry state, and audit database.
3. If exposure exists, invoke the real platform-specific emergency adapter with a unique idempotency key.
4. Require cancellation, reconciliation, reduce-only closure, and verified zero residual exposure before reporting flatten success.
5. Roll back only to a registered, digest-valid, code-compatible known-good bundle.
6. Keep Production disabled until the root cause, evidence, and recovery validation are documented.

Repository code intentionally contains no invented exchange adapter or operational contact destination.

## GameDay minimum

Before GO approval, run a testnet exercise that proves:

- stale data returns no actionable signal;
- invalid or replayed requests fail closed;
- a corrupted or Git-SHA-mismatched candidate does not replace the healthy bundle;
- activation and rollback change the served version/digest as expected;
- deployment fails when the live served identity does not match the approved identity;
- emergency cancellation/flatten is idempotent and reconciled;
- audit records are sufficient to reconstruct the event.

Attach commands, timestamps, identities, outputs, and reviewer approval to the readiness checklist.
