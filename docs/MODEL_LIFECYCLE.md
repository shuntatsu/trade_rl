# Model Lifecycle

## 1. Train and evaluate

The Control Plane builds one feature set, separates development and sealed holdout data when possible, and runs P0, PBT, walk-forward/cost sensitivity, final training, baseline comparison, and optional statistical significance checks.

Failed gates do not produce a promotable model. `--force` is research-only and does not authorize deployment.

## 2. Construct a candidate

A passing run creates one complete `ServingBundle` containing the model, ordered feature schemas, preprocessing, observation contract, post-processing, risk configuration, evaluation identity, Git SHA, and digests.

Production-compatible models use observation progress mode `zero`; episode-relative progress cannot be reconstructed online and is rejected during bundle validation.

## 3. Register

```bash
uv run python scripts/manage_registry.py \
  --registry-dir <registry> register <candidate-directory>
```

Registration:

1. validates the source bundle;
2. copies it to a temporary sibling directory;
3. validates the copy;
4. atomically renames it to `versions/<version>`;
5. leaves `active.json` unchanged.

Re-registering the same version and digest is idempotent through the management CLI. Reusing a version for different content is rejected.

## 4. Produce evidence

Shadow and Canary evaluations must reference the exact model version, Git SHA, bundle identity, source run, and parent evidence where applicable. Production evidence adds Canary results, incident state, approval ticket, and Environment approval.

## 5. Activate

Activation occurs only in the deployment Control Plane after gate success:

```bash
uv run python scripts/manage_registry.py \
  --registry-dir <registry> activate <version> \
  --evidence-identity <trusted-identity>
```

Activation validates the registered bundle and atomically replaces `active.json`. It never copies or renames model files.

## 6. Serve

`ServingRuntime` reads the active identity, validates and loads the bundle beside the currently loaded version, runs readiness checks, and swaps the in-memory reference only after success.

Responses include the served model version and bundle digest so the caller can verify identity.

## 7. Roll back

Rollback selects a registered known-good version and atomically changes `active.json`. Serving returns to that version through the same validated hot-swap path.

## Registry invariants

- one registry implementation;
- one active pointer;
- immutable version directories;
- digest validation at every boundary;
- registration and activation are separate operations;
- failed activation leaves the prior pointer intact;
- served identity must equal active identity;
- no model lifecycle mutation is exposed by the Serving Plane.
