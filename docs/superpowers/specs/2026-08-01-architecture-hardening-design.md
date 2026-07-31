# Architecture Hardening Design

## Scope

This change fixes the architecture findings identified on `main`. It excludes
files changed by the active Studio chart branch.

The maintained 17-layer architecture remains the source of truth. This change
does not reorganize the package tree, add exchange execution, change the
research gate, or modify the Stage A producer implementation.

## Goals

1. Make GPU smoke evidence schema validation use one authoritative contract.
2. Remove the lower-layer `trade_rl.rl` dependency on SB3 integration modules.
3. Make learned-policy deployment semantics independent from residual versus
   target-weight action semantics.
4. Move release-publication orchestration out of the serving package.
5. Treat GPU smoke execution as maintained operations code rather than example
   implementation code.
6. Remove duplicated GPU workflow validation logic.

## Non-goals

- No backward migration of historical serving bundle bytes.
- No modification of active PR #318 Studio files.
- No direct merge to `main`.
- No broad folder reorganization outside the files required by these findings.

## Design

### Authoritative GPU smoke evidence contract

Create `trade_rl/operations/gpu_training_smoke.py` as the maintained entry point
for the CUDA BC-to-PPO resume smoke. The module owns
`GPU_TRAINING_SMOKE_SCHEMA` and the evidence validator. The existing example
script becomes a thin wrapper that imports and invokes the maintained module.

Both manual and main-push GPU workflows invoke the same Python validator rather
than embedding duplicated schema and field assertions in YAML. Workflow tests
assert that both workflows call the validator and do not pin schema literals.

### Layer-safe public exports

`trade_rl.rl` exports only symbols implemented in `trade_rl.rl` or lower layers.
SB3 backends remain public through `trade_rl.integrations`. A focused
architecture test inspects lazy export maps and rejects any export target above
the package layer.

### Deployment kind and action semantics

The existing `PolicyMode` continues to describe whether selection chose a
baseline or a learned policy. Serving bundle manifests gain an explicit
`action_mode` field derived from the authoritative action specification.

A learned target-weight policy may therefore keep the existing learned-policy
selection mode while being identified as `target_weight`. Runtime snapshots and
Studio-facing metadata expose both fields. Error messages use the neutral term
`policy action` instead of assuming residual semantics.

The serving bundle schema is incremented because the digest payload changes.
Loaders fail closed on older schema bytes; historical artifacts remain
inspectable through existing migration tooling but are not silently accepted by
the maintained runtime.

### Release publication orchestration boundary

Move `package_selected_training_run` to
`trade_rl.workflows.release_packaging`. This workflow owns evidence-chain
coordination, staging, copying, and atomic publication. `trade_rl.serving`
retains bundle contracts, bundle construction, loading, registry, and runtime
activation only.

A compatibility import is not added because it would preserve the wrong
dependency direction. CLI and tests import the workflow entry point.

### Workflow consolidation

Introduce a reusable workflow for the common self-hosted GPU validation steps.
The manual workflow and main-push workflow pass only trigger-specific inputs.
The old PR-numbered workflow filename is retired after its replacement is in
place.

The reusable workflow uses immutable action SHAs, read-only permissions,
owner/main guards, the protected GPU environment, exact checkout verification,
and the common Python evidence validator.

## Error handling

- Unknown or mismatched GPU evidence schemas fail closed in Python.
- Lazy exports targeting an upper layer fail architecture tests.
- Serving manifests without explicit action semantics fail construction.
- Runtime action validation rejects incorrect shape, non-finite values, and
  out-of-range values without referring to residual semantics.
- Release publication removes staging output after any failure and never
  replaces an existing bundle directory.
- GPU workflows upload diagnostics on failure.

## Testing

Focused tests cover GPU schema validation, workflow wiring, layer-safe lazy
exports, residual and target-weight serving identities, runtime snapshot
propagation, release packaging ownership, and example wrapper thinness.

Final verification is performed on one unchanged branch head with Ruff, Ruff
format, MyPy, Import Linter, Vulture, full pytest with branch coverage, critical
coverage, Studio tests/typecheck/build/layout, Ubuntu and Windows compatibility,
training image, and PostgreSQL Catalog checks. GPU execution remains a separate
protected-runner gate.
