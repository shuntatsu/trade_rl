# Production Readiness

Current decision: **NO-GO**.

A checked box requires attached evidence, not an assertion. Code owners may check repository-verifiable items; operational, legal, security, and exchange items require the responsible owner.

## Code and CI

- [ ] Exact release head passes Ruff lint.
- [ ] Exact release head passes Ruff format check.
- [ ] Exact release head passes mypy.
- [ ] Exact release head passes the complete pytest suite.
- [ ] Coverage is at least 70%.
- [ ] Critical and Important architecture-review findings are closed.

## Artifact and model identity

- [ ] One complete ServingBundle is produced by the approved Control Plane run.
- [ ] Bundle model version, Git SHA, file digests, and canonical digest are recorded.
- [ ] Shadow and Canary evidence reference the exact same bundle identity.
- [ ] Deployment gate source run and release-branch restrictions are configured.
- [ ] Activated Registry identity equals the version and digest returned by Serving.
- [ ] Rollback to a known-good registered version is demonstrated.

## Serving Plane

- [ ] Serving exposes only health, readiness, and authenticated signal routes.
- [ ] Machine-to-machine token and rotation procedure are configured.
- [ ] Network bind, proxy, TLS, and origin allowlist are configured.
- [ ] Current positions and account state are supplied on every request.
- [ ] Request IDs and market snapshot identities are unique and audited.
- [ ] Corrupted activation preserves the prior healthy bundle.
- [ ] No healthy bundle returns `503` with no actionable weights.

## Risk and execution

- [ ] Trade Platform enforces the returned pre-trade risk verdict before order placement.
- [ ] Pending orders, symbol restrictions, liquidity, reduce-only, and exposure limits are covered end to end.
- [ ] A real exchange/platform `EmergencyExecutionAdapter` is implemented and reviewed.
- [ ] Emergency cancellation, reconciliation, reduce-only closure, and residual-position checks pass on testnet.
- [ ] Idempotency keys prevent duplicate emergency execution.

## Deployment governance

- [ ] `shadow`, `canary`, and `production` GitHub Environments exist.
- [ ] Production has required independent reviewers.
- [ ] `TRADE_RL_REGISTRY_DIR` points to stage-appropriate persistent storage.
- [ ] Evidence producer identity is trusted and branch restricted.
- [ ] Secrets are stored outside the repository and rotation is tested.

## Operations and compliance

- [ ] On-call, risk, security, and compliance contacts are real and tested.
- [ ] Incident severity, escalation, and communication procedures are approved.
- [ ] Applicable jurisdictions and retention periods are determined by qualified owners.
- [ ] Audit storage, access, backup, and deletion policies are approved.
- [ ] Testnet GameDay evidence is attached.
- [ ] Final operational owner signs GO approval.

## GO rule

Production may be marked GO only when every applicable item is checked with evidence and no unresolved Critical or Important finding remains. Until then, all documentation and interfaces must continue to report **NO-GO**.
