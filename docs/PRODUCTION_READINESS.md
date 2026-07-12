# Production Readiness

Current decision: **NO-GO**.

A checked box requires attached evidence, not an assertion. Code owners may check repository-verifiable items; operational, legal, security, and exchange items require the responsible owner.

## Code and CI

- [ ] Exact release head passes Ruff lint.
- [ ] Exact release head passes Ruff format check.
- [ ] Exact release head passes mypy.
- [ ] Exact release head passes the complete pytest suite.
- [ ] Coverage is at least 70%.
- [ ] Critical and Important architecture-review findings are closed or explicitly accepted by the accountable owner.

## Control Plane release eligibility

- [ ] Approved run used no `--force`, `--skip-p0`, `--skip-wf`, or `--skip-gate` override.
- [ ] A non-empty sealed holdout remained outside PBT, walk-forward selection, feature selection, and final training.
- [ ] Gate 2 evaluated the final model on that sealed holdout.
- [ ] P0, walk-forward, Gate 2, and any required significance gate passed.
- [ ] The bundle `release_eligibility` record matches the approved run evidence.
- [ ] Research-only execution is demonstrated to produce no registrable candidate.

## Artifact and model identity

- [ ] One complete ServingBundle is produced by the approved Control Plane run.
- [ ] Bundle model version, Git SHA, file digests, and canonical digest are recorded.
- [ ] Bundle validation confirms eligible release metadata and complete risk policy.
- [ ] Shadow and Canary evidence reference the exact same bundle identity.
- [ ] Deployment gate source run and release-branch restrictions are configured.
- [ ] Running Serving release Git SHA equals the bundle Git SHA.
- [ ] Activated Registry identity equals the version and digest returned by Serving.
- [ ] Post-activation workflow verifies version, digest, and release Git SHA through the live `/ready` endpoint.
- [ ] Rollback to a code-compatible known-good registered version is demonstrated.

## Serving Plane

- [ ] Serving exposes only health, readiness, and authenticated signal routes.
- [ ] Machine-to-machine token and rotation procedure are configured.
- [ ] Network bind, proxy, TLS, and origin allowlist are configured.
- [ ] `TRADE_RL_RELEASE_GIT_SHA` is injected from the immutable deployed release.
- [ ] Current positions and account state are supplied on every request.
- [ ] Request IDs and market snapshot identities are unique and audited.
- [ ] Corrupted or Git-SHA-mismatched activation preserves the prior healthy bundle.
- [ ] No healthy bundle returns `503` with no actionable weights.

## Risk and execution

- [ ] Release risk policy contains finite limits for leverage, single-symbol weight, net exposure, worst-case notional, and minimum order notional.
- [ ] Liquidity caps exactly cover the ordered bundle symbols and forbidden symbols are explicitly recorded.
- [ ] Trade Platform enforces the returned pre-trade risk verdict before order placement.
- [ ] Pending orders, symbol restrictions, liquidity, reduce-only, and exposure limits are covered end to end.
- [ ] A real exchange/platform `EmergencyExecutionAdapter` is implemented and reviewed.
- [ ] Emergency cancellation, reconciliation, reduce-only closure, and residual-position checks pass on testnet.
- [ ] Idempotency keys prevent duplicate emergency execution.

## Deployment governance

- [ ] `shadow`, `canary`, and `production` GitHub Environments exist.
- [ ] Production has required independent reviewers.
- [ ] A self-hosted runner with the `trade-rl-deploy` label is isolated and administered.
- [ ] `TRADE_RL_REGISTRY_DIR` is an absolute stage-appropriate persistent path shared with Serving.
- [ ] `TRADE_RL_SERVING_READY_URL` points to the correct stage `/ready` endpoint.
- [ ] Evidence producer identity is trusted and branch restricted.
- [ ] A deliberately mismatched live identity makes deployment fail.
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
