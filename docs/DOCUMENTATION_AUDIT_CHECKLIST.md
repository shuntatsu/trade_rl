# Documentation audit checklist

- Audit date: 2026-07-11
- Scope: clean M7 rebuild branch, deployment workflow, production-safety modules, tests, and operational documents
- Production status: **NO-GO** until every Production blocker is closed with recorded evidence

Status values: `PASS`, `FIXED`, `OWNER ACTION`, `BLOCKED`, `HISTORICAL`.

## A. Code and workflow consistency

- [x] **FIXED** Deployment promotion no longer accepts self-reported Shadow, Canary, or drift booleans.
- [x] **FIXED** Canary and Production require a downloaded, content-addressed evidence bundle from a successful source run.
- [x] **FIXED** Model and report SHA-256 values are recalculated from files; model version, Git commit, artifact digest, and Shadow-to-Canary lineage are cross-checked.
- [x] **FIXED** Gate thresholds are code-owned; evidence JSON cannot weaken them. Non-finite and out-of-range metrics are rejected.
- [x] **FIXED** Active incident evidence blocks promotion.
- [x] **FIXED** Emergency flatten fails closed without an execution adapter and idempotency key.
- [x] **FIXED** Flatten success requires new-risk blocking, order cancellation, reduce-only close orders, reconciliation, zero open orders, and residual positions within tolerance.
- [x] **FIXED** Worst-case notional models current positions, proposed deltas, and one-sided pending-order fills.
- [x] **FIXED** Minimum order and liquidity checks use execution deltas rather than target holdings.
- [x] **FIXED** Environment and baseline paths pass symbols/current weights and validate the actual post-threshold execution target.
- [x] **FIXED** Replay uses a shared liquidity ledger, actual fill timestamps, and a fixed-frequency mark-to-market equity grid.
- [x] **FIXED** Legacy replay tests now assert the fixed-frequency timeline instead of the obsolete two-point no-order curve.
- [ ] **OWNER ACTION — Production blocker** Implement and configure a real exchange/platform `EmergencyExecutionAdapter`; record the reviewed `module:factory` path.
- [ ] **OWNER ACTION — Production blocker** Ensure live callers pass all currently open orders into `PreTradeRiskVerifier`.

## B. Test and CI integrity

- [x] **PASS** Focused M7 regression tests, Ruff checks, formatting checks, and `mypy mars_lite` passed before materialization was committed.
- [x] **FIXED** Obsolete expectations were updated rather than deleting adversarial coverage.
- [x] **FIXED** No one-shot workflow, patch script, or repository-wide formatting-only changes remain in the clean branch.
- [ ] **BLOCKED until PR CI completes** Full repository pytest suite and 70% coverage gate for the exact final commit.

## C. Deployment documentation and trust boundary

- [x] **FIXED** `deployment_evidence.md` defines required files, identity fields, digest verification, stage ordering, and source-run SHA checks.
- [x] **FIXED** Production requires a `PROD-<digits>` ticket and GitHub `production` Environment approval.
- [ ] **OWNER ACTION — Production blocker** Configure `shadow`, `canary`, and `production` GitHub Environments; require designated reviewers for `production`.
- [ ] **OWNER ACTION — Production blocker** Create and restrict the trusted producer workflow that generates `deployment-evidence`.
- [ ] **OWNER ACTION** Define artifact retention and an external deletion-protected archive.

## D. Incident response and rollback

- [x] **FIXED** The incident runbook no longer equates emitting zero weights with executing a flatten.
- [x] **FIXED** Flatten instructions require adapter, idempotency key, and reconciliation evidence.
- [x] **FIXED** Rollback distinguishes registry state from the model actually loaded by the serving process.
- [ ] **OWNER ACTION — Production blocker** Replace placeholder on-call, risk, and compliance contacts with real destinations.
- [ ] **OWNER ACTION — Production blocker** Define exchange-specific residual-position tolerances and retry/escalation limits.
- [ ] **OWNER ACTION — Production blocker** Run testnet GameDays for disconnects, partial fills, cancellation races, crashes, tampering, rollback, and emergency flatten.

## E. Compliance and governance

- [x] **FIXED** Retention language is an internal policy pending legal determination, not a universal legal claim.
- [x] **FIXED** `model_decision_log.md` records reproducibility hashes, evidence IDs, owners, approvals, limitations, and rollback conditions.
- [ ] **OWNER ACTION — Legal/Compliance blocker** Determine operating entity, customer/fund status, jurisdictions, exchanges, instruments, and applicable retention rules.
- [ ] **OWNER ACTION** Define immutable storage, access control, approval authority, and incident override authority.

## F. Research claims

- [x] **HISTORICAL** `docs/README.md` classifies `ARCHITECTURE.md` and `PROFIT_DESIGN.md` as research records that cannot override current operational documents.
- [x] **FIXED** Return, Sharpe, alpha, and edge statements are treated as historical observations or hypotheses, not proof of future profitability.
- [ ] **BLOCKED — Economic validation** Complete untouched real-data validation followed by recorded Shadow and Canary evidence.

## G. Required evidence before Production GO

- [ ] Full CI result for the exact candidate Git commit.
- [ ] Candidate model, configuration, data-manifest, and dependency-lock SHA-256 values.
- [ ] Verified Shadow, drift, incident, and Canary reports with linked run IDs.
- [ ] Replay-versus-live execution calibration report.
- [ ] Testnet GameDay and emergency-flatten evidence.
- [ ] Completed model decision-log entry.
- [ ] Approved Production ticket and GitHub Environment approver.
- [ ] Reviewed real execution-adapter configuration.
