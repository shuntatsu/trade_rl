# Documentation audit checklist

Status values: `PASS`, `FIXED`, `OWNER ACTION`, `BLOCKED`, `HISTORICAL`.

## A. Code and workflow consistency

- [x] **FIXED** Deployment workflow no longer accepts `shadow_passed`, `canary_passed`, or `drift_report_passed` booleans.
- [x] **FIXED** Canary and Production require a downloaded GitHub Actions evidence artifact.
- [x] **FIXED** Model artifact and evidence report SHA-256 values are recomputed from files.
- [x] **FIXED** Model version, Git commit, model digest, and Shadow→Canary run lineage are cross-checked.
- [x] **FIXED** Active incident evidence blocks promotion.
- [x] **FIXED** Emergency flatten fails closed when no execution adapter or idempotency key is supplied.
- [x] **FIXED** Emergency flatten success requires: new-risk block, open-order cancellation, reduce-only close orders, reconciliation, no open orders, and residual positions within tolerance.
- [x] **FIXED** Worst-case notional supports one-sided fill scenarios using current positions, pending buy/sell orders, and the proposed delta.
- [x] **FIXED** Minimum order notional and liquidity capacity are evaluated against execution deltas rather than target holdings.
- [x] **FIXED** Replay simulation uses shared liquidity, actual fill timestamps, and a uniform equity time grid before Sharpe annualization.
- [ ] **OWNER ACTION — Production blocker** Connect a real exchange/platform implementation of `EmergencyExecutionAdapter` and record its factory path.
- [ ] **OWNER ACTION — Production blocker** Ensure live order construction passes current weights and all open orders to `PreTradeRiskVerifier`.

## B. Deployment documentation

- [x] **FIXED** Evidence bundle format and required identity fields are documented in `deployment_evidence.md`.
- [x] **FIXED** GitHub Environment approval is required for Production.
- [x] **FIXED** Production remains NO-GO without an approved `PROD-<digits>` ticket.
- [ ] **OWNER ACTION — Production blocker** Configure GitHub Environments named `shadow`, `canary`, and `production`; require designated reviewers for `production`.
- [ ] **OWNER ACTION — Production blocker** Create the workflow that produces and uploads the immutable `deployment-evidence` artifact.
- [ ] **OWNER ACTION** Define artifact retention and external immutable archive settings.

## C. Incident response

- [x] **FIXED** Runbook no longer claims that a command succeeded merely because a zero-weight instruction was emitted.
- [x] **FIXED** Runbook requires execution adapter, idempotency key, and post-action reconciliation evidence.
- [x] **FIXED** Rollback procedure distinguishes registry state from the serving process actually loading the model.
- [ ] **OWNER ACTION — Production blocker** Replace placeholder on-call and compliance contacts with real destinations.
- [ ] **OWNER ACTION — Production blocker** Define exchange-specific minimum residual position tolerance.
- [ ] **OWNER ACTION — Production blocker** Execute GameDay scenarios and attach evidence for disconnects, partial fills, cancellation races, process crash, artifact tampering, rollback, and emergency flatten.

## D. Compliance and auditability

- [x] **FIXED** Evidence names now match the deployed evidence bundle.
- [x] **FIXED** Retention language is classified as an internal policy pending legal determination, not an asserted universal legal requirement.
- [ ] **OWNER ACTION — Legal/Compliance blocker** Identify operating legal entity, customer/fund status, jurisdictions, exchanges, and applicable record-retention rules.
- [ ] **OWNER ACTION** Select immutable storage and access-control policy for audit artifacts.
- [ ] **OWNER ACTION** Define who may approve Production and who may override or close incidents.

## E. Research claims and economic validity

- [x] **HISTORICAL** `ARCHITECTURE.md` and `PROFIT_DESIGN.md` are classified as research records, not production authorization.
- [x] **FIXED** Documentation precedence is defined in `README.md`.
- [ ] **BLOCKED — Economic validation** No statement in the docs may be interpreted as proof of future profitability.
- [ ] **BLOCKED — Economic validation** Complete untouched real-data validation, Shadow evidence, and Canary evidence before Production.
- [ ] **OWNER ACTION** Add dated decision-log entries whenever thresholds, features, models, or validation datasets are changed.

## F. Required release evidence

Before marking Production `GO`, attach all of the following:

- [ ] Full CI result for the exact candidate Git commit.
- [ ] Candidate model SHA-256 and configuration/data hashes.
- [ ] Verified Shadow report and run ID.
- [ ] Verified drift and incident reports.
- [ ] Verified Canary report linked to the Shadow run.
- [ ] Replay-versus-live execution calibration report.
- [ ] GameDay report with recovery and reconciliation timings.
- [ ] Model decision log entry.
- [ ] Approved Production ticket and GitHub Environment approver.
- [ ] Real emergency execution adapter configuration and successful testnet flatten evidence.
