# M7 Production Blockers Clean Rebuild Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the M7 production-safety changes on a clean branch, replace obsolete tests with contract-based tests, and remove all diagnostic workflow noise.

**Architecture:** Reuse reviewed immutable Git blobs for the deployment gate, emergency flattening, pre-trade risk, replay simulator, and their focused tests. Apply only three integration patches to the current `main`: pass current weights/symbols from the environment, validate the post-threshold execution target in baselines, and update the legacy replay test to assert the uniform equity timeline. Keep the standard CI workflow unchanged.

**Tech Stack:** Python 3.12, pytest, Ruff, mypy, GitHub Actions.

## Global Constraints

- Never weaken production gate thresholds from evidence JSON.
- Never report emergency flatten success without verified zero open orders and residual positions within tolerance.
- Pre-trade order checks must evaluate execution deltas, not total target holdings.
- Replay Sharpe must use returns from a fixed-frequency mark-to-market equity series.
- Do not include temporary workflows, patch scripts, or repository-wide formatting-only changes in the final PR.

---

### Task 1: Restore reviewed safety implementations

**Files:**
- Modify: `.github/workflows/deploy.yml`
- Modify: `mars_lite/server/deployment_gate.py`
- Modify: `mars_lite/trading/guardrails.py`
- Modify: `mars_lite/trading/pre_trade_risk.py`
- Modify: `mars_lite/eval/replay_sim.py`

- [ ] Reuse the reviewed blobs from the superseded draft PR.
- [ ] Confirm no diagnostic workflow or temporary script is present.

### Task 2: Replace obsolete tests with contract tests

**Files:**
- Modify: `tests/test_deployment_gate.py`
- Modify: `tests/test_deployment_gate_adversarial.py`
- Modify: `tests/test_guardrails_cli.py`
- Modify: `tests/test_pre_trade_risk.py`
- Modify: `tests/test_replay_sim.py`
- Modify: `tests/test_adversarial_m4.py`

- [ ] Preserve adversarial checks for tampering, identity mismatch, path traversal, and active incidents.
- [ ] Test emergency flatten fail-closed and reconciliation behavior.
- [ ] Test one-sided pending-order risk and delta-based order checks.
- [ ] Replace the obsolete two-point no-order equity assertion with a uniform market-timeline assertion.

### Task 3: Wire execution-state inputs

**Files:**
- Modify: `mars_lite/env/portfolio_env.py`
- Modify: `mars_lite/learning/baselines.py`

- [ ] Pass `symbols` and `current_weights` from the environment.
- [ ] Apply `min_trade_delta` before baseline validation and validate `next_weights` against current weights.
- [ ] Add integration tests proving small rebalance deltas are evaluated using post-cost/current portfolio state without brittle exact-value assertions.

### Task 4: Align operational documentation

**Files:**
- Create/Modify: `docs/DOCUMENTATION_AUDIT_CHECKLIST.md`
- Create/Modify: `docs/README.md`
- Create/Modify: `docs/deployment_evidence.md`
- Create/Modify: `docs/model_decision_log.md`
- Modify: `docs/runbook_incident_response.md`
- Modify: `docs/runbook_model_rollback.md`
- Modify: `docs/runbook_compliance.md`

- [ ] Separate verified implementation from owner-supplied production blockers.
- [ ] Keep real adapter, environment approvals, contacts, legal retention, and GameDay evidence unchecked until supplied.

### Task 5: Verification

- [ ] Run `ruff check` and `ruff format --check` on changed Python files.
- [ ] Run `mypy mars_lite`.
- [ ] Run focused M7 and adversarial tests.
- [ ] Run the repository-wide pytest suite with the existing 70% coverage gate.
- [ ] Open a replacement draft PR only after the clean branch contains no temporary files.
